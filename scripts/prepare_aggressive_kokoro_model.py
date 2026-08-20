#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "app/src/main/assets/kokoro_vi"
WORK_DIR = ROOT / "build/aggressive_kokoro_model"
OUT_DIR = ROOT / "build/model-only-output"
FP32 = ASSET_DIR / "kokoro_vi.onnx"
VOICE = ASSET_DIR / "voicepacks/diem_trinh.f32le"
PREPARED = WORK_DIR / "kokoro_prepared.onnx"
LINEAR_INT8 = WORK_DIR / "kokoro_linear_int8.onnx"
FINAL = OUT_DIR / "kokoro_vi.onnx"
REPORT = OUT_DIR / "optimization_report.json"

WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not FP32.is_file():
    raise SystemExit(f"Missing FP32 model: {FP32}")
if not VOICE.is_file():
    raise SystemExit(f"Missing voicepack: {VOICE}")

# This optimizer intentionally pursues a materially smaller/faster model rather
# than a token selective experiment. The duration path is protected structurally:
# every ancestor of the graph's Round operation(s), plus predictor/prosody scopes,
# remains FP32. Large non-duration MatMul/Gemm/LSTM/Gather weights are dynamically
# quantized. Large non-duration Conv weights are then statically quantized using
# representative calibration feeds so ORT can execute QLinearConv instead of
# paying dynamic activation-quantization overhead on every convolution.
#
# The final artifact must satisfy ALL of these gates:
#   * >=25% smaller than the original 325 MB model
#   * exact waveform sample count on 21/50/71/110-token probes
#   * deterministic waveform correlation >=0.985 on every probe
#   * deterministic SNR >=16 dB on every probe
#   * normalized RMSE <=0.16 on every probe
#   * model loads and runs in ONNX Runtime 1.17.1
# If no candidate satisfies the gates, the workflow fails and no misleading
# "optimized" model is published.

MIN_SIZE_REDUCTION = 0.25
MIN_CORRELATION = 0.985
MIN_SNR_DB = 16.0
MAX_NORMALIZED_RMSE = 0.16
TOKEN_COUNTS = (21, 50, 71, 110)


def node_label(node: onnx.NodeProto, index: int) -> str:
    return node.name or f"__unnamed_{index}_{node.op_type}"


def initializer_bytes(init: onnx.TensorProto) -> int:
    # raw_data is populated for this model, but numpy fallback makes this robust.
    if init.raw_data:
        return len(init.raw_data)
    return int(numpy_helper.to_array(init).nbytes)


def build_dependency_protection(model: onnx.ModelProto) -> tuple[set[str], dict]:
    producer: dict[str, tuple[int, onnx.NodeProto]] = {}
    for idx, node in enumerate(model.graph.node):
        for out in node.output:
            if out:
                producer[out] = (idx, node)

    round_nodes = [(idx, n) for idx, n in enumerate(model.graph.node) if n.op_type == "Round"]
    if not round_nodes:
        raise SystemExit("No Round node found; cannot structurally protect Kokoro duration path")

    protected: set[str] = set()
    stack: list[str] = []
    for idx, node in round_nodes:
        protected.add(node_label(node, idx))
        stack.extend([x for x in node.input if x])

    while stack:
        tensor = stack.pop()
        item = producer.get(tensor)
        if item is None:
            continue
        idx, node = item
        label = node_label(node, idx)
        if label in protected:
            continue
        protected.add(label)
        stack.extend([x for x in node.input if x])

    # Extra semantic guardrail: prosody/predictor scopes may feed F0/noise rather
    # than the duration Round. Keep them FP32 to protect rhythm and voice character.
    semantic_tokens = (
        "predictor", "duration", "prosody", "duration_proj", "align",
        "repeat_interleave", "f0", "noise_pred",
    )
    semantic_added = 0
    for idx, node in enumerate(model.graph.node):
        text = " ".join([node.name, *node.input, *node.output]).lower()
        if any(tok in text for tok in semantic_tokens):
            label = node_label(node, idx)
            if label not in protected:
                protected.add(label)
                semantic_added += 1

    return protected, {
        "roundNodeCount": len(round_nodes),
        "protectedNodeCount": len(protected),
        "semanticProtectionAdded": semantic_added,
    }


def clone_initializer(array: np.ndarray, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(array), name=name)


def rewrite_safe_gemm_to_matmul(model: onnx.ModelProto, protected: set[str]) -> tuple[onnx.ModelProto, dict[str, str], dict]:
    """Rewrite every non-protected Gemm exactly as MatMul(+Add).

    ORT 1.17.1 dynamic IntegerOps quantization does not quantize Gemm directly.
    This rewrite is algebraically exact and allows the large weights to use
    MatMulInteger. New weight/bias initializers are created whenever Gemm attrs
    require transpose/scaling, so shared original weights remain untouched.
    """
    initializer_by_name = {x.name: x for x in model.graph.initializer}
    new_initializers: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []
    mapping: dict[str, str] = {}
    rewritten = 0
    skipped: list[dict] = []

    for idx, node in enumerate(model.graph.node):
        label = node_label(node, idx)
        if node.op_type != "Gemm" or label in protected:
            new_nodes.append(node)
            continue

        if len(node.input) < 2 or node.input[1] not in initializer_by_name:
            skipped.append({"name": label, "reason": "weight is not a constant initializer"})
            new_nodes.append(node)
            continue

        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        alpha = float(attrs.get("alpha", 1.0))
        beta = float(attrs.get("beta", 1.0))
        trans_a = int(attrs.get("transA", 0))
        trans_b = int(attrs.get("transB", 0))
        if trans_a not in (0, 1) or trans_b not in (0, 1):
            skipped.append({"name": label, "reason": f"unsupported transpose attrs transA={trans_a}, transB={trans_b}"})
            new_nodes.append(node)
            continue

        weight = numpy_helper.to_array(initializer_by_name[node.input[1]]).astype(np.float32, copy=False)
        if trans_b:
            weight = weight.T
        if not math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-12):
            weight = weight * np.float32(alpha)
        new_weight_name = f"{node.input[1]}__opt_{rewritten}_matmul_weight"
        new_initializers.append(clone_initializer(weight, new_weight_name))

        input_a = node.input[0]
        if trans_a:
            transposed_a = f"{node.output[0]}__opt_transA"
            new_nodes.append(onnx.helper.make_node(
                "Transpose", [input_a], [transposed_a], name=f"{label}__opt_transA", perm=[1, 0]
            ))
            input_a = transposed_a

        has_bias = len(node.input) >= 3 and bool(node.input[2])
        matmul_output = f"{node.output[0]}__opt_pre_bias" if has_bias else node.output[0]
        matmul_name = f"{label}__opt_matmul"
        new_nodes.append(onnx.helper.make_node(
            "MatMul", [input_a, new_weight_name], [matmul_output], name=matmul_name
        ))

        if has_bias:
            bias_name = node.input[2]
            if bias_name in initializer_by_name and not math.isclose(beta, 1.0, rel_tol=0.0, abs_tol=1e-12):
                bias = numpy_helper.to_array(initializer_by_name[bias_name]).astype(np.float32, copy=False)
                new_bias_name = f"{bias_name}__opt_{rewritten}_bias"
                new_initializers.append(clone_initializer(bias * np.float32(beta), new_bias_name))
                bias_name = new_bias_name
            elif not math.isclose(beta, 1.0, rel_tol=0.0, abs_tol=1e-12):
                # Non-constant bias scaling would need an extra Mul. Keep exactness
                # by leaving this uncommon Gemm untouched.
                new_nodes.pop()
                if trans_a:
                    new_nodes.pop()
                new_initializers.pop()  # weight created for this node
                skipped.append({"name": label, "reason": "non-constant bias with beta != 1"})
                new_nodes.append(node)
                continue
            new_nodes.append(onnx.helper.make_node(
                "Add", [matmul_output, bias_name], [node.output[0]], name=f"{label}__opt_bias_add"
            ))

        mapping[label] = matmul_name
        rewritten += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    onnx.checker.check_model(model)
    try:
        model = onnx.shape_inference.infer_shapes(model, check_type=False, strict_mode=False, data_prop=False)
    except Exception as exc:
        raise SystemExit(f"Shape inference failed after Gemm rewrite: {exc}") from exc
    onnx.checker.check_model(model)
    return model, mapping, {"rewrittenGemm": rewritten, "skippedGemm": skipped}


def force_deterministic_random(src: Path, dst: Path, seed: float = 1729.0) -> None:
    model = onnx.load(str(src), load_external_data=True)

    def patch_graph(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            if node.op_type in {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"}:
                attrs = [a for a in node.attribute if a.name != "seed"]
                del node.attribute[:]
                node.attribute.extend(attrs)
                node.attribute.append(onnx.helper.make_attribute("seed", float(seed)))
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    patch_graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for sub in attr.graphs:
                        patch_graph(sub)

    patch_graph(model.graph)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def load_voice() -> np.ndarray:
    voice = np.fromfile(VOICE, dtype="<f4")
    if voice.size != 510 * 256:
        raise SystemExit(f"Unexpected voicepack float count: {voice.size}")
    return voice.reshape(510, 256)


VOICE_MATRIX = load_voice()


def make_feed(token_count: int, pattern: int = 0, speed_value: float = 1.0) -> dict[str, np.ndarray]:
    if token_count < 3:
        raise ValueError(token_count)
    ids = np.empty((1, token_count), dtype=np.int64)
    ids[0, 0] = 0
    ids[0, -1] = 0
    # The Kokoro exported vocabulary in this project has >100 usable IDs. Use a
    # deterministic spread rather than repeated token 1 so calibration sees a
    # broader activation range while remaining model-valid.
    for j in range(1, token_count - 1):
        ids[0, j] = 1 + ((j * (7 + pattern * 2) + pattern * 17) % 100)
    style_row = min(509, max(0, token_count - 3 + pattern))
    style = VOICE_MATRIX[style_row:style_row + 1].astype(np.float32, copy=False)
    speed = np.asarray(speed_value, dtype=np.float32)
    return {"input_ids": ids, "ref_s": style, "speed": speed}


def validation_feeds() -> list[dict[str, np.ndarray]]:
    return [make_feed(n, pattern=i % 3) for i, n in enumerate(TOKEN_COUNTS)]


def calibration_feeds() -> list[dict[str, np.ndarray]]:
    out: list[dict[str, np.ndarray]] = []
    for pattern in range(3):
        for i, n in enumerate(TOKEN_COUNTS):
            speed = (0.92, 1.0, 1.08)[pattern]
            out.append(make_feed(n, pattern=(pattern + i) % 3, speed_value=speed))
    return out


class FeedReader(CalibrationDataReader):
    def __init__(self, feeds: Iterable[dict[str, np.ndarray]]):
        self._feeds = list(feeds)
        self._iter = iter(self._feeds)

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._feeds)


def run_waveforms(model_path: Path, feeds: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs: list[np.ndarray] = []
    for feed in feeds:
        audio = np.asarray(session.run(["waveform"], feed)[0], dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError(f"Invalid waveform from {model_path.name}")
        outputs.append(audio)
    return outputs


def compare_waveforms(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict:
    if len(reference) != len(candidate):
        raise ValueError("Waveform case count mismatch")
    cases: list[dict] = []
    exact_duration = True
    min_corr = 1.0
    min_snr = float("inf")
    max_nrmse = 0.0
    for i, (ref, cand) in enumerate(zip(reference, candidate)):
        same_samples = ref.size == cand.size
        exact_duration = exact_duration and same_samples
        if not same_samples:
            cases.append({
                "tokens": TOKEN_COUNTS[i],
                "referenceSamples": int(ref.size),
                "candidateSamples": int(cand.size),
                "durationExact": False,
            })
            min_corr = -1.0
            min_snr = -999.0
            max_nrmse = 999.0
            continue
        err = ref - cand
        rmse = float(np.sqrt(np.mean(err * err)))
        ref_rms = float(np.sqrt(np.mean(ref * ref)))
        nrmse = rmse / max(ref_rms, 1e-12)
        if float(np.std(ref)) > 1e-12 and float(np.std(cand)) > 1e-12:
            corr = float(np.corrcoef(ref, cand)[0, 1])
        else:
            corr = 1.0 if np.allclose(ref, cand) else 0.0
        snr = float(20.0 * math.log10(max(ref_rms, 1e-12) / max(rmse, 1e-12)))
        mae = float(np.mean(np.abs(err)))
        min_corr = min(min_corr, corr)
        min_snr = min(min_snr, snr)
        max_nrmse = max(max_nrmse, nrmse)
        cases.append({
            "tokens": TOKEN_COUNTS[i],
            "samples": int(ref.size),
            "durationExact": True,
            "correlation": corr,
            "snrDb": snr,
            "normalizedRmse": nrmse,
            "mae": mae,
        })
    return {
        "durationExactAll": exact_duration,
        "minCorrelation": min_corr,
        "minSnrDb": min_snr,
        "maxNormalizedRmse": max_nrmse,
        "cases": cases,
    }


def quality_pass(metrics: dict) -> bool:
    return (
        metrics["durationExactAll"]
        and metrics["minCorrelation"] >= MIN_CORRELATION
        and metrics["minSnrDb"] >= MIN_SNR_DB
        and metrics["maxNormalizedRmse"] <= MAX_NORMALIZED_RMSE
    )


def model_node_weight_bytes(model: onnx.ModelProto) -> dict[str, int]:
    init = {x.name: x for x in model.graph.initializer}
    result: dict[str, int] = {}
    for idx, node in enumerate(model.graph.node):
        label = node_label(node, idx)
        names: list[str] = []
        if node.op_type in {"MatMul", "Gemm", "Conv", "Gather"}:
            if len(node.input) > 1 and node.op_type != "Gather":
                names = [node.input[1]]
            elif node.op_type == "Gather" and node.input:
                names = [node.input[0]]
        elif node.op_type == "LSTM":
            names = [x for x in node.input[1:3] if x]
        total = sum(initializer_bytes(init[n]) for n in names if n in init)
        result[label] = total
    return result


def select_dynamic_targets(model: onnx.ModelProto, protected: set[str], gemm_mapping: dict[str, str]) -> tuple[list[str], dict]:
    weights = model_node_weight_bytes(model)
    targets: list[str] = []
    by_op = Counter()
    bytes_by_op = Counter()

    # Protected labels refer to the original graph. Rewritten safe Gemm nodes have
    # new names and are intentionally quantizable. Other nodes retain their names.
    protected_after = set(protected)
    protected_after.difference_update(gemm_mapping.keys())

    for idx, node in enumerate(model.graph.node):
        label = node_label(node, idx)
        if label in protected_after:
            continue
        if node.op_type not in {"MatMul", "LSTM", "Gather"}:
            continue
        weight_bytes = weights.get(label, 0)
        # Quantizing tiny ops often adds more activation-quantization overhead than
        # useful compute reduction. Focus on material weights.
        minimum = 64 * 1024 if node.op_type == "MatMul" else 256 * 1024
        if weight_bytes < minimum:
            continue
        targets.append(label)
        by_op[node.op_type] += 1
        bytes_by_op[node.op_type] += weight_bytes

    return targets, {
        "targetCount": len(targets),
        "targetsByOp": dict(by_op),
        "targetWeightBytesByOp": dict(bytes_by_op),
        "targetWeightBytesTotal": int(sum(bytes_by_op.values())),
    }


def select_conv_targets(model: onnx.ModelProto, protected: set[str], minimum_weight_bytes: int) -> tuple[list[str], int]:
    weights = model_node_weight_bytes(model)
    names: list[str] = []
    total = 0
    for idx, node in enumerate(model.graph.node):
        label = node_label(node, idx)
        if node.op_type != "Conv" or label in protected:
            continue
        wb = weights.get(label, 0)
        if wb < minimum_weight_bytes:
            continue
        names.append(label)
        total += wb
    return names, total


def benchmark_model(model_path: Path, feeds: list[dict[str, np.ndarray]]) -> dict:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    # Short warmup first.
    session.run(["waveform"], feeds[0])
    times: list[float] = []
    for feed in feeds:
        started = time.perf_counter()
        session.run(["waveform"], feed)
        times.append((time.perf_counter() - started) * 1000.0)
    weighted = times[0] * 4 + times[1] * 3 + times[2] * 2 + times[3]
    return {"caseMs": times, "weightedScoreMs": weighted}


# Load original model and protect duration/prosody structurally.
original = onnx.load(str(FP32), load_external_data=True)
onnx.checker.check_model(original)
protected, protection_report = build_dependency_protection(original)
fp32_bytes = FP32.stat().st_size

# Rewriting all non-protected Gemm nodes creates a single mathematically equivalent
# prepared graph. It does not quantize anything yet.
prepared_model, gemm_mapping, rewrite_report = rewrite_safe_gemm_to_matmul(original, protected)
onnx.save(prepared_model, str(PREPARED))

# Deterministic equivalence gate for the rewrite itself.
ref_det = WORK_DIR / "fp32_deterministic.onnx"
prepared_det = WORK_DIR / "prepared_deterministic.onnx"
force_deterministic_random(FP32, ref_det)
force_deterministic_random(PREPARED, prepared_det)
feeds = validation_feeds()
reference_waves = run_waveforms(ref_det, feeds)
prepared_waves = run_waveforms(prepared_det, feeds)
rewrite_metrics = compare_waveforms(reference_waves, prepared_waves)
if not rewrite_metrics["durationExactAll"] or rewrite_metrics["maxNormalizedRmse"] > 0.0005:
    raise SystemExit("Gemm rewrite failed deterministic equivalence gate: " + json.dumps(rewrite_metrics))

# Stage 1: dynamic INT8 for large non-duration MatMul/LSTM/Gather weights.
dynamic_targets, dynamic_report = select_dynamic_targets(prepared_model, protected, gemm_mapping)
if not dynamic_targets:
    raise SystemExit("No material non-duration dynamic INT8 targets were found")
if LINEAR_INT8.exists():
    LINEAR_INT8.unlink()
quantize_dynamic(
    model_input=str(PREPARED),
    model_output=str(LINEAR_INT8),
    op_types_to_quantize=["MatMul", "LSTM", "Gather"],
    nodes_to_quantize=dynamic_targets,
    per_channel=True,
    reduce_range=False,
    weight_type=QuantType.QInt8,
    extra_options={"MatMulConstBOnly": True},
)
linear_model = onnx.load(str(LINEAR_INT8), load_external_data=True)
onnx.checker.check_model(linear_model)

# Calibration uses actual Diem Trinh style vectors and varied token patterns.
calibration = calibration_feeds()

# Stage 2 candidate sweep: quantize progressively smaller Conv weights. Large-first
# thresholds prevent tiny final/output convolutions from being quantized unless a
# more conservative candidate cannot meet the compression target.
conv_thresholds = [
    1024 * 1024,
    512 * 1024,
    256 * 1024,
    128 * 1024,
    64 * 1024,
]

candidate_reports: list[dict] = []
passing_candidates: list[tuple[Path, dict]] = []

for threshold in conv_thresholds:
    conv_names, conv_weight_bytes = select_conv_targets(prepared_model, protected, threshold)
    if not conv_names:
        continue
    candidate = WORK_DIR / f"candidate_conv_{threshold}.onnx"
    if candidate.exists():
        candidate.unlink()
    try:
        quantize_static(
            model_input=str(LINEAR_INT8),
            model_output=str(candidate),
            calibration_data_reader=FeedReader(calibration),
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=["Conv"],
            nodes_to_quantize=conv_names,
            per_channel=True,
            reduce_range=False,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
            },
        )
        candidate_model = onnx.load(str(candidate), load_external_data=True)
        onnx.checker.check_model(candidate_model)
        ops = Counter(n.op_type for n in candidate_model.graph.node)

        deterministic = WORK_DIR / f"candidate_conv_{threshold}_det.onnx"
        force_deterministic_random(candidate, deterministic)
        waves = run_waveforms(deterministic, feeds)
        quality = compare_waveforms(reference_waves, waves)
        size_bytes = candidate.stat().st_size
        reduction = 1.0 - (size_bytes / fp32_bytes)
        qpass = quality_pass(quality)
        compression_pass = reduction >= MIN_SIZE_REDUCTION

        # CI-host CPU speed is not the target phone's ARM CPU, so record it as a
        # diagnostic only. It is deliberately not a hard gate.
        speed_diag = benchmark_model(candidate, feeds)

        report = {
            "convMinimumWeightBytes": threshold,
            "convTargetCount": len(conv_names),
            "convTargetWeightBytes": conv_weight_bytes,
            "sizeBytes": size_bytes,
            "sizeReductionPercent": reduction * 100.0,
            "opsAfter": dict(sorted(ops.items())),
            "quality": quality,
            "qualityPass": qpass,
            "compressionPass": compression_pass,
            "ciHostSpeed": speed_diag,
            "eligible": bool(qpass and compression_pass),
        }
        candidate_reports.append(report)
        if qpass and compression_pass:
            passing_candidates.append((candidate, report))
    except Exception as exc:
        candidate_reports.append({
            "convMinimumWeightBytes": threshold,
            "error": repr(exc),
            "eligible": False,
        })

# Also evaluate the dynamic-linear-only stage so the report shows exactly why a
# tiny improvement is rejected rather than silently accepted.
linear_det = WORK_DIR / "linear_int8_deterministic.onnx"
force_deterministic_random(LINEAR_INT8, linear_det)
linear_quality = compare_waveforms(reference_waves, run_waveforms(linear_det, feeds))
linear_size = LINEAR_INT8.stat().st_size
linear_reduction = 1.0 - linear_size / fp32_bytes
linear_ops = Counter(n.op_type for n in linear_model.graph.node)

if not passing_candidates:
    failure_report = {
        "status": "rejected_no_material_candidate",
        "runtime": ort.__version__,
        "fp32Bytes": fp32_bytes,
        "requiredMinimumSizeReductionPercent": MIN_SIZE_REDUCTION * 100.0,
        "qualityThresholds": {
            "minCorrelation": MIN_CORRELATION,
            "minSnrDb": MIN_SNR_DB,
            "maxNormalizedRmse": MAX_NORMALIZED_RMSE,
            "durationExact": True,
        },
        "protection": protection_report,
        "rewrite": rewrite_report,
        "rewriteQuality": rewrite_metrics,
        "dynamicStage": {
            **dynamic_report,
            "sizeBytes": linear_size,
            "sizeReductionPercent": linear_reduction * 100.0,
            "opsAfter": dict(sorted(linear_ops.items())),
            "quality": linear_quality,
        },
        "convCandidates": candidate_reports,
    }
    REPORT.write_text(json.dumps(failure_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raise SystemExit("No candidate achieved both >=25% compression and the quality gates. See optimization_report.json")

# Prefer the fastest CI-host candidate among eligible models, but only after the
# hard quality/compression gates. Tie-break toward smaller files. This is a sane
# proxy while the real ARM speed is ultimately measured on-device.
passing_candidates.sort(key=lambda item: (
    item[1]["ciHostSpeed"]["weightedScoreMs"],
    item[1]["sizeBytes"],
))
selected_path, selected_report = passing_candidates[0]
shutil.copy2(selected_path, FINAL)

final_model = onnx.load(str(FINAL), load_external_data=True)
onnx.checker.check_model(final_model)
final_ops = Counter(n.op_type for n in final_model.graph.node)

# Final load/run verification with the exact semantic runtime used by Android.
final_session = ort.InferenceSession(str(FINAL), providers=["CPUExecutionProvider"])
for feed in feeds:
    audio = np.asarray(final_session.run(["waveform"], feed)[0], dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise SystemExit("Final model verification returned invalid waveform")

final_size = FINAL.stat().st_size
final_reduction = 1.0 - final_size / fp32_bytes
sha256 = __import__("hashlib").sha256(FINAL.read_bytes()).hexdigest()

report = {
    "status": "accepted_material_optimization",
    "runtime": ort.__version__,
    "artifact": "kokoro_vi.onnx",
    "sha256": sha256,
    "fp32Bytes": fp32_bytes,
    "optimizedBytes": final_size,
    "sizeReductionPercent": final_reduction * 100.0,
    "qualityThresholds": {
        "minCorrelation": MIN_CORRELATION,
        "minSnrDb": MIN_SNR_DB,
        "maxNormalizedRmse": MAX_NORMALIZED_RMSE,
        "durationExact": True,
    },
    "durationAndProsodyProtection": protection_report,
    "gemmRewrite": rewrite_report,
    "gemmRewriteQuality": rewrite_metrics,
    "dynamicStage": {
        **dynamic_report,
        "sizeBytes": linear_size,
        "sizeReductionPercent": linear_reduction * 100.0,
        "opsAfter": dict(sorted(linear_ops.items())),
        "quality": linear_quality,
    },
    "selectedCandidate": selected_report,
    "allConvCandidates": candidate_reports,
    "finalOps": dict(sorted(final_ops.items())),
    "note": "CI-host timing is diagnostic only; real speed must be measured on the target ARM64 phone. The final model is a drop-in ONNX model with the same inputs/outputs as the original and was accepted only after multi-length duration and deterministic waveform quality gates.",
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"FINAL_MODEL={FINAL}")
print(f"FINAL_BYTES={final_size}")
print(f"SIZE_REDUCTION_PERCENT={final_reduction * 100.0:.2f}")
print(f"SHA256={sha256}")
