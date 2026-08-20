#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

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
WORK = ROOT / "build/aggressive_kokoro_v2"
OUT = ROOT / "build/model-only-output"
FP32 = ASSET_DIR / "kokoro_vi.onnx"
VOICE = ASSET_DIR / "voicepacks/diem_trinh.f32le"
PREPARED = WORK / "prepared.onnx"
LINEAR = WORK / "linear_int8.onnx"
FINAL = OUT / "kokoro_vi.onnx"
REPORT = OUT / "optimization_report.json"

WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

MIN_REDUCTION = 0.25
MIN_CORR = 0.985
MIN_SNR = 16.0
MAX_NRMSE = 0.16
TOKEN_COUNTS = (21, 50, 71, 110)

if not FP32.is_file() or FP32.stat().st_size < 300_000_000:
    raise SystemExit("Missing/invalid original kokoro_vi.onnx")
if not VOICE.is_file() or VOICE.stat().st_size != 510 * 256 * 4:
    raise SystemExit("Missing/invalid Diem Trinh voicepack")


def label(node: onnx.NodeProto, idx: int) -> str:
    return node.name or f"__node_{idx}_{node.op_type}"


def tensor_bytes(t: onnx.TensorProto) -> int:
    return len(t.raw_data) if t.raw_data else int(numpy_helper.to_array(t).nbytes)


def recursive_inputs(graph: onnx.GraphProto) -> set[str]:
    used: set[str] = set()
    for node in graph.node:
        used.update(x for x in node.input if x)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                used.update(recursive_inputs(attr.g))
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    used.update(recursive_inputs(g))
    return used


def prune_unused_initializers(model: onnx.ModelProto) -> int:
    used = recursive_inputs(model.graph)
    before = len(model.graph.initializer)
    kept = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return before - len(kept)


def duration_protection(model: onnx.ModelProto) -> tuple[set[str], dict]:
    producer: dict[str, tuple[int, onnx.NodeProto]] = {}
    for idx, node in enumerate(model.graph.node):
        for out in node.output:
            if out:
                producer[out] = (idx, node)

    rounds = [(i, n) for i, n in enumerate(model.graph.node) if n.op_type == "Round"]
    if not rounds:
        raise SystemExit("Cannot locate duration Round node")

    protected: set[str] = set()
    stack: list[str] = []
    for idx, node in rounds:
        protected.add(label(node, idx))
        stack.extend(x for x in node.input if x)
    while stack:
        t = stack.pop()
        p = producer.get(t)
        if p is None:
            continue
        idx, node = p
        name = label(node, idx)
        if name in protected:
            continue
        protected.add(name)
        stack.extend(x for x in node.input if x)

    semantic = (
        "predictor", "duration", "duration_proj", "prosody", "align",
        "repeat_interleave", "noise_pred", "/f0", "f0_",
    )
    extra = 0
    for idx, node in enumerate(model.graph.node):
        haystack = " ".join([node.name, *node.input, *node.output]).lower()
        if any(k in haystack for k in semantic):
            name = label(node, idx)
            if name not in protected:
                protected.add(name)
                extra += 1
    return protected, {
        "roundNodes": len(rounds),
        "protectedNodes": len(protected),
        "semanticExtra": extra,
    }


def rewrite_safe_gemm(model: onnx.ModelProto, protected: set[str]) -> tuple[onnx.ModelProto, dict]:
    inits = {x.name: x for x in model.graph.initializer}
    added_inits: list[onnx.TensorProto] = []
    out_nodes: list[onnx.NodeProto] = []
    converted: list[str] = []
    skipped: list[dict] = []

    for idx, node in enumerate(model.graph.node):
        name = label(node, idx)
        if node.op_type != "Gemm" or name in protected:
            out_nodes.append(node)
            continue
        if len(node.input) < 2 or node.input[1] not in inits:
            skipped.append({"name": name, "reason": "non-constant weight"})
            out_nodes.append(node)
            continue

        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        alpha = float(attrs.get("alpha", 1.0))
        beta = float(attrs.get("beta", 1.0))
        trans_a = int(attrs.get("transA", 0))
        trans_b = int(attrs.get("transB", 0))
        if trans_a not in (0, 1) or trans_b not in (0, 1):
            skipped.append({"name": name, "reason": "unsupported transpose attributes"})
            out_nodes.append(node)
            continue

        w = numpy_helper.to_array(inits[node.input[1]]).astype(np.float32, copy=False)
        if trans_b:
            w = w.T
        if not math.isclose(alpha, 1.0, abs_tol=1e-12):
            w = w * np.float32(alpha)
        wname = f"{node.input[1]}__v2_{len(converted)}"
        added_inits.append(numpy_helper.from_array(np.ascontiguousarray(w), name=wname))

        a = node.input[0]
        if trans_a:
            a2 = f"{node.output[0]}__v2_transA"
            out_nodes.append(onnx.helper.make_node("Transpose", [a], [a2], name=f"{name}__v2_transA", perm=[1, 0]))
            a = a2

        has_bias = len(node.input) > 2 and bool(node.input[2])
        mm_out = f"{node.output[0]}__v2_prebias" if has_bias else node.output[0]
        mm_name = f"{name}__v2_matmul"
        out_nodes.append(onnx.helper.make_node("MatMul", [a, wname], [mm_out], name=mm_name))

        if has_bias:
            bname = node.input[2]
            if not math.isclose(beta, 1.0, abs_tol=1e-12):
                if bname not in inits:
                    # Restore original Gemm if exact beta scaling cannot be folded.
                    if trans_a:
                        out_nodes.pop()
                    out_nodes.pop()
                    added_inits.pop()
                    skipped.append({"name": name, "reason": "non-constant bias with beta != 1"})
                    out_nodes.append(node)
                    continue
                b = numpy_helper.to_array(inits[bname]).astype(np.float32, copy=False) * np.float32(beta)
                b2 = f"{bname}__v2_{len(converted)}"
                added_inits.append(numpy_helper.from_array(np.ascontiguousarray(b), name=b2))
                bname = b2
            out_nodes.append(onnx.helper.make_node("Add", [mm_out, bname], [node.output[0]], name=f"{name}__v2_bias"))
        converted.append(mm_name)

    del model.graph.node[:]
    model.graph.node.extend(out_nodes)
    model.graph.initializer.extend(added_inits)
    pruned = prune_unused_initializers(model)
    model = onnx.shape_inference.infer_shapes(model, check_type=False, strict_mode=False, data_prop=False)
    onnx.checker.check_model(model)
    return model, {"convertedGemmToMatMul": len(converted), "convertedMatMulNames": converted, "skipped": skipped, "prunedInitializers": pruned}


def patch_random(src: Path, dst: Path, seed: float = 1729.0) -> None:
    m = onnx.load(str(src), load_external_data=True)

    def walk(g: onnx.GraphProto) -> None:
        for n in g.node:
            if n.op_type in {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"}:
                attrs = [a for a in n.attribute if a.name != "seed"]
                del n.attribute[:]
                n.attribute.extend(attrs)
                n.attribute.append(onnx.helper.make_attribute("seed", float(seed)))
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    walk(a.g)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sg in a.graphs:
                        walk(sg)

    walk(m.graph)
    prune_unused_initializers(m)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


voice = np.fromfile(VOICE, dtype="<f4").reshape(510, 256)


def feed(n: int, pattern: int = 0, speed: float = 1.0) -> dict[str, np.ndarray]:
    ids = np.empty((1, n), dtype=np.int64)
    ids[0, 0] = 0
    ids[0, -1] = 0
    for j in range(1, n - 1):
        ids[0, j] = 1 + ((j * (7 + 2 * pattern) + 13 * pattern) % 100)
    row = min(509, max(0, n - 3 + pattern))
    return {
        "input_ids": ids,
        "ref_s": voice[row:row + 1].astype(np.float32, copy=False),
        "speed": np.asarray(speed, dtype=np.float32),
    }


validation = [feed(n, i % 3) for i, n in enumerate(TOKEN_COUNTS)]
calibration = [feed(n, (p + i) % 3, (0.92, 1.0, 1.08)[p]) for p in range(3) for i, n in enumerate(TOKEN_COUNTS)]


class Reader(CalibrationDataReader):
    def __init__(self, rows):
        self.rows = list(rows)
        self.rewind()
    def get_next(self):
        return next(self.it, None)
    def rewind(self):
        self.it = iter(self.rows)


def run(path: Path, rows) -> list[np.ndarray]:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = []
    for x in rows:
        y = np.asarray(s.run(["waveform"], x)[0], dtype=np.float32).reshape(-1)
        if y.size == 0 or not np.isfinite(y).all():
            raise RuntimeError(f"Invalid waveform from {path}")
        out.append(y)
    return out


def metrics(refs, cands) -> dict:
    cases = []
    exact = True
    min_corr = 1.0
    min_snr = 999.0
    max_nrmse = 0.0
    for n, a, b in zip(TOKEN_COUNTS, refs, cands):
        if a.size != b.size:
            exact = False
            cases.append({"tokens": n, "durationExact": False, "referenceSamples": int(a.size), "candidateSamples": int(b.size)})
            min_corr, min_snr, max_nrmse = -1.0, -999.0, 999.0
            continue
        e = a - b
        rmse = float(np.sqrt(np.mean(e * e)))
        rms = float(np.sqrt(np.mean(a * a)))
        nrmse = rmse / max(rms, 1e-12)
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else 0.0
        snr = float(20 * math.log10(max(rms, 1e-12) / max(rmse, 1e-12)))
        min_corr = min(min_corr, corr)
        min_snr = min(min_snr, snr)
        max_nrmse = max(max_nrmse, nrmse)
        cases.append({"tokens": n, "durationExact": True, "samples": int(a.size), "correlation": corr, "snrDb": snr, "normalizedRmse": nrmse})
    return {"durationExactAll": exact, "minCorrelation": min_corr, "minSnrDb": min_snr, "maxNormalizedRmse": max_nrmse, "cases": cases}


def quality_ok(m: dict) -> bool:
    return m["durationExactAll"] and m["minCorrelation"] >= MIN_CORR and m["minSnrDb"] >= MIN_SNR and m["maxNormalizedRmse"] <= MAX_NRMSE


def node_weight_bytes(model: onnx.ModelProto, node: onnx.NodeProto) -> int:
    init = {x.name: x for x in model.graph.initializer}
    if node.op_type in {"MatMul", "Conv"} and len(node.input) > 1 and node.input[1] in init:
        return tensor_bytes(init[node.input[1]])
    return 0


def bench(path: Path) -> dict:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    s.run(["waveform"], validation[0])
    ms = []
    for row in validation:
        t0 = time.perf_counter()
        s.run(["waveform"], row)
        ms.append((time.perf_counter() - t0) * 1000.0)
    return {"caseMs": ms, "weightedScoreMs": ms[0] * 4 + ms[1] * 3 + ms[2] * 2 + ms[3]}


original = onnx.load(str(FP32), load_external_data=True)
onnx.checker.check_model(original)
protected, protection_report = duration_protection(original)
prepared, rewrite_report = rewrite_safe_gemm(original, protected)
onnx.save(prepared, str(PREPARED))

# Exact rewrite validation before any quantization.
ref_det = WORK / "ref_det.onnx"
prep_det = WORK / "prep_det.onnx"
patch_random(FP32, ref_det)
patch_random(PREPARED, prep_det)
refs = run(ref_det, validation)
rewrite_metrics = metrics(refs, run(prep_det, validation))
if not rewrite_metrics["durationExactAll"] or rewrite_metrics["maxNormalizedRmse"] > 0.0005:
    raise SystemExit("Gemm rewrite is not numerically equivalent: " + json.dumps(rewrite_metrics))

# Dynamic INT8: only material top-level MatMul weights. ORT 1.17.1 can corrupt
# Loop/Sequence topology when LSTM/Gather are included in this graph, so those ops
# are deliberately left FP32. This also avoids spending quant/dequant overhead on
# small recurrent/lookup work.
matmul_targets = []
matmul_weight_bytes = 0
for idx, n in enumerate(prepared.graph.node):
    name = label(n, idx)
    if n.op_type != "MatMul" or name in protected:
        continue
    wb = node_weight_bytes(prepared, n)
    if wb < 64 * 1024:
        continue
    matmul_targets.append(name)
    matmul_weight_bytes += wb

if not matmul_targets:
    raise SystemExit("No material safe MatMul targets")
quantize_dynamic(
    model_input=str(PREPARED),
    model_output=str(LINEAR),
    op_types_to_quantize=["MatMul"],
    nodes_to_quantize=matmul_targets,
    per_channel=True,
    reduce_range=False,
    weight_type=QuantType.QInt8,
    extra_options={"MatMulConstBOnly": True},
)
linear_model = onnx.load(str(LINEAR), load_external_data=True)
pruned_linear = prune_unused_initializers(linear_model)
onnx.save(linear_model, str(LINEAR))
onnx.checker.check_model(linear_model)
# Runtime load is the compatibility gate that matters for the eventual drop-in file.
ort.InferenceSession(str(LINEAR), providers=["CPUExecutionProvider"])

linear_det = WORK / "linear_det.onnx"
patch_random(LINEAR, linear_det)
linear_quality = metrics(refs, run(linear_det, validation))
fp32_bytes = FP32.stat().st_size
linear_bytes = LINEAR.stat().st_size

# Static Conv INT8 candidates. QOperator QUInt8/QInt8 gives QLinearConv kernels;
# unlike dynamic Conv, activation ranges are calibrated once and do not need to be
# recomputed every synthesis call. Candidate thresholds are large-first so tiny
# output convolutions stay FP32 unless necessary.
thresholds = [2 * 1024 * 1024, 1024 * 1024, 512 * 1024, 256 * 1024, 128 * 1024, 64 * 1024, 32 * 1024]
candidates = []
eligible = []
for threshold in thresholds:
    conv_names = []
    conv_weight_bytes = 0
    for idx, n in enumerate(prepared.graph.node):
        name = label(n, idx)
        if n.op_type != "Conv" or name in protected:
            continue
        wb = node_weight_bytes(prepared, n)
        if wb < threshold:
            continue
        conv_names.append(name)
        conv_weight_bytes += wb
    if not conv_names:
        continue

    path = WORK / f"mixed_{threshold}.onnx"
    try:
        quantize_static(
            model_input=str(LINEAR),
            model_output=str(path),
            calibration_data_reader=Reader(calibration),
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=["Conv"],
            nodes_to_quantize=conv_names,
            per_channel=True,
            reduce_range=False,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
        m = onnx.load(str(path), load_external_data=True)
        pruned = prune_unused_initializers(m)
        onnx.save(m, str(path))
        onnx.checker.check_model(m)
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        det = WORK / f"mixed_{threshold}_det.onnx"
        patch_random(path, det)
        q = metrics(refs, run(det, validation))
        size = path.stat().st_size
        reduction = 1.0 - size / fp32_bytes
        op_counts = Counter(x.op_type for x in m.graph.node)
        speed = bench(path)
        item = {
            "convMinimumWeightBytes": threshold,
            "convTargets": len(conv_names),
            "convTargetWeightBytes": conv_weight_bytes,
            "optimizedBytes": size,
            "sizeReductionPercent": reduction * 100.0,
            "prunedInitializers": pruned,
            "quality": q,
            "qualityPass": quality_ok(q),
            "compressionPass": reduction >= MIN_REDUCTION,
            "ciHostSpeed": speed,
            "integerOps": {
                "MatMulInteger": op_counts.get("MatMulInteger", 0),
                "QLinearConv": op_counts.get("QLinearConv", 0),
                "ConvInteger": op_counts.get("ConvInteger", 0),
            },
        }
        item["eligible"] = bool(item["qualityPass"] and item["compressionPass"])
        candidates.append(item)
        if item["eligible"]:
            eligible.append((path, item))
    except Exception as exc:
        candidates.append({"convMinimumWeightBytes": threshold, "eligible": False, "error": repr(exc)})

base_report = {
    "runtime": ort.__version__,
    "originalBytes": fp32_bytes,
    "requiredReductionPercent": MIN_REDUCTION * 100.0,
    "qualityGates": {"durationExact": True, "minCorrelation": MIN_CORR, "minSnrDb": MIN_SNR, "maxNormalizedRmse": MAX_NRMSE},
    "protection": protection_report,
    "rewrite": rewrite_report,
    "rewriteQuality": rewrite_metrics,
    "dynamicMatMul": {
        "targets": len(matmul_targets),
        "targetWeightBytes": matmul_weight_bytes,
        "prunedInitializers": pruned_linear,
        "bytes": linear_bytes,
        "sizeReductionPercent": (1.0 - linear_bytes / fp32_bytes) * 100.0,
        "quality": linear_quality,
    },
    "convCandidates": candidates,
}

if not eligible:
    REPORT.write_text(json.dumps({"status": "rejected_no_material_candidate", **base_report}, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit("No >=25% smaller Conv+MatMul candidate passed all quality gates")

# Select fastest eligible candidate on the CI host, then smaller size as tie-break.
# This timing is only a proxy; the user's ARM64 device remains the real speed test.
eligible.sort(key=lambda x: (x[1]["ciHostSpeed"]["weightedScoreMs"], x[1]["optimizedBytes"]))
selected_path, selected = eligible[0]
shutil.copy2(selected_path, FINAL)

# Final exact ORT 1.17.1 load/inference verification.
s = ort.InferenceSession(str(FINAL), providers=["CPUExecutionProvider"])
for row in validation:
    y = np.asarray(s.run(["waveform"], row)[0], dtype=np.float32).reshape(-1)
    if y.size == 0 or not np.isfinite(y).all():
        raise SystemExit("Final model returned invalid waveform")

final_bytes = FINAL.stat().st_size
sha = hashlib.sha256(FINAL.read_bytes()).hexdigest()
report = {
    "status": "accepted_material_optimization",
    **base_report,
    "artifact": "kokoro_vi.onnx",
    "optimizedBytes": final_bytes,
    "sizeReductionPercent": (1.0 - final_bytes / fp32_bytes) * 100.0,
    "sha256": sha,
    "selectedCandidate": selected,
    "note": "Drop-in ONNX model only. Duration/prosody path is structurally FP32-protected. MatMul uses dynamic QInt8; large Conv uses calibrated QLinearConv. CI-host timing is diagnostic; real speed must be measured on the target ARM64 device.",
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"FINAL={FINAL}")
print(f"BYTES={final_bytes}")
print(f"REDUCTION={(1.0-final_bytes/fp32_bytes)*100.0:.2f}%")
print(f"SHA256={sha}")
