#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "app/src/main/assets/kokoro_vi"
WORK_DIR = ROOT / "build/selective_int8_validation"
FP32 = ASSET_DIR / "kokoro_vi.onnx"
PREPARED = WORK_DIR / "kokoro_vi_int8_prepared.onnx"
VALID_FP32 = WORK_DIR / "kokoro_vi_fp32_deterministic.onnx"
VALID_PREPARED = WORK_DIR / "kokoro_vi_prepared_deterministic.onnx"
VALID_INT8 = WORK_DIR / "kokoro_vi_int8_deterministic.onnx"
INT8 = ASSET_DIR / "kokoro_vi_int8.onnx"
VOICE = ASSET_DIR / "voicepacks/diem_trinh.f32le"
REPORT = ASSET_DIR / "int8_quantization_report.json"

WORK_DIR.mkdir(parents=True, exist_ok=True)
if not FP32.is_file():
    raise SystemExit(f"Missing FP32 model: {FP32}")
if not VOICE.is_file():
    raise SystemExit(f"Missing default voicepack: {VOICE}")

# Safety policy:
# - Never quantize BERT/text encoder/predictor/duration/LSTM/alignment paths.
# - Only touch decoder/generator linear layers after duration expansion.
# - ORT 1.17.1 dynamic IntegerOps quantization supports MatMul but not Gemm.
#   Duration-safe Gemm layers are therefore rewritten exactly as MatMul + Add,
#   only when alpha=1, beta=1, transA=0 and B is an unshared constant.
# - Validation copies force identical random seeds so stochastic generator noise
#   cannot masquerade as a Gemm-rewrite error. The shipped model keeps its
#   original random behavior.
# - Final INT8 must preserve waveform sample count exactly.

model = onnx.load(str(FP32), load_external_data=True)
onnx.checker.check_model(model)
op_counts_before = Counter(node.op_type for node in model.graph.node)
linear_nodes = [node for node in model.graph.node if node.op_type in {"MatMul", "Gemm"}]
if not linear_nodes:
    raise SystemExit("Kokoro graph has no MatMul/Gemm nodes to quantize")


def is_duration_safe_decoder_node(node: onnx.NodeProto) -> bool:
    haystack = " ".join([node.name, *node.input, *node.output]).lower()
    in_decoder = any(token in haystack for token in ("decoder", "generator", "synthesis"))
    timing_or_text = any(token in haystack for token in (
        "duration", "duration_proj", "predictor", "bert", "text_encoder",
        "prosody", "lstm", "align", "length", "repeat_interleave",
    ))
    return in_decoder and not timing_or_text


def force_deterministic_random(model_path: Path, output_path: Path, seed: float = 1729.0) -> None:
    m = onnx.load(str(model_path), load_external_data=True)

    def patch_graph(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            if node.op_type in {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"}:
                kept = [attr for attr in node.attribute if attr.name != "seed"]
                del node.attribute[:]
                node.attribute.extend(kept)
                node.attribute.append(onnx.helper.make_attribute("seed", float(seed)))
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    patch_graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        patch_graph(subgraph)

    patch_graph(m.graph)
    onnx.checker.check_model(m)
    onnx.save(m, str(output_path))


safe_linear = [node for node in linear_nodes if node.name and is_duration_safe_decoder_node(node)]
if not safe_linear:
    raise SystemExit("No duration-safe decoder MatMul/Gemm nodes were found")

initializer_by_name = {init.name: init for init in model.graph.initializer}
input_use_count = Counter(inp for node in model.graph.node for inp in node.input if inp)
safe_gemm_names = {node.name for node in safe_linear if node.op_type == "Gemm"}
safe_matmul_names = {node.name for node in safe_linear if node.op_type == "MatMul"}
converted_gemm_names: list[str] = []
skipped_gemm: list[dict] = []
quantize_matmul_names: list[str] = list(sorted(safe_matmul_names))
new_nodes: list[onnx.NodeProto] = []

for node in model.graph.node:
    if node.op_type != "Gemm" or node.name not in safe_gemm_names:
        new_nodes.append(node)
        continue

    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    trans_a = int(attrs.get("transA", 0))
    trans_b = int(attrs.get("transB", 0))
    weight_name = node.input[1] if len(node.input) >= 2 else ""
    weight = initializer_by_name.get(weight_name)

    reason = None
    if len(node.input) < 2:
        reason = "missing weight input"
    elif weight is None:
        reason = "weight is not a graph initializer"
    elif input_use_count[weight_name] != 1:
        reason = f"weight is shared by {input_use_count[weight_name]} nodes"
    elif not math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-8):
        reason = f"alpha={alpha} is not 1"
    elif not math.isclose(beta, 1.0, rel_tol=0.0, abs_tol=1e-8):
        reason = f"beta={beta} is not 1"
    elif trans_a != 0:
        reason = f"transA={trans_a} is unsupported for exact rewrite"
    elif trans_b not in (0, 1):
        reason = f"transB={trans_b} is invalid"

    if reason is not None:
        skipped_gemm.append({"name": node.name, "reason": reason})
        new_nodes.append(node)
        continue

    if trans_b == 1:
        weight_array = numpy_helper.to_array(weight)
        if weight_array.ndim != 2:
            skipped_gemm.append({"name": node.name, "reason": f"weight rank={weight_array.ndim}, expected 2"})
            new_nodes.append(node)
            continue
        weight.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(weight_array.T), name=weight_name))

    matmul_name = node.name + "__duration_safe_int8_matmul"
    has_bias = len(node.input) >= 3 and bool(node.input[2])
    matmul_output = node.output[0] + "__pre_bias" if has_bias else node.output[0]
    new_nodes.append(onnx.helper.make_node(
        "MatMul",
        [node.input[0], weight_name],
        [matmul_output],
        name=matmul_name,
    ))
    if has_bias:
        new_nodes.append(onnx.helper.make_node(
            "Add",
            [matmul_output, node.input[2]],
            [node.output[0]],
            name=node.name + "__duration_safe_bias_add",
        ))

    converted_gemm_names.append(node.name)
    quantize_matmul_names.append(matmul_name)

# Preserve graph order and infer the temporary MatMul output types/shapes required
# by ORT's MatMulInteger quantizer.
del model.graph.node[:]
model.graph.node.extend(new_nodes)
if not converted_gemm_names:
    raise SystemExit(
        "No duration-safe Gemm node could be rewritten exactly; refusing a fake INT8 build."
    )
onnx.checker.check_model(model)
try:
    model = onnx.shape_inference.infer_shapes(model, check_type=False, strict_mode=False, data_prop=False)
except Exception as exc:
    raise SystemExit(f"Shape inference failed after duration-safe Gemm rewrite: {exc}") from exc
onnx.checker.check_model(model)
onnx.save(model, str(PREPARED))

providers = ["CPUExecutionProvider"]
voice = np.fromfile(VOICE, dtype="<f4")
if voice.size != 510 * 256:
    raise SystemExit(f"Unexpected voicepack float count: {voice.size}")
style = voice.reshape(510, 256)[18:19].astype(np.float32, copy=False)
ids = np.ones((1, 21), dtype=np.int64)
ids[0, 0] = 0
ids[0, -1] = 0
speed = np.asarray(1.0, dtype=np.float32)
feeds = {"input_ids": ids, "ref_s": style, "speed": speed}

# Validate Gemm->MatMul+Add with deterministic generator noise. These copies are
# never packaged into the APK.
force_deterministic_random(FP32, VALID_FP32)
force_deterministic_random(PREPARED, VALID_PREPARED)
fp_det_sess = ort.InferenceSession(str(VALID_FP32), providers=providers)
prepared_det_sess = ort.InferenceSession(str(VALID_PREPARED), providers=providers)
fp_det_audio = np.asarray(fp_det_sess.run(["waveform"], feeds)[0], dtype=np.float32).reshape(-1)
prepared_det_audio = np.asarray(prepared_det_sess.run(["waveform"], feeds)[0], dtype=np.float32).reshape(-1)
if fp_det_audio.size != prepared_det_audio.size:
    raise SystemExit(
        f"Exact Gemm rewrite changed duration: FP32={fp_det_audio.size}, prepared={prepared_det_audio.size}"
    )
rewrite_err = fp_det_audio - prepared_det_audio
rewrite_max_abs = float(np.max(np.abs(rewrite_err)))
rewrite_rmse = float(np.sqrt(np.mean(rewrite_err * rewrite_err)))
if rewrite_max_abs > 2e-4 or rewrite_rmse > 2e-5:
    raise SystemExit(
        f"Exact Gemm rewrite failed deterministic equivalence: maxAbs={rewrite_max_abs}, rmse={rewrite_rmse}"
    )

if INT8.exists():
    INT8.unlink()
quantize_dynamic(
    model_input=str(PREPARED),
    model_output=str(INT8),
    op_types_to_quantize=["MatMul"],
    nodes_to_quantize=quantize_matmul_names,
    per_channel=True,
    reduce_range=False,
    weight_type=QuantType.QInt8,
    extra_options={"MatMulConstBOnly": True},
)

if not INT8.is_file() or INT8.stat().st_size <= 0:
    raise SystemExit("INT8 quantizer did not produce a model")
quant_model = onnx.load(str(INT8), load_external_data=True)
onnx.checker.check_model(quant_model)
op_counts_after = Counter(node.op_type for node in quant_model.graph.node)
matmul_integer_count = op_counts_after.get("MatMulInteger", 0)
expected_targets = len(quantize_matmul_names)
minimum_expected = max(8, int(math.floor(expected_targets * 0.80)))
if matmul_integer_count < minimum_expected:
    raise SystemExit(
        f"Too few decoder linears became MatMulInteger: got {matmul_integer_count}, "
        f"targets={expected_targets}, minimum={minimum_expected}"
    )

fp32_bytes = FP32.stat().st_size
int8_bytes = INT8.stat().st_size
size_ratio = int8_bytes / fp32_bytes
if int8_bytes >= fp32_bytes:
    raise SystemExit(
        f"Real INT8 model did not shrink: FP32={fp32_bytes}, INT8={int8_bytes}. Refusing misleading build."
    )

# Real shipped models: duration must still match exactly with original stochastic
# behavior. Randomness occurs after the protected duration path, so sample count is
# the hard structural invariant.
fp_sess = ort.InferenceSession(str(FP32), providers=providers)
q_sess = ort.InferenceSession(str(INT8), providers=providers)
fp_audio = np.asarray(fp_sess.run(["waveform"], feeds)[0], dtype=np.float32).reshape(-1)
q_audio = np.asarray(q_sess.run(["waveform"], feeds)[0], dtype=np.float32).reshape(-1)
if fp_audio.size == 0 or q_audio.size == 0:
    raise SystemExit("Smoke inference returned empty waveform")
if fp_audio.size != q_audio.size:
    raise SystemExit(
        f"Duration-safety invariant failed after INT8: FP32={fp_audio.size}, INT8={q_audio.size}. "
        "Refusing to build this INT8 model."
    )
if not np.isfinite(fp_audio).all() or not np.isfinite(q_audio).all():
    raise SystemExit("Smoke inference produced NaN/Inf")

# Meaningful numeric quality metrics use deterministic copies so generator noise is
# identical between FP32 and INT8. These are guardrail diagnostics, not a substitute
# for listening to real Vietnamese speech on-device.
force_deterministic_random(INT8, VALID_INT8)
q_det_sess = ort.InferenceSession(str(VALID_INT8), providers=providers)
q_det_audio = np.asarray(q_det_sess.run(["waveform"], feeds)[0], dtype=np.float32).reshape(-1)
if fp_det_audio.size != q_det_audio.size:
    raise SystemExit(
        f"Deterministic duration mismatch after INT8: FP32={fp_det_audio.size}, INT8={q_det_audio.size}"
    )
err = fp_det_audio - q_det_audio
mae = float(np.mean(np.abs(err)))
rmse = float(np.sqrt(np.mean(err * err)))
fp_rms = float(np.sqrt(np.mean(fp_det_audio * fp_det_audio)))
normalized_rmse = rmse / max(fp_rms, 1e-12)
if float(np.std(fp_det_audio)) > 1e-12 and float(np.std(q_det_audio)) > 1e-12:
    correlation = float(np.corrcoef(fp_det_audio, q_det_audio)[0, 1])
else:
    correlation = 1.0 if np.allclose(fp_det_audio, q_det_audio) else 0.0
snr_db = float(20.0 * math.log10(max(fp_rms, 1e-12) / max(rmse, 1e-12)))

report = {
    "quantizerRuntime": ort.__version__,
    "strategy": "duration-safe dynamic INT8: exact decoder/generator Gemm->MatMul+Add rewrite, then QInt8 MatMul only; predictor/duration/BERT/text/LSTM/Conv remain FP32",
    "fp32Bytes": fp32_bytes,
    "int8Bytes": int8_bytes,
    "int8ToFp32SizeRatio": size_ratio,
    "sizeReductionPercent": (1.0 - size_ratio) * 100.0,
    "allMatMulGemmNodes": len(linear_nodes),
    "durationSafeLinearCandidates": len(safe_linear),
    "convertedDurationSafeGemm": len(converted_gemm_names),
    "skippedDurationSafeGemm": skipped_gemm,
    "int8MatMulTargets": expected_targets,
    "matMulIntegerCount": matmul_integer_count,
    "rewriteValidationUsesFixedRandomSeed": True,
    "rewriteMaxAbsError": rewrite_max_abs,
    "rewriteRmse": rewrite_rmse,
    "opsBefore": dict(sorted(op_counts_before.items())),
    "opsAfter": dict(sorted(op_counts_after.items())),
    "smokeWaveformSamples": int(fp_audio.size),
    "durationSamplesExactMatch": True,
    "qualityMetricsUseFixedRandomSeed": True,
    "smokeMae": mae,
    "smokeRmse": rmse,
    "smokeNormalizedRmse": normalized_rmse,
    "smokeCorrelation": correlation,
    "smokeSnrDb": snr_db,
    "note": "Gemm rewrite is validated with identical random seeds; shipped model keeps original randomness. Duration sample count is mandatory. Perceptual Vietnamese voice quality still requires on-device A/B listening.",
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# Temporary validation/prepared models live under build/, not assets, and are never
# packaged. Leave them available for CI debugging only during this job.
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"FP32 model: {fp32_bytes:,} bytes")
print(f"INT8 model: {int8_bytes:,} bytes ({size_ratio:.3f}x FP32)")
print(f"Exact Gemm rewrites: {len(converted_gemm_names)}; MatMulInteger: {matmul_integer_count}/{expected_targets}")
print("Real duration-safe selective INT8 model prepared and smoke-tested successfully")
