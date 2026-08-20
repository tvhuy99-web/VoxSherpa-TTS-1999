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
FP32 = ASSET_DIR / "kokoro_vi.onnx"
PREPARED = ASSET_DIR / "kokoro_vi_int8_prepared.onnx"
INT8 = ASSET_DIR / "kokoro_vi_int8.onnx"
VOICE = ASSET_DIR / "voicepacks/diem_trinh.f32le"
REPORT = ASSET_DIR / "int8_quantization_report.json"

if not FP32.is_file():
    raise SystemExit(f"Missing FP32 model: {FP32}")
if not VOICE.is_file():
    raise SystemExit(f"Missing default voicepack: {VOICE}")

# Safety policy:
# - Never quantize BERT/text encoder/predictor/duration/LSTM/alignment paths.
# - Only touch decoder/generator linear layers after duration expansion.
# - ONNX Runtime 1.17.1 dynamic IntegerOps quantization supports MatMul but not
#   Gemm. Therefore duration-safe Gemm layers are first rewritten exactly as
#   MatMul + Add, but only when the Gemm attributes make that transformation
#   mathematically exact (alpha=1, beta=1, transA=0, constant unshared B).
# - Waveform sample count must match FP32 exactly before an APK is allowed.

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

    # Reuse the original initializer name. If Gemm requested transB=1, rewrite
    # the constant once in-place so MatMul sees [K,N]. No duplicate FP32 weight
    # is retained, which makes model-size reduction a meaningful validation.
    if trans_b == 1:
        weight_array = numpy_helper.to_array(weight)
        if weight_array.ndim != 2:
            skipped_gemm.append({"name": node.name, "reason": f"weight rank={weight_array.ndim}, expected 2"})
            new_nodes.append(node)
            continue
        transposed = np.ascontiguousarray(weight_array.T)
        weight.CopyFrom(numpy_helper.from_array(transposed, name=weight_name))

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

# Preserve original graph order while replacing only approved Gemm nodes.
del model.graph.node[:]
model.graph.node.extend(new_nodes)

if not converted_gemm_names:
    raise SystemExit(
        "No duration-safe Gemm node could be rewritten exactly; refusing an INT8 build that would only quantize the existing MatMul."
    )

onnx.checker.check_model(model)
if PREPARED.exists():
    PREPARED.unlink()
onnx.save(model, str(PREPARED))

# Verify that the exact Gemm->MatMul+Add rewrite is numerically and structurally
# equivalent BEFORE quantization. This catches any mistake in transpose/bias logic.
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

fp_sess = ort.InferenceSession(str(FP32), providers=providers)
prepared_sess = ort.InferenceSession(str(PREPARED), providers=providers)
fp_audio = np.asarray(fp_sess.run(["waveform"], feeds)[0], dtype=np.float32).reshape(-1)
prepared_audio = np.asarray(prepared_sess.run(["waveform"], feeds)[0], dtype=np.float32).reshape(-1)
if fp_audio.size != prepared_audio.size:
    raise SystemExit(
        f"Exact Gemm rewrite changed duration: FP32={fp_audio.size}, prepared={prepared_audio.size}"
    )
rewrite_max_abs = float(np.max(np.abs(fp_audio - prepared_audio)))
rewrite_rmse = float(np.sqrt(np.mean((fp_audio - prepared_audio) ** 2)))
if rewrite_max_abs > 2e-5 or rewrite_rmse > 2e-6:
    raise SystemExit(
        f"Exact Gemm rewrite failed numeric equivalence: maxAbs={rewrite_max_abs}, rmse={rewrite_rmse}"
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
# Almost every exact rewritten decoder linear should become MatMulInteger. Refuse
# a misleading 'INT8' artifact if ORT silently leaves most targets in FP32.
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

q_sess = ort.InferenceSession(str(INT8), providers=providers)
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

err = fp_audio - q_audio
mae = float(np.mean(np.abs(err)))
rmse = float(np.sqrt(np.mean(err * err)))
fp_rms = float(np.sqrt(np.mean(fp_audio * fp_audio)))
normalized_rmse = rmse / max(fp_rms, 1e-12)
if float(np.std(fp_audio)) > 1e-12 and float(np.std(q_audio)) > 1e-12:
    correlation = float(np.corrcoef(fp_audio, q_audio)[0, 1])
else:
    correlation = 1.0 if np.allclose(fp_audio, q_audio) else 0.0
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
    "rewriteMaxAbsError": rewrite_max_abs,
    "rewriteRmse": rewrite_rmse,
    "opsBefore": dict(sorted(op_counts_before.items())),
    "opsAfter": dict(sorted(op_counts_after.items())),
    "smokeWaveformSamples": int(fp_audio.size),
    "durationSamplesExactMatch": True,
    "smokeMae": mae,
    "smokeRmse": rmse,
    "smokeNormalizedRmse": normalized_rmse,
    "smokeCorrelation": correlation,
    "smokeSnrDb": snr_db,
    "note": "The Gemm rewrite is validated against FP32 before quantization; duration is protected and exact sample count is mandatory. Perceptual Vietnamese voice quality still requires on-device A/B listening.",
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"FP32 model: {fp32_bytes:,} bytes")
print(f"INT8 model: {int8_bytes:,} bytes ({size_ratio:.3f}x FP32)")
print(f"Exact Gemm rewrites: {len(converted_gemm_names)}; MatMulInteger: {matmul_integer_count}/{expected_targets}")
print("Real duration-safe selective INT8 model prepared and smoke-tested successfully")
