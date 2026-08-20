#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "app/src/main/assets/kokoro_vi"
FP32 = ASSET_DIR / "kokoro_vi.onnx"
INT8 = ASSET_DIR / "kokoro_vi_int8.onnx"
VOICE = ASSET_DIR / "voicepacks/diem_trinh.f32le"
REPORT = ASSET_DIR / "int8_quantization_report.json"

if not FP32.is_file():
    raise SystemExit(f"Missing FP32 model: {FP32}")
if not VOICE.is_file():
    raise SystemExit(f"Missing default voicepack: {VOICE}")

# Kokoro determines waveform length from the duration predictor before decoder
# synthesis. Quantizing the full MatMul/Gemm set changed a 21-token smoke waveform
# from 51,600 to 43,800 samples, which proves the first attempt touched timing.
# This second pass is deliberately duration-safe: quantize only MatMul/Gemm nodes
# whose exported module scope belongs to the decoder/generator path. Predictor,
# duration, BERT/text-encoder and all convolution/vocoder ops remain FP32.
model = onnx.load(str(FP32), load_external_data=True)
op_counts_before = Counter(node.op_type for node in model.graph.node)
linear_nodes = [node for node in model.graph.node if node.op_type in {"MatMul", "Gemm"}]
if not linear_nodes:
    raise SystemExit("Kokoro graph has no MatMul/Gemm nodes to quantize")

inventory = [
    {
        "name": node.name,
        "opType": node.op_type,
        "inputs": list(node.input[:2]),
        "outputs": list(node.output[:1]),
    }
    for node in linear_nodes
]

def is_duration_safe_decoder_node(node: onnx.NodeProto) -> bool:
    haystack = " ".join([node.name, *node.input, *node.output]).lower()
    # Positive scope: only decoder/generator/audio synthesis after duration expansion.
    in_decoder = any(token in haystack for token in ("decoder", "generator", "synthesis"))
    # Hard exclusions: never let a fuzzy exported name pull timing/text paths into INT8.
    timing_or_text = any(token in haystack for token in (
        "duration", "duration_proj", "predictor", "bert", "text_encoder",
        "prosody", "lstm", "align", "length", "repeat_interleave",
    ))
    return in_decoder and not timing_or_text

safe_nodes = [node.name for node in linear_nodes if node.name and is_duration_safe_decoder_node(node)]
if not safe_nodes:
    preview = json.dumps(inventory[:120], ensure_ascii=False, indent=2)
    raise SystemExit(
        "No duration-safe decoder MatMul/Gemm nodes were identified by exported scope. "
        "Refusing broad INT8. Linear-node inventory follows:\n" + preview
    )

if INT8.exists():
    INT8.unlink()

quantize_dynamic(
    model_input=str(FP32),
    model_output=str(INT8),
    op_types_to_quantize=["MatMul", "Gemm"],
    nodes_to_quantize=safe_nodes,
    per_channel=True,
    reduce_range=False,
    weight_type=QuantType.QInt8,
    extra_options={"MatMulConstBOnly": True},
)

if not INT8.is_file() or INT8.stat().st_size <= 0:
    raise SystemExit("INT8 quantizer did not produce a model")

quant_model = onnx.load(str(INT8), load_external_data=True)
op_counts_after = Counter(node.op_type for node in quant_model.graph.node)
quantized_integer_ops = (
    op_counts_after.get("MatMulInteger", 0)
    + op_counts_after.get("QLinearMatMul", 0)
    + op_counts_after.get("DynamicQuantizeLinear", 0)
)
if quantized_integer_ops <= 0:
    raise SystemExit(
        "Duration-safe INT8 selection produced no quantized MatMul path; refusing to ship a fake INT8 mode."
    )

fp32_bytes = FP32.stat().st_size
int8_bytes = INT8.stat().st_size
size_ratio = int8_bytes / fp32_bytes

# Smoke-test with exactly the same semantic ORT version used by Android. The
# absolute audio content here is synthetic; the hard safety invariant is that a
# decoder-only quantization MUST NOT alter predicted duration / waveform length.
providers = ["CPUExecutionProvider"]
fp_sess = ort.InferenceSession(str(FP32), providers=providers)
q_sess = ort.InferenceSession(str(INT8), providers=providers)

voice = np.fromfile(VOICE, dtype="<f4")
if voice.size != 510 * 256:
    raise SystemExit(f"Unexpected voicepack float count: {voice.size}")
style = voice.reshape(510, 256)[18:19].astype(np.float32, copy=False)
ids = np.ones((1, 21), dtype=np.int64)
ids[0, 0] = 0
ids[0, -1] = 0
speed = np.asarray(1.0, dtype=np.float32)
feeds = {"input_ids": ids, "ref_s": style, "speed": speed}

fp_audio = np.asarray(fp_sess.run(["waveform"], feeds)[0], dtype=np.float32).reshape(-1)
q_audio = np.asarray(q_sess.run(["waveform"], feeds)[0], dtype=np.float32).reshape(-1)
if fp_audio.size == 0 or q_audio.size == 0:
    raise SystemExit("Smoke inference returned empty waveform")
if fp_audio.size != q_audio.size:
    raise SystemExit(
        f"Duration-safety invariant failed: FP32={fp_audio.size}, INT8={q_audio.size}. "
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
    "strategy": "duration-safe dynamic INT8: decoder/generator MatMul/Gemm only; predictor/duration/BERT/text/vocoder-conv remain FP32",
    "fp32Bytes": fp32_bytes,
    "int8Bytes": int8_bytes,
    "int8ToFp32SizeRatio": size_ratio,
    "allMatMulGemmNodes": len(linear_nodes),
    "selectedDecoderLinearNodes": len(safe_nodes),
    "selectedNodeNames": safe_nodes,
    "opsBefore": dict(sorted(op_counts_before.items())),
    "opsAfter": dict(sorted(op_counts_after.items())),
    "smokeWaveformSamples": int(fp_audio.size),
    "durationSamplesExactMatch": True,
    "smokeMae": mae,
    "smokeRmse": rmse,
    "smokeNormalizedRmse": normalized_rmse,
    "smokeCorrelation": correlation,
    "smokeSnrDb": snr_db,
    "note": "Duration is structurally protected and smoke sample count must match exactly; perceptual Vietnamese voice quality still requires on-device A/B listening.",
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"FP32 model: {fp32_bytes:,} bytes")
print(f"INT8 model: {int8_bytes:,} bytes ({size_ratio:.3f}x FP32)")
print(f"Duration-safe decoder linear nodes selected: {len(safe_nodes)} / {len(linear_nodes)}")
print("Duration-safe selective INT8 model prepared and smoke-tested successfully")
