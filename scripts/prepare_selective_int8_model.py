#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
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

# Inspect the original graph first. This experiment intentionally quantizes only
# constant-weight MatMul/Gemm paths. Convolution/vocoder/output-sensitive paths
# remain FP32 so the first test prioritizes voice fidelity over maximum compression.
model = onnx.load(str(FP32), load_external_data=True)
op_counts_before = Counter(node.op_type for node in model.graph.node)
matmul_gemm_nodes = [node.name for node in model.graph.node if node.op_type in {"MatMul", "Gemm"}]
if not matmul_gemm_nodes:
    raise SystemExit("Kokoro graph has no MatMul/Gemm nodes to quantize")

if INT8.exists():
    INT8.unlink()

quantize_dynamic(
    model_input=str(FP32),
    model_output=str(INT8),
    op_types_to_quantize=["MatMul", "Gemm"],
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
    raise SystemExit("Selective INT8 model contains no quantized MatMul path")

fp32_bytes = FP32.stat().st_size
int8_bytes = INT8.stat().st_size
size_ratio = int8_bytes / fp32_bytes

# Validate with the exact same semantic runtime version used in the Android APK.
# A small real voice style is used. Token id 1 is deliberately conservative and
# valid for the embedding table; this smoke test checks load/run/shape/finiteness,
# not perceptual speech quality (that must be judged on-device with real text).
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
    raise SystemExit(f"Waveform size mismatch: FP32={fp_audio.size}, INT8={q_audio.size}")
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
    "strategy": "dynamic selective INT8, QInt8 weights, per-channel, MatMul/Gemm only, constant-B MatMul only",
    "fp32Bytes": fp32_bytes,
    "int8Bytes": int8_bytes,
    "int8ToFp32SizeRatio": size_ratio,
    "matMulGemmNodesBefore": len(matmul_gemm_nodes),
    "opsBefore": dict(sorted(op_counts_before.items())),
    "opsAfter": dict(sorted(op_counts_after.items())),
    "smokeWaveformSamples": int(fp_audio.size),
    "smokeMae": mae,
    "smokeRmse": rmse,
    "smokeNormalizedRmse": normalized_rmse,
    "smokeCorrelation": correlation,
    "smokeSnrDb": snr_db,
    "note": "Synthetic smoke metrics are structural/numeric only; real Vietnamese voice quality must be A/B listened on device.",
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"FP32 model: {fp32_bytes:,} bytes")
print(f"INT8 model: {int8_bytes:,} bytes ({size_ratio:.3f}x FP32)")
print("Selective INT8 model prepared and smoke-tested successfully")
