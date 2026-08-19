#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
engine = ROOT / "app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroEngine.java"
cpp = ROOT / "app/src/main/cpp/kokoro_vi/kokoro_vi_jni.cpp"

text = engine.read_text(encoding="utf-8")
text = text.replace(
    "            activeNnapi = false;\n        activeQnnGpu = false;\n            throw e;",
    "            activeNnapi = false;\n            activeQnnGpu = false;\n            throw e;",
)
text = text.replace(
    "            activeNnapi = false;\n        activeQnnGpu = false;\n            prepare(app);",
    "            activeNnapi = false;\n            activeQnnGpu = false;\n            prepare(app);",
)
text = text.replace('"benchmark_restore_nnapi"', '"benchmark_restore_accelerator"')
engine.write_text(text, encoding="utf-8")

text = cpp.read_text(encoding="utf-8")
text = text.replace(
    "    g_active_nnapi.store(false, std::memory_order_release);\n        g_active_qnn_gpu.store(false, std::memory_order_release);\n}",
    "    g_active_nnapi.store(false, std::memory_order_release);\n    g_active_qnn_gpu.store(false, std::memory_order_release);\n}",
)
cpp.write_text(text, encoding="utf-8")

print("QNN GPU experiment source cleanup applied")
