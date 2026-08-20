#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

from onnxruntime.quantization.onnx_model import ONNXModel


# ONNX Runtime 1.17.1 ONNXQuantizer.__init__ unconditionally invokes
# ONNXModel.replace_gemm_with_matmul() for dynamic quantization. That helper
# recursively rewrites Gemm inside Loop/Sequence subgraphs and corrupts the
# Kokoro loop-body topological order (SequenceAt ends up before its
# SplitToSequence producer). The v2 optimizer already performs an exact,
# duration-protected Gemm->MatMul(+Add) rewrite on the TOP-LEVEL graph, so the
# ORT automatic recursive rewrite is both redundant and harmful here.
#
# This monkey patch affects only the offline model-generation process. It does
# not alter the produced ONNX graph runtime, Android code, or ORT library. The
# output still must pass onnx.checker and load/run under ORT 1.17.1.
def _skip_recursive_gemm_rewrite(self: ONNXModel) -> None:
    return None


ONNXModel.replace_gemm_with_matmul = _skip_recursive_gemm_rewrite
print("ORT 1.17.1 recursive Gemm rewrite disabled for this offline optimizer run")

script = Path(__file__).with_name("prepare_aggressive_kokoro_model_v2.py")
runpy.run_path(str(script), run_name="__main__")
