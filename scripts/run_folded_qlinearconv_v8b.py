#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path
import onnx

# prepare_folded_qlinearconv_v8 adds intermediate Conv activations as temporary
# graph outputs for calibration. ONNX requires a shape field on graph outputs.
# All candidate operators are 1-D Conv, so activation rank is N,C,L (rank 3).
# Patch only the helper call that receives shape=None; all explicitly shaped
# tensors keep their original metadata.
_orig = onnx.helper.make_tensor_value_info

def _with_conv1d_rank(name, elem_type, shape, doc_string="", shape_denotation=None):
    if shape is None:
        shape = [None, None, None]
    return _orig(name, elem_type, shape, doc_string=doc_string, shape_denotation=shape_denotation)

onnx.helper.make_tensor_value_info = _with_conv1d_rank
print('Calibration fallback graph outputs use dynamic rank-3 shape [N,C,L]')
runpy.run_path(str(Path(__file__).with_name('prepare_folded_qlinearconv_v8.py')), run_name='__main__')
