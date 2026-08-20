#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path
import onnx

# Fast material candidate: the weight inventory shows decoder Conv tensors at
# ~13.4 MB, 12.6 MB, 10.5 MB, 6.7 MB, 6.3 MB, etc. Selecting all constant
# decoder Conv weights >=6 MiB targets a little over 110 MiB FP32 weight,
# enough to clear ~25% whole-model reduction when stored as INT8, while touching
# far fewer audio layers than broad quantization.
base = Path(__file__).with_name('prepare_folded_qlinearconv_v8.py')
text = base.read_text(encoding='utf-8')
old = "THRESHOLDS=(12*1024*1024,10*1024*1024,8*1024*1024,6*1024*1024,4*1024*1024,3*1024*1024,2*1024*1024,1536*1024,1024*1024,768*1024,512*1024,384*1024,256*1024,192*1024,128*1024,96*1024,64*1024,48*1024,32*1024)"
new = "THRESHOLDS=(6*1024*1024,)"
if text.count(old) != 1:
    raise SystemExit('Could not patch v8 threshold tuple exactly once')
text = text.replace(old, new, 1)
tmp = Path(__file__).with_name('_generated_folded_qlinearconv_v9_fast.py')
tmp.write_text(text, encoding='utf-8')

# Fix calibration-only graph output metadata: selected ops are Conv1D N,C,L.
_orig = onnx.helper.make_tensor_value_info
def _rank3(name, elem_type, shape, doc_string='', shape_denotation=None):
    if shape is None:
        shape = [None, None, None]
    return _orig(name, elem_type, shape, doc_string=doc_string, shape_denotation=shape_denotation)
onnx.helper.make_tensor_value_info = _rank3

print('Fast v9: testing one >=6MiB folded decoder-Conv QLinearConv candidate')
runpy.run_path(str(tmp), run_name='__main__')
