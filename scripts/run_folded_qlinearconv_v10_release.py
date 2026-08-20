#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path
import onnx

# Reproduce the already accepted v8b candidate exactly: all folded decoder Conv
# weights >=4 MiB. v8b measured 12 QLinearConv nodes, 116,920,320 FP32 source
# weight bytes, 237,564,533-byte final model (27.067% reduction), exact duration
# across 12 probes, min correlation 0.989247, min SNR 17.4198 dB.
base = Path(__file__).with_name('prepare_folded_qlinearconv_v8.py')
text = base.read_text(encoding='utf-8')
old = "THRESHOLDS=(12*1024*1024,10*1024*1024,8*1024*1024,6*1024*1024,4*1024*1024,3*1024*1024,2*1024*1024,1536*1024,1024*1024,768*1024,512*1024,384*1024,256*1024,192*1024,128*1024,96*1024,64*1024,48*1024,32*1024)"
new = "THRESHOLDS=(4*1024*1024,)"
if text.count(old) != 1:
    raise SystemExit('Could not patch v8 threshold tuple exactly once')
text = text.replace(old, new, 1)
tmp = Path(__file__).with_name('_generated_folded_qlinearconv_v10_release.py')
tmp.write_text(text, encoding='utf-8')

_orig = onnx.helper.make_tensor_value_info
def _rank3(name, elem_type, shape, doc_string='', shape_denotation=None):
    if shape is None:
        shape = [None, None, None]
    return _orig(name, elem_type, shape, doc_string=doc_string, shape_denotation=shape_denotation)
onnx.helper.make_tensor_value_info = _rank3

print('v10 release: reproducing accepted >=4MiB folded decoder QLinearConv candidate')
runpy.run_path(str(tmp), run_name='__main__')
