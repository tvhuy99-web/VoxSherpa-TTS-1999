#!/usr/bin/env python3
from pathlib import Path
from collections import Counter, defaultdict
import json
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'app/src/main/assets/kokoro_vi/kokoro_vi.onnx'
OUTDIR=ROOT/'build/constant-fold-test'; OUTDIR.mkdir(parents=True,exist_ok=True)
REPORT=ROOT/'.ci-state/kokoro-constant-fold-report.json'; REPORT.parent.mkdir(parents=True,exist_ok=True)

def nb(t): return len(t.raw_data) if t.raw_data else int(numpy_helper.to_array(t).nbytes)

def summarize(path):
    m=onnx.load(str(path),load_external_data=True); init={x.name:x for x in m.graph.initializer}; cons=defaultdict(set)
    for n in m.graph.node:
        for x in n.input:
            if x in init: cons[n.op_type].add(x)
    return {
      'fileBytes':Path(path).stat().st_size,
      'initializerCount':len(init),
      'initializerBytes':sum(nb(x) for x in init.values()),
      'opCounts':dict(Counter(n.op_type for n in m.graph.node)),
      'initializerBytesByConsumerOp':{op:sum(nb(init[x]) for x in names) for op,names in sorted(cons.items(),key=lambda kv:sum(nb(init[x]) for x in kv[1]),reverse=True)},
      'constantConvWeightBytes':sum(nb(init[n.input[1]]) for n in m.graph.node if n.op_type=='Conv' and len(n.input)>1 and n.input[1] in init),
      'constantConvTransposeWeightBytes':sum(nb(init[n.input[1]]) for n in m.graph.node if n.op_type=='ConvTranspose' and len(n.input)>1 and n.input[1] in init),
    }

result={'runtime':ort.__version__,'source':summarize(SRC),'levels':{}}
for name,level in [('basic',ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),('extended',ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)]:
    p=OUTDIR/f'kokoro_{name}.onnx'
    so=ort.SessionOptions(); so.graph_optimization_level=level; so.optimized_model_filepath=str(p)
    ort.InferenceSession(str(SRC),sess_options=so,providers=['CPUExecutionProvider'])
    result['levels'][name]=summarize(p)
REPORT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
