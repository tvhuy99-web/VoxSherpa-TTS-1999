#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import onnx
from onnx import numpy_helper

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'app/src/main/assets/kokoro_vi/kokoro_vi.onnx'
OUT=ROOT/'.ci-state/kokoro-weight-inventory.json'
OUT.parent.mkdir(parents=True,exist_ok=True)
m=onnx.load(str(MODEL),load_external_data=True)
init={x.name:x for x in m.graph.initializer}

def nbytes(t):
    return len(t.raw_data) if t.raw_data else int(numpy_helper.to_array(t).nbytes)

def scope(name):
    s=name.lower()
    if 'bert' in s or 'text_encoder' in s: return 'bert_text'
    if 'predictor' in s or 'duration' in s or 'prosody' in s: return 'predictor_prosody'
    if 'decoder' in s or 'generator' in s: return 'decoder_generator'
    return 'other'

consumers=defaultdict(list)
for i,n in enumerate(m.graph.node):
    for inp in n.input:
        if inp in init:
            consumers[inp].append({'op':n.op_type,'name':n.name or f'__{i}_{n.op_type}','scope':scope(n.name)})

op_unique=defaultdict(set); scope_unique=defaultdict(set); all_consumed=set()
for name,cs in consumers.items():
    all_consumed.add(name)
    for c in cs:
        op_unique[c['op']].add(name); scope_unique[c['scope']].add(name)

by_op={k:{'uniqueInitializers':len(v),'bytes':sum(nbytes(init[x]) for x in v)} for k,v in op_unique.items()}
by_scope={k:{'uniqueInitializers':len(v),'bytes':sum(nbytes(init[x]) for x in v)} for k,v in scope_unique.items()}

top=[]
for name,t in init.items():
    top.append({'name':name,'bytes':nbytes(t),'dims':list(t.dims),'consumers':consumers.get(name,[])[:8]})
top.sort(key=lambda x:x['bytes'],reverse=True)

unconsumed=[x for x in init if x not in all_consumed]
report={
 'modelBytes':MODEL.stat().st_size,
 'initializerCount':len(init),
 'initializerBytes':sum(nbytes(x) for x in init.values()),
 'topLevelConsumedInitializerBytes':sum(nbytes(init[x]) for x in all_consumed),
 'topLevelUnconsumedInitializerCount':len(unconsumed),
 'topLevelUnconsumedInitializerBytes':sum(nbytes(init[x]) for x in unconsumed),
 'bytesByConsumerOp':dict(sorted(by_op.items(),key=lambda kv:kv[1]['bytes'],reverse=True)),
 'bytesByConsumerScope':dict(sorted(by_scope.items(),key=lambda kv:kv[1]['bytes'],reverse=True)),
 'topInitializers':top[:120],
}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
