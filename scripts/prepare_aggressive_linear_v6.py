#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "app/src/main/assets/kokoro_vi"
OUT = ROOT / "build/model-only-output"
WORK = ROOT / "build/aggressive-linear-v6"
FP32 = A / "kokoro_vi.onnx"
VOICE = A / "voicepacks/diem_trinh.f32le"
PREP = WORK / "prepared.onnx"
FINAL = OUT / "kokoro_vi.onnx"
REPORT = OUT / "optimization_report.json"
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

MIN_REDUCTION = 0.25
MIN_CORR = 0.985
MIN_SNR = 16.0
MAX_NRMSE = 0.16
TOKEN_COUNTS = (21, 50, 71, 110)

if not FP32.is_file() or FP32.stat().st_size < 300_000_000:
    raise SystemExit("original model missing")
voice = np.fromfile(VOICE, dtype="<f4").reshape(510, 256)


def label(n, i):
    return n.name or f"__n_{i}_{n.op_type}"


def hard_sensitive(n) -> bool:
    h = " ".join([n.name, *n.input, *n.output]).lower()
    # Protect timing/prosody prediction itself. Text/BERT is allowed to quantize,
    # but any resulting timing drift is independently rejected by multi-pattern
    # exact-duration validation below.
    keys = (
        "predictor", "duration", "duration_proj", "prosody", "align",
        "repeat_interleave", "noise_pred", "/f0", "f0_",
    )
    return any(k in h for k in keys)


def prune(m):
    used = {x for n in m.graph.node for x in n.input if x}
    before = len(m.graph.initializer)
    kept = [x for x in m.graph.initializer if x.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept)
    return before - len(kept)


def rewrite_gemm(m):
    init = {x.name: x for x in m.graph.initializer}
    nodes, added = [], []
    converted = []
    for i, n in enumerate(m.graph.node):
        name = label(n, i)
        if n.op_type != "Gemm" or hard_sensitive(n) or len(n.input) < 2 or n.input[1] not in init:
            nodes.append(n)
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        alpha = float(attrs.get("alpha", 1.0)); beta = float(attrs.get("beta", 1.0))
        ta = int(attrs.get("transA", 0)); tb = int(attrs.get("transB", 0))
        if ta not in (0,1) or tb not in (0,1):
            nodes.append(n); continue
        w = numpy_helper.to_array(init[n.input[1]]).astype(np.float32, copy=False)
        if tb: w = w.T
        if not math.isclose(alpha, 1.0, abs_tol=1e-12): w = w * np.float32(alpha)
        wn = f"{n.input[1]}__v6_{len(converted)}"
        added.append(numpy_helper.from_array(np.ascontiguousarray(w), name=wn))
        x = n.input[0]
        if ta:
            xt = n.output[0] + "__v6_ta"
            nodes.append(onnx.helper.make_node("Transpose", [x], [xt], name=name+"__v6_ta", perm=[1,0])); x = xt
        has_bias = len(n.input) > 2 and bool(n.input[2])
        mmout = n.output[0] + "__v6_prebias" if has_bias else n.output[0]
        mmname = name + "__v6_matmul"
        nodes.append(onnx.helper.make_node("MatMul", [x, wn], [mmout], name=mmname))
        if has_bias:
            bn = n.input[2]
            if not math.isclose(beta, 1.0, abs_tol=1e-12):
                if bn not in init:
                    raise SystemExit(f"cannot exactly fold nonconstant beta for {name}")
                b = numpy_helper.to_array(init[bn]).astype(np.float32, copy=False) * np.float32(beta)
                bn2 = bn + f"__v6_{len(converted)}"
                added.append(numpy_helper.from_array(np.ascontiguousarray(b), name=bn2)); bn = bn2
            nodes.append(onnx.helper.make_node("Add", [mmout, bn], list(n.output), name=name+"__v6_bias"))
        converted.append(mmname)
    del m.graph.node[:]; m.graph.node.extend(nodes); m.graph.initializer.extend(added)
    pruned = prune(m)
    m = onnx.shape_inference.infer_shapes(m, check_type=False, strict_mode=False, data_prop=False)
    onnx.checker.check_model(m)
    return m, converted, pruned


def manual_quantize(m, targets):
    init = {x.name:x for x in m.graph.initializer}
    targetset=set(targets); nodes=[]; added=[]; cache={}; converted=[]; total_weight=0
    for i,n in enumerate(m.graph.node):
        name=label(n,i)
        if n.op_type!="MatMul" or name not in targetset or len(n.input)<2 or n.input[1] not in init:
            nodes.append(n); continue
        w=numpy_helper.to_array(init[n.input[1]])
        if w.ndim!=2 or w.dtype.kind!='f': nodes.append(n); continue
        w=w.astype(np.float32,copy=False); total_weight += int(w.nbytes)
        scale=np.max(np.abs(w),axis=0).astype(np.float32)/np.float32(127.0)
        scale=np.where(scale>1e-12,scale,np.float32(1.0)).astype(np.float32)
        qw=np.clip(np.rint(w/scale.reshape(1,-1)),-127,127).astype(np.int8)
        wz=np.zeros((qw.shape[1],),dtype=np.int8)
        stem=name.replace(':','_'); qwn=stem+'__w_i8'; wsn=stem+'__w_s'; wzn=stem+'__w_z'
        added.extend([numpy_helper.from_array(qw,name=qwn),numpy_helper.from_array(scale,name=wsn),numpy_helper.from_array(wz,name=wzn)])
        x=n.input[0]
        if x not in cache:
            aq=x+f"__v6_u8_{len(converted)}"; xs=x+f"__v6_s_{len(converted)}"; xz=x+f"__v6_z_{len(converted)}"
            nodes.append(onnx.helper.make_node("DynamicQuantizeLinear",[x],[aq,xs,xz],name=stem+'__dql')); cache[x]=(aq,xs,xz)
        aq,xs,xz=cache[x]
        i32=n.output[0]+'__v6_i32'; f32=n.output[0]+'__v6_f32'; sc=n.output[0]+'__v6_sc'
        nodes.extend([
            onnx.helper.make_node("MatMulInteger",[aq,qwn,xz,wzn],[i32],name=stem+'__mmi'),
            onnx.helper.make_node("Cast",[i32],[f32],name=stem+'__cast',to=onnx.TensorProto.FLOAT),
            onnx.helper.make_node("Mul",[xs,wsn],[sc],name=stem+'__smul'),
            onnx.helper.make_node("Mul",[f32,sc],list(n.output),name=stem+'__deq'),
        ]); converted.append(name)
    del m.graph.node[:]; m.graph.node.extend(nodes); m.graph.initializer.extend(added)
    pruned=prune(m); onnx.checker.check_model(m)
    return m,converted,total_weight,pruned


def patch_random(src,dst,seed=1729.0):
    m=onnx.load(str(src),load_external_data=True)
    def walk(g):
        for n in g.node:
            if n.op_type in {"RandomNormal","RandomNormalLike","RandomUniform","RandomUniformLike"}:
                attrs=[a for a in n.attribute if a.name!='seed']; del n.attribute[:]; n.attribute.extend(attrs); n.attribute.append(onnx.helper.make_attribute('seed',float(seed)))
            for a in n.attribute:
                if a.type==onnx.AttributeProto.GRAPH: walk(a.g)
                elif a.type==onnx.AttributeProto.GRAPHS:
                    for sg in a.graphs: walk(sg)
    walk(m.graph); onnx.checker.check_model(m); onnx.save(m,str(dst))


def feeds():
    rows=[]
    # 3 independent token patterns at each representative length = 12 probes.
    for n in TOKEN_COUNTS:
        for p in range(3):
            ids=np.empty((1,n),dtype=np.int64); ids[0,0]=0; ids[0,-1]=0
            for j in range(1,n-1): ids[0,j]=1+((j*(7+2*p)+17*p)%100)
            row=min(509,max(0,n-3+p))
            rows.append((n,p,{"input_ids":ids,"ref_s":voice[row:row+1].astype(np.float32,copy=False),"speed":np.asarray(1.0,dtype=np.float32)}))
    return rows


def run(path, rows):
    s=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']); out=[]
    for n,p,f in rows:
        y=np.asarray(s.run(['waveform'],f)[0],dtype=np.float32).reshape(-1)
        if y.size==0 or not np.isfinite(y).all(): raise RuntimeError('invalid waveform')
        out.append((n,p,y))
    return out


def compare(ref,cand):
    cases=[]; exact=True; minc=1.; mins=999.; maxn=0.
    for (n,p,a),(n2,p2,b) in zip(ref,cand):
        assert (n,p)==(n2,p2)
        if a.size!=b.size:
            exact=False; minc=-1.; mins=-999.; maxn=999.; cases.append({'tokens':n,'pattern':p,'durationExact':False,'refSamples':int(a.size),'candSamples':int(b.size)}); continue
        e=a-b; rmse=float(np.sqrt(np.mean(e*e))); rms=float(np.sqrt(np.mean(a*a))); nr=rmse/max(rms,1e-12)
        c=float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else 0.; sn=float(20*math.log10(max(rms,1e-12)/max(rmse,1e-12)))
        minc=min(minc,c); mins=min(mins,sn); maxn=max(maxn,nr); cases.append({'tokens':n,'pattern':p,'durationExact':True,'samples':int(a.size),'correlation':c,'snrDb':sn,'normalizedRmse':nr})
    return {'durationExactAll':exact,'minCorrelation':minc,'minSnrDb':mins,'maxNormalizedRmse':maxn,'cases':cases}


orig=onnx.load(str(FP32),load_external_data=True); onnx.checker.check_model(orig)
prepared, rewritten, pruned_rewrite = rewrite_gemm(orig)
onnx.save(prepared,str(PREP))

# Confirm rewrite itself is exact across 12 deterministic probes.
refdet=WORK/'refdet.onnx'; prepdet=WORK/'prepdet.onnx'; patch_random(FP32,refdet); patch_random(PREP,prepdet)
rows=feeds(); ref=run(refdet,rows); prepwaves=run(prepdet,rows); rewrite_quality=compare(ref,prepwaves)
if not rewrite_quality['durationExactAll'] or rewrite_quality['maxNormalizedRmse']>0.0005:
    raise SystemExit('rewrite failed exactness gate')

# All material top-level MatMul outside explicitly sensitive predictor/prosody scope.
init={x.name:x for x in prepared.graph.initializer}; targets=[]; target_scopes=Counter(); target_bytes=0
for i,n in enumerate(prepared.graph.node):
    name=label(n,i)
    if n.op_type!='MatMul' or hard_sensitive(n) or len(n.input)<2 or n.input[1] not in init: continue
    w=numpy_helper.to_array(init[n.input[1]])
    if w.nbytes < 64*1024: continue
    targets.append(name); target_bytes += int(w.nbytes)
    h=' '.join([n.name,*n.input,*n.output]).lower()
    scope='bert_text' if ('bert' in h or 'text_encoder' in h) else ('decoder' if 'decoder' in h or 'generator' in h else 'other')
    target_scopes[scope]+=1

quant,converted,qweightbytes,pruned_quant=manual_quantize(prepared,targets)
onnx.save(quant,str(FINAL)); onnx.checker.check_model(quant)
# Exact runtime used by Android must load before any acceptance decision.
ort.InferenceSession(str(FINAL),providers=['CPUExecutionProvider'])
qdet=WORK/'qdet.onnx'; patch_random(FINAL,qdet); quality=compare(ref,run(qdet,rows))
size=FINAL.stat().st_size; original_size=FP32.stat().st_size; reduction=1-size/original_size
ops=Counter(n.op_type for n in quant.graph.node)
accepted=(reduction>=MIN_REDUCTION and quality['durationExactAll'] and quality['minCorrelation']>=MIN_CORR and quality['minSnrDb']>=MIN_SNR and quality['maxNormalizedRmse']<=MAX_NRMSE)
report={
    'status':'accepted_material_optimization' if accepted else 'rejected_linear_candidate',
    'runtime':ort.__version__,'artifact':'kokoro_vi.onnx','originalBytes':original_size,'optimizedBytes':size,'sizeReductionPercent':reduction*100,
    'requiredReductionPercent':MIN_REDUCTION*100,'qualityGates':{'durationExact':True,'minCorrelation':MIN_CORR,'minSnrDb':MIN_SNR,'maxNormalizedRmse':MAX_NRMSE},
    'strategy':'aggressive top-level dynamic INT8 linear: BERT/text/decoder allowed; predictor/duration/prosody/F0 scopes remain FP32; 12 deterministic duration/quality probes',
    'rewrittenGemm':len(rewritten),'rewritePrunedInitializers':pruned_rewrite,'rewriteQuality':rewrite_quality,
    'targetMatMulCount':len(targets),'convertedMatMulInteger':len(converted),'targetWeightBytes':target_bytes,'quantizedWeightBytes':qweightbytes,'targetScopes':dict(target_scopes),'quantPrunedInitializers':pruned_quant,
    'integerOps':{'MatMulInteger':ops.get('MatMulInteger',0),'DynamicQuantizeLinear':ops.get('DynamicQuantizeLinear',0)},'quality':quality,
}
if accepted: report['sha256']=hashlib.sha256(FINAL.read_bytes()).hexdigest()
REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
if not accepted:
    FINAL.unlink(missing_ok=True)
    raise SystemExit('aggressive linear candidate did not meet both >=25% size and quality gates')
