#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'app/src/main/assets/kokoro_vi'
WORK=ROOT/'build/folded-qlinearconv-v8'; WORK.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'build/model-only-output'; OUT.mkdir(parents=True,exist_ok=True)
SRC=A/'kokoro_vi.onnx'; VOICE=A/'voicepacks/diem_trinh.f32le'
FOLDED=WORK/'kokoro_folded_basic.onnx'; FINAL=OUT/'kokoro_vi.onnx'; REPORT=OUT/'optimization_report.json'

MIN_REDUCTION=0.25
MIN_CORR=0.985
MIN_SNR=16.0
MAX_NRMSE=0.16
TOKEN_COUNTS=(21,50,71,110)
THRESHOLDS=(12*1024*1024,10*1024*1024,8*1024*1024,6*1024*1024,4*1024*1024,3*1024*1024,2*1024*1024,1536*1024,1024*1024,768*1024,512*1024,384*1024,256*1024,192*1024,128*1024,96*1024,64*1024,48*1024,32*1024)

if not SRC.is_file() or SRC.stat().st_size<300_000_000: raise SystemExit('missing original model')
voice=np.fromfile(VOICE,dtype='<f4').reshape(510,256)

def nb(t): return len(t.raw_data) if t.raw_data else int(numpy_helper.to_array(t).nbytes)
def label(n,i): return n.name or f'__n_{i}_{n.op_type}'
def decoder_conv(n):
    h=' '.join([n.name,*n.input,*n.output]).lower()
    return n.op_type=='Conv' and ('decoder' in h or 'generator' in h) and not any(k in h for k in ('predictor','duration','prosody','text_encoder','bert','/f0','f0_','noise_pred'))

def recursive_used(g):
    used=set()
    for n in g.node:
        used.update(x for x in n.input if x)
        for a in n.attribute:
            if a.type==onnx.AttributeProto.GRAPH: used.update(recursive_used(a.g))
            elif a.type==onnx.AttributeProto.GRAPHS:
                for sg in a.graphs: used.update(recursive_used(sg))
    return used

def prune(m):
    used=recursive_used(m.graph); before=len(m.graph.initializer)
    kept=[x for x in m.graph.initializer if x.name in used]
    del m.graph.initializer[:]; m.graph.initializer.extend(kept)
    return before-len(kept)

def make_feed(n,p=0,speed=1.0):
    ids=np.empty((1,n),dtype=np.int64); ids[0,0]=0; ids[0,-1]=0
    for j in range(1,n-1): ids[0,j]=1+((j*(7+2*p)+17*p)%100)
    row=min(509,max(0,n-3+p))
    return {'input_ids':ids,'ref_s':voice[row:row+1].astype(np.float32,copy=False),'speed':np.asarray(speed,dtype=np.float32)}

def validation_rows(): return [(n,p,make_feed(n,p,1.0)) for n in TOKEN_COUNTS for p in range(3)]
def calibration_rows(): return [make_feed(n,(p+i)%3,(0.92,1.0,1.08)[p]) for p in range(3) for i,n in enumerate(TOKEN_COUNTS)]

def patch_random(src,dst,seed=1729.0):
    m=onnx.load(str(src),load_external_data=True)
    def walk(g):
        for n in g.node:
            if n.op_type in {'RandomNormal','RandomNormalLike','RandomUniform','RandomUniformLike'}:
                attrs=[a for a in n.attribute if a.name!='seed']; del n.attribute[:]; n.attribute.extend(attrs); n.attribute.append(onnx.helper.make_attribute('seed',float(seed)))
            for a in n.attribute:
                if a.type==onnx.AttributeProto.GRAPH: walk(a.g)
                elif a.type==onnx.AttributeProto.GRAPHS:
                    for sg in a.graphs: walk(sg)
    walk(m.graph); onnx.checker.check_model(m); onnx.save(m,str(dst))

def run_waves(path,rows):
    s=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']); out=[]
    for n,p,f in rows:
        y=np.asarray(s.run(['waveform'],f)[0],dtype=np.float32).reshape(-1)
        if y.size==0 or not np.isfinite(y).all(): raise RuntimeError(f'invalid waveform from {path}')
        out.append((n,p,y))
    return out

def compare(ref,cand):
    cases=[]; exact=True; minc=1.; mins=999.; maxnr=0.
    for (n,p,a),(n2,p2,b) in zip(ref,cand):
        assert (n,p)==(n2,p2)
        if a.size!=b.size:
            exact=False; minc=-1.; mins=-999.; maxnr=999.; cases.append({'tokens':n,'pattern':p,'durationExact':False,'refSamples':int(a.size),'candSamples':int(b.size)}); continue
        e=a-b; rmse=float(np.sqrt(np.mean(e*e))); rms=float(np.sqrt(np.mean(a*a))); nr=rmse/max(rms,1e-12)
        c=float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else 0.; sn=float(20*math.log10(max(rms,1e-12)/max(rmse,1e-12)))
        minc=min(minc,c); mins=min(mins,sn); maxnr=max(maxnr,nr); cases.append({'tokens':n,'pattern':p,'durationExact':True,'samples':int(a.size),'correlation':c,'snrDb':sn,'normalizedRmse':nr})
    return {'durationExactAll':exact,'minCorrelation':minc,'minSnrDb':mins,'maxNormalizedRmse':maxnr,'cases':cases}
def quality_ok(q): return q['durationExactAll'] and q['minCorrelation']>=MIN_CORR and q['minSnrDb']>=MIN_SNR and q['maxNormalizedRmse']<=MAX_NRMSE

# Step 1: let the exact Android runtime version fold weight normalization into
# constant Conv/ConvTranspose initializers. BASIC avoids hardware-specific fusions.
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC; so.optimized_model_filepath=str(FOLDED)
ort.InferenceSession(str(SRC),sess_options=so,providers=['CPUExecutionProvider'])
folded=onnx.load(str(FOLDED),load_external_data=True); onnx.checker.check_model(folded)
init={x.name:x for x in folded.graph.initializer}
constant_decoder_convs=[]
for i,n in enumerate(folded.graph.node):
    if decoder_conv(n) and len(n.input)>1 and n.input[1] in init:
        constant_decoder_convs.append((label(n,i),nb(init[n.input[1]])))
constant_decoder_convs.sort(key=lambda x:x[1],reverse=True)
if not constant_decoder_convs: raise SystemExit('folding produced no constant decoder Conv weights')

# Verify BASIC folding preserves deterministic waveform before quantizing anything.
rows=validation_rows(); src_det=WORK/'src_det.onnx'; fold_det=WORK/'fold_det.onnx'; patch_random(SRC,src_det); patch_random(FOLDED,fold_det)
ref=run_waves(src_det,rows); folded_quality=compare(ref,run_waves(fold_det,rows))
if not folded_quality['durationExactAll'] or folded_quality['minCorrelation']<0.99999 or folded_quality['maxNormalizedRmse']>0.002:
    raise SystemExit('ORT BASIC folded model did not meet near-lossless equivalence gate: '+json.dumps(folded_quality))

# Instrument a candidate model to expose selected Conv inputs+outputs and collect
# representative min/max ranges. This is done per candidate so only necessary
# tensors are returned from ORT, keeping memory bounded.
def calibration_ranges(model_path,selected_names):
    m=onnx.load(str(model_path),load_external_data=True); selected=set(selected_names); tensors=[]
    existing_out={x.name for x in m.graph.output}
    value_info={x.name:x for x in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)}
    for i,n in enumerate(m.graph.node):
        if label(n,i) not in selected: continue
        tensors.extend([n.input[0],n.output[0]])
    tensors=list(dict.fromkeys(tensors))
    for t in tensors:
        if t in existing_out: continue
        if t in value_info: m.graph.output.append(copy.deepcopy(value_info[t]))
        else: m.graph.output.append(onnx.helper.make_tensor_value_info(t,onnx.TensorProto.FLOAT,None))
    inst=WORK/'calibration_instrumented.onnx'; onnx.save(m,str(inst)); onnx.checker.check_model(m)
    s=ort.InferenceSession(str(inst),providers=['CPUExecutionProvider']); outnames=[x.name for x in s.get_outputs()]
    wanted={t:outnames.index(t) for t in tensors}
    ranges={t:[float('inf'),float('-inf')] for t in tensors}
    for f in calibration_rows():
        outs=s.run(None,f)
        for t,idx in wanted.items():
            a=np.asarray(outs[idx]); lo=float(np.min(a)); hi=float(np.max(a)); ranges[t][0]=min(ranges[t][0],lo); ranges[t][1]=max(ranges[t][1],hi)
    return ranges

def u8_params(lo,hi):
    lo=min(0.0,float(lo)); hi=max(0.0,float(hi))
    if hi-lo<1e-12: return np.float32(1.0),np.uint8(0)
    s=np.float32((hi-lo)/255.0); z=int(round(-lo/float(s))); return s,np.uint8(max(0,min(255,z)))

def make_qlinear_candidate(selected_names,ranges,path):
    m=onnx.load(str(FOLDED),load_external_data=True); inits={x.name:x for x in m.graph.initializer}; sel=set(selected_names)
    nodes=[]; added=[]; qcache={}; converted=0; qweight_source_bytes=0
    for i,n in enumerate(m.graph.node):
        name=label(n,i)
        if name not in sel:
            nodes.append(n); continue
        if n.op_type!='Conv' or len(n.input)<2 or n.input[1] not in inits: raise RuntimeError(f'selected node not constant Conv: {name}')
        w=numpy_helper.to_array(inits[n.input[1]]).astype(np.float32,copy=False); qweight_source_bytes+=int(w.nbytes)
        axes=tuple(range(1,w.ndim)); ws=np.max(np.abs(w),axis=axes).astype(np.float32)/np.float32(127.0); ws=np.where(ws>1e-12,ws,np.float32(1.0)).astype(np.float32)
        shape=(w.shape[0],)+((1,)*(w.ndim-1)); qw=np.clip(np.rint(w/ws.reshape(shape)),-127,127).astype(np.int8); wz=np.zeros((w.shape[0],),dtype=np.int8)
        stem=name.replace(':','_'); qwn=stem+'__ql_w'; wsn=stem+'__ql_ws'; wzn=stem+'__ql_wz'
        added.extend([numpy_helper.from_array(qw,name=qwn),numpy_helper.from_array(ws,name=wsn),numpy_helper.from_array(wz,name=wzn)])
        x=n.input[0]; y=n.output[0]
        xs,xz=u8_params(*ranges[x]); ys,yz=u8_params(*ranges[y])
        xsn=stem+'__x_scale'; xzn=stem+'__x_zero'; ysn=stem+'__y_scale'; yzn=stem+'__y_zero'
        added.extend([numpy_helper.from_array(np.asarray(xs,dtype=np.float32),name=xsn),numpy_helper.from_array(np.asarray(xz,dtype=np.uint8),name=xzn),numpy_helper.from_array(np.asarray(ys,dtype=np.float32),name=ysn),numpy_helper.from_array(np.asarray(yz,dtype=np.uint8),name=yzn)])
        cachekey=(x,float(xs),int(xz))
        if cachekey not in qcache:
            xq=x+'__ql_u8_'+str(len(qcache)); nodes.append(onnx.helper.make_node('QuantizeLinear',[x,xsn,xzn],[xq],name=stem+'__quant_x')); qcache[cachekey]=xq
        xq=qcache[cachekey]; yq=y+'__ql_u8'
        qinputs=[xq,xsn,xzn,qwn,wsn,wzn,ysn,yzn]
        if len(n.input)>2 and n.input[2]:
            if n.input[2] not in inits: raise RuntimeError(f'nonconstant Conv bias: {name}')
            b=numpy_helper.to_array(inits[n.input[2]]).astype(np.float32,copy=False); bscale=np.asarray(xs,dtype=np.float32)*ws
            qb=np.rint(b/bscale).astype(np.int32); qbn=stem+'__ql_bias'; added.append(numpy_helper.from_array(qb,name=qbn)); qinputs.append(qbn)
        attrs={a.name:onnx.helper.get_attribute_value(a) for a in n.attribute}
        nodes.append(onnx.helper.make_node('QLinearConv',qinputs,[yq],name=stem+'__QLinearConv',**attrs))
        nodes.append(onnx.helper.make_node('DequantizeLinear',[yq,ysn,yzn],[y],name=stem+'__dequant_y'))
        converted+=1
    del m.graph.node[:]; m.graph.node.extend(nodes); m.graph.initializer.extend(added); pruned=prune(m)
    onnx.checker.check_model(m); onnx.save(m,str(path)); ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    return {'convertedConv':converted,'quantizedSourceWeightBytes':qweight_source_bytes,'prunedInitializers':pruned,'opCounts':dict(Counter(n.op_type for n in m.graph.node))}

def bench(path):
    s=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']); probes=[make_feed(n,0,1.0) for n in TOKEN_COUNTS]; s.run(['waveform'],probes[0]); ms=[]
    for f in probes:
        t=time.perf_counter(); s.run(['waveform'],f); ms.append((time.perf_counter()-t)*1000)
    return {'caseMs':ms,'weightedScoreMs':ms[0]*4+ms[1]*3+ms[2]*2+ms[3]}

candidates=[]; eligible=[]; original_bytes=SRC.stat().st_size
for threshold in THRESHOLDS:
    names=[name for name,size in constant_decoder_convs if size>=threshold]
    if not names: continue
    # Avoid duplicate candidate sets when adjacent thresholds choose same nodes.
    if candidates and candidates[-1].get('selectedNames')==names: continue
    selected_bytes=sum(size for name,size in constant_decoder_convs if name in set(names))
    ranges=calibration_ranges(FOLDED,names)
    path=WORK/f'candidate_{threshold}.onnx'
    try:
        build=make_qlinear_candidate(names,ranges,path); size=path.stat().st_size; reduction=1-size/original_bytes
        det=WORK/f'candidate_{threshold}_det.onnx'; patch_random(path,det); q=compare(ref,run_waves(det,rows)); qpass=quality_ok(q); cpass=reduction>=MIN_REDUCTION
        item={'convMinimumWeightBytes':threshold,'selectedNames':names,'selectedConvCount':len(names),'selectedFp32ConvWeightBytes':selected_bytes,'optimizedBytes':size,'sizeReductionPercent':reduction*100,'quality':q,'qualityPass':qpass,'compressionPass':cpass,**build}
        # Only spend timing work on candidates that could actually be shipped.
        if qpass and cpass: item['ciHostSpeed']=bench(path); item['eligible']=True; eligible.append((path,item))
        else: item['eligible']=False
        candidates.append(item)
        # largest-first: once a candidate meets both gates, stop. This minimizes
        # the amount of quantized audio decoder while still delivering material size reduction.
        if item['eligible']: break
    except Exception as e:
        candidates.append({'convMinimumWeightBytes':threshold,'selectedNames':names,'selectedConvCount':len(names),'selectedFp32ConvWeightBytes':selected_bytes,'eligible':False,'error':repr(e)})

base={'runtime':ort.__version__,'strategy':'ORT 1.17.1 BASIC fold weight normalization, then largest-first per-channel QInt8 QLinearConv only on decoder/generator Conv; BERT/text/predictor/prosody remain FP32','originalBytes':original_bytes,'foldedBytes':FOLDED.stat().st_size,'foldedQuality':folded_quality,'constantDecoderConvCount':len(constant_decoder_convs),'constantDecoderConvWeightBytes':sum(x[1] for x in constant_decoder_convs),'requiredReductionPercent':MIN_REDUCTION*100,'qualityGates':{'durationExact':True,'minCorrelation':MIN_CORR,'minSnrDb':MIN_SNR,'maxNormalizedRmse':MAX_NRMSE},'candidates':candidates}
if not eligible:
    REPORT.write_text(json.dumps({'status':'rejected_no_qlinearconv_candidate',**base},ensure_ascii=False,indent=2)+'\n'); raise SystemExit('no largest-first QLinearConv candidate met size+quality gates')
path,item=eligible[0]; FINAL.write_bytes(path.read_bytes()); sha=hashlib.sha256(FINAL.read_bytes()).hexdigest(); report={'status':'accepted_material_optimization',**base,'artifact':'kokoro_vi.onnx','optimizedBytes':FINAL.stat().st_size,'sizeReductionPercent':(1-FINAL.stat().st_size/original_bytes)*100,'sha256':sha,'selectedCandidate':item}
REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2)); print('FINAL',FINAL,'SHA256',sha)
