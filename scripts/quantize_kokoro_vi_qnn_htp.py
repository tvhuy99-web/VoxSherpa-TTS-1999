#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "app/src/main/assets/kokoro_vi"
FP32 = ASSET / "kokoro_vi.onnx"
VOICE = ASSET / "voicepacks/diem_trinh.f32le"
OUT = ASSET / "kokoro_vi_qdq.onnx"
WORK = ROOT / "build/qnn-htp-quant"
WORK.mkdir(parents=True, exist_ok=True)

# Representative token lengths and speech-rate settings from the real device workload.
LENGTHS = [21, 32, 45, 50, 71, 110, 112]
SPEEDS = [0.90, 1.00, 1.10]
BODY = [43, 56, 51, 16, 62, 60, 47, 56, 16, 55, 57, 62, 16, 53, 50, 57, 60, 57, 4]


def make_ids(n: int) -> np.ndarray:
    n = max(3, min(510, int(n)))
    body = [BODY[i % len(BODY)] for i in range(n - 2)]
    return np.asarray([[0, *body, 0]], dtype=np.int64)


VOICE_ROWS = np.fromfile(VOICE, dtype="<f4").reshape(510, 256)


def style_for(ids: np.ndarray) -> np.ndarray:
    row = max(1, min(510, ids.shape[1])) - 1
    return VOICE_ROWS[row : row + 1].astype(np.float32, copy=True)


def feed(n: int, speed: float = 1.0) -> dict[str, np.ndarray]:
    ids = make_ids(n)
    return {
        "input_ids": ids,
        "ref_s": style_for(ids),
        "speed": np.asarray(speed, dtype=np.float32),
    }


class Reader(CalibrationDataReader):
    def __init__(self):
        self.samples = [feed(n, s) for n in LENGTHS for s in SPEEDS]
        self.it = None

    def get_next(self):
        if self.it is None:
            self.it = iter(self.samples)
        return next(self.it, None)

    def rewind(self):
        self.it = None


def _subgraphs(node):
    for attr in node.attribute:
        if attr.type == AttributeProto.GRAPH:
            yield attr.g
        elif attr.type == AttributeProto.GRAPHS:
            yield from attr.graphs


def _free_refs(graph) -> set[str]:
    local_defs = {x.name for x in graph.input}
    local_defs.update(x.name for x in graph.initializer)
    local_defs.update(out for node in graph.node for out in node.output if out)
    used: set[str] = set()
    for node in graph.node:
        used.update(x for x in node.input if x)
        for sg in _subgraphs(node):
            used.update(_free_refs(sg))
    return used - local_defs


def _sort_graph(graph, path: str) -> int:
    """Stable topological sort including lexical captures used by Loop subgraphs."""
    moved = 0
    for node in graph.node:
        for i, sg in enumerate(_subgraphs(node)):
            moved += _sort_graph(sg, f"{path}/{node.name or node.op_type}:subgraph{i}")

    nodes = list(graph.node)
    if len(nodes) < 2:
        return moved

    producer: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for out in node.output:
            if out:
                producer[out] = i

    deps: list[set[int]] = []
    reverse: list[set[int]] = [set() for _ in nodes]
    for i, node in enumerate(nodes):
        names = {x for x in node.input if x}
        for sg in _subgraphs(node):
            names.update(_free_refs(sg))
        d = {producer[name] for name in names if name in producer and producer[name] != i}
        deps.append(d)
        for p in d:
            reverse[p].add(i)

    ready = sorted(i for i, d in enumerate(deps) if not d)
    order: list[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for consumer in sorted(reverse[i]):
            if i in deps[consumer]:
                deps[consumer].remove(i)
                if not deps[consumer] and consumer not in order and consumer not in ready:
                    ready.append(consumer)
        ready.sort()

    if len(order) != len(nodes):
        unresolved = [nodes[i].name or nodes[i].op_type for i, d in enumerate(deps) if d]
        raise RuntimeError(f"Topological repair found a dependency cycle in {path}: {unresolved[:12]}")

    if order != list(range(len(nodes))):
        sorted_nodes = [nodes[i] for i in order]
        del graph.node[:]
        graph.node.extend(sorted_nodes)
        moved += sum(1 for new_i, old_i in enumerate(order) if new_i != old_i)
    return moved


def _count_ops(graph, counts: Counter[str]) -> None:
    for node in graph.node:
        counts[node.op_type] += 1
        for sg in _subgraphs(node):
            _count_ops(sg, counts)


def repair_and_validate(model_path: Path) -> dict:
    model = onnx.load(str(model_path), load_external_data=True)
    moved = _sort_graph(model.graph, "graph")
    onnx.save_model(model, str(model_path))

    checked = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(checked, full_check=True)
    counts: Counter[str] = Counter()
    _count_ops(checked.graph, counts)
    q = counts["QuantizeLinear"]
    dq = counts["DequantizeLinear"]
    if q <= 0 or dq <= 0:
        raise RuntimeError(f"QNN QDQ model contains no usable Q/DQ nodes: Q={q}, DQ={dq}")
    return {
        "structuralValid": True,
        "topologyNodesMoved": moved,
        "quantizeLinearNodes": q,
        "dequantizeLinearNodes": dq,
        "convNodes": counts["Conv"],
        "matMulNodes": counts["MatMul"],
        "gemmNodes": counts["Gemm"],
        "loopNodes": counts["Loop"],
    }


def main():
    if not FP32.is_file() or not VOICE.is_file():
        raise SystemExit("Kokoro FP32 model / Diem Trinh voicepack missing")

    # Follow the QNN EP reference path: uint16 activations + uint8 weights and let
    # get_qnn_qdq_config choose its normal QNN-safe defaults. Host CPU execution is
    # deliberately not a quality oracle for this QNN-specific graph; the target APK
    # performs the real HTP session/warm-up/A-B test on device.
    name = "u16u8"
    pre = WORK / "kokoro.u16u8.pre.onnx"
    changed = qnn_preprocess_model(str(FP32), str(pre))
    source = pre if changed else FP32
    onnx.checker.check_model(onnx.load(str(source), load_external_data=True))

    dst = WORK / "kokoro.u16u8.qdq.onnx"
    cfg = get_qnn_qdq_config(
        str(source),
        Reader(),
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
    )
    quantize(str(source), str(dst), cfg)
    structural = repair_and_validate(dst)
    structural.update({
        "activationType": name,
        "weightType": "u8",
        "bytes": dst.stat().st_size,
        "hostCpuQualityEvaluation": "deferred_to_target_qnn_htp",
        "deviceQualityRequired": True,
    })

    OUT.write_bytes(dst.read_bytes())
    report = {
        "fp32Bytes": FP32.stat().st_size,
        "selected": structural,
        "selectedModel": str(OUT),
        "selectedBytes": OUT.stat().st_size,
        "qualityGate": "target_device_qnn_htp",
    }
    (WORK / "quant_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SELECTED_QNN_HTP_MODEL", json.dumps(structural, ensure_ascii=False))


if __name__ == "__main__":
    main()
