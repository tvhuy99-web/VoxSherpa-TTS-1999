#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
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

# Representative Kokoro token lengths seen in real TalkBack/device logs.
LENGTHS = [21, 32, 45, 50, 71, 110, 112]
SPEEDS = [0.90, 1.00, 1.10]
# Valid Kokoro vocabulary ids (0 is boundary/pad; body is deliberately varied).
BODY = [43, 56, 51, 16, 62, 60, 47, 56, 16, 55, 57, 62, 16, 53, 50, 57, 60, 57, 4]


def make_ids(n: int) -> np.ndarray:
    n = max(3, min(510, int(n)))
    body_n = n - 2
    body = [BODY[i % len(BODY)] for i in range(body_n)]
    return np.asarray([[0, *body, 0]], dtype=np.int64)


VOICE_ROWS = np.fromfile(VOICE, dtype="<f4").reshape(510, 256)


def style_for(ids: np.ndarray) -> np.ndarray:
    # App uses row=(phonemeCount clamped 1..510)-1. Using token count here is a
    # conservative calibration approximation and spans the same style table.
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


def metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    test = np.asarray(test, dtype=np.float32).reshape(-1)
    exact = len(ref) == len(test)
    m = min(len(ref), len(test))
    if m == 0:
        return {"durationExact": exact, "corr": -1.0, "snrDb": -999.0, "nrmse": 999.0,
                "refSamples": len(ref), "testSamples": len(test)}
    a, b = ref[:m], test[:m]
    diff = a - b
    rmse = float(np.sqrt(np.mean(diff * diff)))
    rms = float(np.sqrt(np.mean(a * a))) + 1e-12
    nrmse = rmse / rms
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    signal = float(np.sum(a * a)) + 1e-20
    noise = float(np.sum(diff * diff)) + 1e-20
    snr = 10.0 * math.log10(signal / noise)
    return {"durationExact": exact, "corr": corr, "snrDb": snr, "nrmse": nrmse,
            "refSamples": len(ref), "testSamples": len(test)}


def evaluate(model: Path) -> dict:
    base = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
    cand = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    probes = []
    for n in LENGTHS:
        x = feed(n, 1.0)
        y0 = base.run(["waveform"], x)[0]
        y1 = cand.run(["waveform"], x)[0]
        q = metrics(y0, y1)
        q["tokens"] = n
        probes.append(q)
    return {
        "durationExactAll": all(p["durationExact"] for p in probes),
        "minCorrelation": min(p["corr"] for p in probes),
        "minSnrDb": min(p["snrDb"] for p in probes),
        "maxNormalizedRmse": max(p["nrmse"] for p in probes),
        "probes": probes,
    }


def accepted(q: dict, activation: str) -> bool:
    return (q["durationExactAll"] and q["minCorrelation"] >= 0.985
            and q["minSnrDb"] >= 16.0 and q["maxNormalizedRmse"] <= 0.16)


def _subgraphs(node):
    for attr in node.attribute:
        if attr.type == AttributeProto.GRAPH:
            yield attr.g
        elif attr.type == AttributeProto.GRAPHS:
            yield from attr.graphs


def _free_refs(graph) -> set[str]:
    """Names consumed from an enclosing lexical scope, including nested subgraphs."""
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
    """Stable topological sort that accounts for ONNX control-flow lexical captures."""
    moved = 0
    for i, node in enumerate(graph.node):
        for j, sg in enumerate(_subgraphs(node)):
            moved += _sort_graph(sg, f"{path}/{node.name or node.op_type}:subgraph{j}")

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

    ready = [i for i, d in enumerate(deps) if not d]
    ready.sort()
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


def repair_quantized_topology(model_path: Path) -> int:
    """Repair quantizer ordering around Loop/SequenceAt without changing graph semantics."""
    model = onnx.load(str(model_path), load_external_data=True)
    moved = _sort_graph(model.graph, "graph")
    onnx.save_model(model, str(model_path))
    # The checker stays mandatory: repair is accepted only if the resulting model is valid.
    onnx.checker.check_model(onnx.load(str(model_path), load_external_data=True), full_check=True)
    return moved


def build_candidate(name: str, activation_type: QuantType) -> tuple[Path, dict]:
    pre = WORK / f"kokoro.{name}.pre.onnx"
    changed = qnn_preprocess_model(str(FP32), str(pre))
    source = pre if changed else FP32
    # Validate the source independently so a preprocessing regression is never hidden.
    onnx.checker.check_model(onnx.load(str(source), load_external_data=True))

    dst = WORK / f"kokoro.{name}.qdq.onnx"
    reader = Reader()
    cfg = get_qnn_qdq_config(
        str(source), reader,
        activation_type=activation_type,
        weight_type=QuantType.QUInt8,
        per_channel=True,
        activation_symmetric=False,
        weight_symmetric=False,
    )
    quantize(str(source), str(dst), cfg)
    moved = repair_quantized_topology(dst)
    print(f"TOPOLOGY_REPAIR candidate={name} moved={moved}")

    q = evaluate(dst)
    q["bytes"] = dst.stat().st_size
    q["activationType"] = name
    q["topologyNodesMoved"] = moved
    return dst, q


def main():
    if not FP32.is_file() or not VOICE.is_file():
        raise SystemExit("Kokoro FP32 model / Diem Trinh voicepack missing")

    report = {"fp32Bytes": FP32.stat().st_size, "candidates": []}
    selected = None

    # Prefer fastest HTP-friendly 8-bit activations. Fall back to 16-bit activations
    # only if the all-8-bit graph changes duration/waveform beyond the quality gate.
    for name, qt in [("u8u8", QuantType.QUInt8), ("u16u8", QuantType.QUInt16)]:
        try:
            path, q = build_candidate(name, qt)
            q["accepted"] = accepted(q, name)
            report["candidates"].append(q)
            print(json.dumps(q, ensure_ascii=False, indent=2))
            if q["accepted"]:
                selected = (path, q)
                break
        except Exception as e:
            report["candidates"].append({"activationType": name, "accepted": False, "error": repr(e)})
            print(f"Candidate {name} failed: {e!r}")

    if selected is None:
        (WORK / "quant_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit("No QNN HTP QDQ candidate met quality/duration gates")

    path, q = selected
    OUT.write_bytes(path.read_bytes())
    report["selected"] = q
    report["selectedModel"] = str(OUT)
    report["selectedBytes"] = OUT.stat().st_size
    (WORK / "quant_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SELECTED_QNN_HTP_MODEL", json.dumps(q, ensure_ascii=False))


if __name__ == "__main__":
    main()
