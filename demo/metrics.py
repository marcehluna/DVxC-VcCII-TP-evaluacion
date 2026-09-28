"""Lee métricas históricas del repo sin cargar checkpoints ni ejecutar modelos."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

from demo.inference import ROOT


REPORTS = {
    "YOLOv8 AABB": "entrenamientos/YOLO8-AABB.md",
    "Faster R-CNN": "entrenamientos/Faster-R-CNN.md",
    "YOLO26 segmentación": "entrenamientos/YOLO26s-seg.md",
}
HEADERS = ["Modelo", "Salida", "mAP@0.5", "mAP@0.5:0.95", "Precisión", "Recall",
           "Latencia registrada (ms)", "Fuente"]


def _number(text: str) -> float | None:
    match = re.search(r"\d+(?:[.,]\d+)?", text)
    return float(match[0].replace(",", ".")) if match else None


def _report_metrics(path: Path, partition: str, geometry: str) -> dict:
    text = path.read_text(encoding="utf-8")
    title = "Validación" if partition == "Validación" else "Test"
    section = next((s for s in re.split(r"(?m)^#### ", text) if s.startswith(title)), "")
    values = {}
    # La tabla de test tiene columnas Métrica | Validación | Test.
    value_index = 1 if partition == "Validación" else 2
    is_segmenter = path.name == "YOLO26s-seg.md"
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().replace("*", "") for cell in line.strip("|").split("|")]
        if len(cells) <= value_index:
            continue
        metric = cells[0].lower()
        if metric.startswith("latencia"):
            values["latency"] = cells[value_index].replace(",", ".")
            continue
        row_geometry = "Máscaras" if "máscara" in metric else "Cajas"
        if row_geometry != geometry or (geometry == "Máscaras" and not is_segmenter):
            continue
        key = ("map95" if "map@0.5:0.95" in metric else
               "map50" if "map@0.5" in metric else
               "precision" if metric.startswith("precisión") else
               "recall" if metric.startswith("recall") else None)
        if key:
            values[key] = _number(cells[value_index])
    return values


def _format(value) -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.3f}"
    return "—"


def metric_rows(partition: str = "Validación", geometry: str = "Cajas",
                root: Path = ROOT) -> tuple[list[list], str]:
    if partition not in {"Validación", "Test"} or geometry not in {"Cajas", "Máscaras"}:
        raise ValueError("Partición o tipo de salida desconocido.")
    rows, warnings = [], []
    for name, relative in REPORTS.items():
        if geometry == "Máscaras" and name != "YOLO26 segmentación":
            continue
        try:
            values = _report_metrics(root / relative, partition, geometry)
            if "map50" not in values:
                warnings.append(f"No hay métricas de {geometry.lower()} en {relative}.")
        except (OSError, UnicodeError) as exc:
            values = {}
            warnings.append(f"No se pudo leer {relative}: {type(exc).__name__}.")
        rows.append([name, geometry, *[_format(values.get(k)) for k in
                     ("map50", "map95", "precision", "recall")], values.get("latency", "—"), relative])

    suffix = "validacion" if partition == "Validación" else "test"
    relative = f"results/tables/mask_rcnn_resumen_{suffix}.json"
    try:
        report = json.loads((root / relative).read_text(encoding="utf-8"))
        results = report["resultados"]
        values = results["global"] if partition == "Validación" else results
        prefix = "box" if geometry == "Cajas" else "mask"
        scores = [_format(values.get(f"{prefix}_{key}")) for key in
                  ("map50", "map50_95", "precision", "recall")]
        latency = _format(values.get("latencia_ms"))
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        scores, latency = ["—"] * 4, "—"
        warnings.append(f"No se pudo leer {relative}: {type(exc).__name__}.")
    rows.append(["Mask R-CNN", geometry, *scores, latency, relative])
    note = (
        "Resultados históricos de Book Spine 2 v4. Los valores son de las corridas documentadas; "
        "cambiar la ruta de un checkpoint no actualiza estas métricas. "
        "mAP, precisión y recall se muestran entre 0 y 1; — indica un dato no disponible. "
        "Las latencias de los informes corresponden a GPU Tesla T4 (~ significa aproximado). "
        "Los protocolos de evaluación pueden diferir: consultá la fuente antes de comparar. "
        "YOLO26 no informa precisión/recall globales de máscara en estas tablas."
    )
    if warnings:
        note += "\n\n**Fuentes faltantes o incompletas:**\n\n" + "\n".join(f"- {w}" for w in warnings)
    return rows, note
