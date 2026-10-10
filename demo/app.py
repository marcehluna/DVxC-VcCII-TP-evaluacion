"""Demo Gradio: compara los detectores entrenados sobre una imagen."""

from __future__ import annotations

import csv
from html import escape

import gradio as gr
from PIL import ImageOps

from demo.inference import MODELS, ROOT, iter_predictions, selected_paths
from demo.metrics import HEADERS, metric_rows
from demo.ocr import OCRReading, iter_readings


PRIMARY_MODEL_NAME = "YOLO26 segmentación"
PRIMARY_MODEL_INDEX = next(index for index, spec in enumerate(MODELS) if spec.name == PRIMARY_MODEL_NAME)
OCR_HEADERS = ["Recorte", "Texto leído por PaddleOCR", "Título del catálogo", "Similitud"]
LIVE_OCR_HEADERS = [
    "Lomo", "Texto OCR", "Confianza OCR", "Rotación", "Título del catálogo",
    "Autor", "Similitud", "Tiempo OCR (ms)",
]


# La interfaz conserva componentes nativos de Gradio para accesibilidad y callbacks,
# pero los presenta como una mesa de consulta de biblioteca.
APP_CSS = """
:root {
    --library-ink: #21352f;
    --library-forest: #315f50;
    --library-copper: #b96845;
    --library-paper-deep: #ebe2d2;
    --library-line: #d8cebd;
    --library-muted: #68736c;
    --library-shadow: 0 16px 42px rgba(50, 46, 38, .08);
}
.gradio-container {
    background: radial-gradient(circle at 8% 5%, rgba(185,104,69,.10), transparent 25rem),
                linear-gradient(180deg, #f8f4ec 0, #f4efe6 46rem, #f8f5ef 100%) !important;
    color: var(--library-ink) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif !important;
    margin: 0 auto !important;
    max-width: none !important;
    padding: 0 0 4rem !important;
}
.gradio-container > .main { gap: 0 !important; }
.library-topbar, .library-hero, .library-flow, .library-section, .library-footer {
    margin-left: auto !important; margin-right: auto !important;
    max-width: 1240px; width: calc(100% - 48px);
}
.library-topbar {
    align-items: center; border-bottom: 1px solid rgba(49,95,80,.22); display: flex;
    justify-content: space-between; padding: 22px 0 16px;
}
.library-brand { align-items: center; display: flex; gap: 12px; }
.library-mark {
    align-items: center; background: #315f50; border-radius: 5px 5px 2px 2px;
    box-shadow: 5px 0 0 #d4aa76, 9px 0 0 var(--library-copper); color: #fffaf0;
    display: flex; font-family: Georgia, serif; font-size: 20px; height: 38px;
    justify-content: center; margin-right: 9px; width: 31px;
}
.library-mark svg { height: 25px; width: 22px; }
.library-brand strong { display: block; font-family: Georgia, serif; font-size: 18px; }
.library-brand small, .library-topbar-note {
    color: var(--library-muted); font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
}
.library-topbar-note { align-items: center; display: flex; gap: 8px; }
.library-topbar-note::before {
    background: #4e8a68; border-radius: 50%; box-shadow: 0 0 0 4px rgba(78,138,104,.12);
    content: ""; height: 7px; width: 7px;
}
.library-hero {
    align-items: end; display: grid; gap: 54px; grid-template-columns: minmax(0,1.6fr) minmax(300px,.8fr);
    padding: 76px 0 56px;
}
.library-kicker, .section-kicker {
    color: var(--library-copper); font-size: 12px; font-weight: 800; letter-spacing: .16em;
    margin: 0 0 16px; text-transform: uppercase;
}
.library-hero h1 {
    color: var(--library-ink); font-family: Georgia, serif; font-size: clamp(44px,6vw,74px);
    font-weight: 500; letter-spacing: -.045em; line-height: .98; margin: 0; max-width: 820px;
}
.library-hero h1 em { color: var(--library-copper); font-weight: 500; }
.library-hero-copy { color: #53625b; font-size: 17px; line-height: 1.7; margin: 24px 0 0; max-width: 700px; }
.library-stats {
    background: #315f50; border-radius: 8px 8px 26px 8px; box-shadow: 16px 18px 0 var(--library-paper-deep);
    color: white; display: grid; grid-template-columns: 1fr 1fr; overflow: hidden;
}
.library-stat { border-bottom: 1px solid rgba(255,255,255,.16); border-right: 1px solid rgba(255,255,255,.16); padding: 24px; }
.library-stat:nth-child(even) { border-right: 0; }
.library-stat:nth-child(n+3) { border-bottom: 0; }
.library-stat strong { display: block; font-family: Georgia, serif; font-size: 32px; font-weight: 500; line-height: 1; }
.library-stat span { color: rgba(255,255,255,.72); display: block; font-size: 11px; letter-spacing: .08em; margin-top: 9px; text-transform: uppercase; }
.library-flow {
    background: #e8dfd0; border: 1px solid var(--library-line); border-radius: 12px; display: grid;
    grid-template-columns: repeat(4,1fr); margin-bottom: 84px; overflow: hidden;
}
.library-flow-item { align-items: center; display: grid; gap: 14px; grid-template-columns: auto 1fr; padding: 18px 22px; }
.library-flow-item + .library-flow-item { border-left: 1px solid var(--library-line); }
.library-flow-item b { color: var(--library-copper); font-family: Georgia, serif; font-size: 25px; font-weight: 500; }
.library-flow-item strong { display: block; font-size: 14px; }
.library-flow-item span { color: var(--library-muted); font-size: 12px; }
.library-section { margin-bottom: 88px !important; }
.section-heading { align-items: end; display: grid; gap: 32px; grid-template-columns: 1fr minmax(280px,470px); margin-bottom: 24px; }
.section-heading h2 {
    color: var(--library-ink); font-family: Georgia, serif; font-size: clamp(32px,4vw,47px);
    font-weight: 500; letter-spacing: -.025em; line-height: 1.05; margin: 0;
}
.section-heading p:last-child { color: var(--library-muted); line-height: 1.65; margin: 0; }
.library-panel { background: rgba(255,253,248,.82) !important; border: 1px solid var(--library-line) !important; border-radius: 14px !important; box-shadow: var(--library-shadow); padding: 18px !important; }
.control-rail { background: #eee6d8 !important; border: 1px solid var(--library-line) !important; border-radius: 10px !important; padding: 18px !important; }
.control-rail .block { background: transparent !important; }
.library-section label > span, .library-section .label-wrap span { color: #4d5b55 !important; font-size: 12px !important; font-weight: 750 !important; letter-spacing: .035em; }
.library-section button { border-radius: 7px !important; font-weight: 720 !important; min-height: 43px; }
.library-section button.primary { background: var(--library-copper) !important; border-color: var(--library-copper) !important; color: white !important; }
.library-section button.primary:hover { background: #9f5638 !important; }
.library-section button.primary:disabled { background: #d9b8a6 !important; border-color: #d9b8a6 !important; }
#historical-metrics { border-radius: 9px !important; }
#historical-metrics table { font-size: 13px !important; }
#historical-metrics th, #historical-metrics td { white-space: normal !important; }
#historical-metrics th { background: #e7ded0 !important; color: var(--library-ink) !important; }
.metric-note { border-left: 3px solid var(--library-copper) !important; color: var(--library-muted) !important; font-size: 13px !important; line-height: 1.55 !important; margin-top: 14px !important; padding: 4px 14px !important; }
.checkpoint-card ul { display: grid; gap: 7px 24px; grid-template-columns: repeat(2,minmax(0,1fr)); }
.model-feature {
    background: #315f50; border-radius: 12px 12px 28px 12px; color: white;
    display: grid; gap: 24px; grid-template-columns: auto 1fr; padding: 26px;
}
.model-feature-badge {
    align-items: center; background: #d07a55; border-radius: 50%; display: flex;
    font-family: Georgia, serif; font-size: 27px; height: 68px; justify-content: center; width: 68px;
}
.model-feature h3 { color: white; font-family: Georgia, serif; font-size: 30px; font-weight: 500; margin: 0 0 6px; }
.model-feature p { color: rgba(255,255,255,.75); line-height: 1.55; margin: 0; }
.model-reasons { display: grid; gap: 1px; grid-template-columns: repeat(3,1fr); margin-top: 18px; }
.model-reason { background: rgba(255,255,255,.08); padding: 14px; }
.model-reason strong { display: block; font-family: Georgia,serif; font-size: 21px; }
.model-reason span { color: rgba(255,255,255,.7); font-size: 11px; text-transform: uppercase; }
.pipeline-card {
    background: #fffdf8; border: 1px solid var(--library-line); border-radius: 10px;
    min-height: 138px; padding: 20px;
}
.pipeline-card b { color: var(--library-copper); font-family: Georgia,serif; font-size: 24px; font-weight: 500; }
.pipeline-card strong { color: var(--library-ink); display: block; font-size: 14px; margin: 12px 0 5px; }
.pipeline-card span { color: var(--library-muted); font-size: 12px; line-height: 1.5; }
.ocr-kpis { display: grid; gap: 12px; grid-template-columns: repeat(4,1fr); margin-bottom: 18px; }
.ocr-kpi { background: #e9e0d2; border: 1px solid var(--library-line); border-radius: 9px; padding: 18px; }
.ocr-kpi strong { color: var(--library-ink); display: block; font-family: Georgia,serif; font-size: 28px; font-weight: 500; }
.ocr-kpi span { color: var(--library-muted); font-size: 11px; letter-spacing: .06em; text-transform: uppercase; }
.ocr-gallery { background: #243d34 !important; border-radius: 10px !important; }
.ocr-gallery .thumbnail-item img { max-height: 260px !important; object-fit: contain !important; }
.model-picker .wrap { gap: 10px !important; }
.model-picker label { background: #fffdf8 !important; border: 1px solid var(--library-line) !important; border-radius: 7px !important; padding: 10px 12px !important; }
.model-picker label:has(input:checked) { background: #e6eee9 !important; border-color: #315f50 !important; box-shadow: inset 3px 0 0 #315f50; }
.advanced-paths { background: rgba(255,253,248,.65) !important; border-radius: 9px !important; margin-top: 14px !important; }
#source-image { background: #fffdf8 !important; border: 1px dashed #af9d85 !important; border-radius: 11px !important; min-height: 410px; overflow: hidden; }
.run-card { background: #315f50 !important; border: 0 !important; border-radius: 12px 12px 28px 12px !important; color: white !important; min-height: 410px; padding: 26px !important; }
.run-card .block, .run-card .form { background: transparent !important; }
.run-card label > span, .run-card .prose, .run-card p { color: rgba(255,255,255,.78) !important; }
.run-card input { color: #21352f !important; }
.run-card button.primary { background: #d07a55 !important; border-color: #d07a55 !important; margin-top: 12px; }
.run-intro h3 { color: white; font-family: Georgia, serif; font-size: 29px; font-weight: 500; margin: 0 0 8px; }
.run-intro p { line-height: 1.6; margin: 0 0 24px; }
.examples-holder { margin-top: 14px !important; }
.run-status { background: #e9e0d2 !important; border: 1px solid var(--library-line) !important; border-radius: 8px !important; color: var(--library-ink) !important; margin-bottom: 16px !important; padding: 12px 16px !important; }
.run-status p { margin: 0 !important; }
#result-gallery { background: #2b3e37 !important; border: 0 !important; border-radius: 12px !important; min-height: 360px; overflow: hidden; }
#result-gallery .thumbnail-item img { max-height: 500px !important; object-fit: contain !important; width: 100% !important; }
.result-table { margin-top: 18px !important; }
.fine-print { color: var(--library-muted) !important; font-size: 12px !important; line-height: 1.6 !important; margin-top: 12px !important; }
.library-footer { border-top: 1px solid var(--library-line); color: var(--library-muted); display: flex; font-size: 12px; justify-content: space-between; padding: 22px 0; }
footer { display: none !important; }
.dark .gradio-container {
    --library-ink: #edf2ec; --library-paper-deep: #28372f; --library-line: #43534a; --library-muted: #afbbb4;
    background: radial-gradient(circle at 8% 5%, rgba(185,104,69,.14), transparent 25rem), linear-gradient(180deg,#18231f 0,#131c19 100%) !important;
}
.dark .library-panel, .dark .library-section .block:not(.html-container) { background-color: #1f2b26 !important; }
.dark .control-rail, .dark .run-status, .dark #historical-metrics th { background: #2a3831 !important; }
.dark .library-hero-copy { color: #b9c5be; }
.dark .library-flow { background: #e8dfd0; color: #21352f; }
.dark .library-flow span { color: #68736c; }
.dark #historical-metrics thead * { background: #2a3831 !important; color: #f7f1e7 !important; }
.dark .model-picker label { background: #24322c !important; }
.dark .model-picker label:has(input:checked) { background: #304a3f !important; }
.dark .advanced-paths, .dark #source-image { background: #1d2924 !important; }
.dark .pipeline-card { background: #24322c; }
.dark .ocr-kpi { background: #2a3831; }
@media (max-width: 850px) {
    .library-topbar, .library-hero, .library-flow, .library-section, .library-footer { width: calc(100% - 28px); }
    .library-topbar-note { display: none; }
    .library-hero { gap: 36px; grid-template-columns: 1fr; padding-top: 52px; }
    .library-flow { grid-template-columns: 1fr; margin-bottom: 64px; }
    .library-flow-item + .library-flow-item { border-left: 0; border-top: 1px solid var(--library-line); }
    .section-heading { align-items: start; grid-template-columns: 1fr; gap: 12px; }
    .checkpoint-card ul { grid-template-columns: 1fr; }
    .model-reasons, .ocr-kpis { grid-template-columns: repeat(2,1fr); }
    .library-section { margin-bottom: 64px !important; }
}
"""


def _section_heading(number: str, title: str, description: str) -> str:
    return (
        '<div class="section-heading">'
        f'<div><p class="section-kicker">{escape(number)}</p><h2>{escape(title)}</h2></div>'
        f'<p>{escape(description)}</p></div>'
    )


def ocr_experiment(root=ROOT) -> tuple[list[list], dict[str, str], list[tuple[str, str]]]:
    """Resume los artefactos generados por notebooks/OCR.ipynb."""
    catalog_path = root / "resultados_ocr" / "comparativa_catalogo.csv"
    comparison_path = root / "resultados_ocr" / "comparativa_ocr.csv"
    crops_dir = root / "resultados_ocr" / "recortes"

    with catalog_path.open(encoding="utf-8", newline="") as handle:
        catalog_rows = list(csv.DictReader(handle))
    with comparison_path.open(encoding="utf-8", newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))

    matched = [row for row in catalog_rows if row.get("paddleocr_coincide", "").lower() == "true"]
    table = [
        [
            row["crop_id"],
            row["paddleocr_texto"],
            row["paddleocr_match_titulo"].title(),
            f'{float(row["paddleocr_match_score"]):.1f}%',
        ]
        for row in matched
    ]
    confidences = [float(row["paddleocr_confianza"]) for row in comparison_rows]
    latencies = [float(row["paddleocr_latencia_ms"]) for row in comparison_rows]
    stats = {
        "crops": str(len(comparison_rows)),
        "confidence": f"{sum(confidences) / len(confidences):.3f}" if confidences else "—",
        "matches": f"{len(matched)}/{len(catalog_rows)}",
        "latency": f"{sum(latencies) / len(latencies):.0f} ms" if latencies else "—",
    }
    gallery = []
    for row in matched:
        crop_path = crops_dir / f'{row["crop_id"]}.jpg'
        if crop_path.is_file():
            gallery.append((str(crop_path), row["paddleocr_match_titulo"].title()))
    return table, stats, gallery


def primary_checkpoint_status(path_override: str) -> str:
    path = MODELS[PRIMARY_MODEL_INDEX].resolve_path(path_override)
    text_style = "color:#7a170e!important"
    if not path.is_file():
        return (
            '<div class="checkpoint-card" role="alert" style="padding:16px 18px;border:1px solid #cf6458;'
            f'border-radius:10px;background:#fff1f0!important;{text_style};font-size:13px;line-height:1.4">'
            f'<div style="{text_style};font-size:18px;font-weight:750">Falta el checkpoint de YOLO26</div>'
            f'<span style="{text_style};font-family:monospace;overflow-wrap:anywhere">{escape(str(path))}</span>'
            f'<div style="{text_style};margin-top:6px">Indicá una ruta válida para analizar estantes.</div></div>'
        )
    return (
        '<div class="checkpoint-card" role="status" style="padding:16px 18px;border:1px solid #4b8b65;'
        'border-radius:10px;background:#effaf1!important;color:#155c31!important;font-size:15px">'
        '<strong>YOLO26 listo.</strong> El segmentador elegido está disponible para analizar el estante.</div>'
    )


def checkpoint_status(*path_overrides: str) -> str:
    paths = selected_paths(path_overrides or None)
    missing = [(spec, path) for spec, path in zip(MODELS, paths) if not path.is_file()]
    if missing:
        text_style = "color:#7a170e!important"
        rows = "".join(
            f'<li style="{text_style};margin:8px 0">'
            f'<span style="{text_style};font-weight:700">{escape(spec.name)}</span><br>'
            f'<span style="{text_style};font-family:monospace;overflow-wrap:anywhere">'
            f'{escape(str(path))}</span></li>'
            for spec, path in missing
        )
        return (
            '<div class="checkpoint-card" role="alert" style="padding:16px 18px;border:1px solid #cf6458;'
            f'border-radius:10px;background:#fff1f0!important;{text_style};font-size:13px;line-height:1.4">'
            f'<div style="{text_style};font-size:18px;font-weight:750;margin-bottom:5px">'
            f'Faltan {len(missing)} de {len(MODELS)} checkpoints</div>'
            f'<div style="{text_style}">Podés explorar las métricas. Para ejecutar, copiá los pesos o editá sus rutas y pulsá '
            f'<span style="{text_style};font-weight:700">Actualizar rutas y comprobar pesos</span>.</div>'
            f'<ul style="{text_style};margin:12px 0 0;padding-left:18px;overflow-wrap:anywhere">'
            f'{rows}</ul></div>'
        )
    return (
        '<div class="checkpoint-card" role="status" style="padding:16px 18px;border:1px solid #4b8b65;'
        'border-radius:10px;background:#effaf1!important;color:#155c31!important;font-size:15px">'
        f'<strong>Catálogo listo.</strong> Los {len(MODELS)} checkpoints están disponibles para comparar.</div>'
    )


def choose_available(*path_overrides):
    return [spec.name for spec, path in zip(MODELS, selected_paths(path_overrides or None))
            if path.is_file()]


def clear_results():
    return [], [], "La imagen o configuración cambió. Ejecutá la selección para ver resultados nuevos."


def inputs_changed(image, model_names):
    return (*clear_results(), gr.Button(interactive=image is not None and bool(model_names)))


def pipeline_inputs_changed(image):
    return (
        [], [], "La imagen o configuración cambió. Analizá el estante nuevamente.",
        [], [], "El OCR se ejecutará después de segmentar los lomos.",
        gr.Button(interactive=image is not None),
    )


def _ocr_row(reading: OCRReading) -> list:
    return [
        reading.number,
        reading.text or "—",
        round(reading.confidence, 3),
        f"{reading.rotation}°",
        reading.catalog_title or "Sin coincidencia",
        reading.catalog_author or "—",
        f"{reading.match_score:.1f}%",
        round(reading.latency_ms, 1),
    ]


def _ocr_caption(reading: OCRReading) -> str:
    if reading.catalog_title:
        return f"Lomo {reading.number} · {reading.catalog_title} · {reading.match_score:.0f}%"
    return f"Lomo {reading.number} · {reading.text or 'sin texto'}"


def run_library_pipeline(image, confidence, max_books, *path_overrides):
    """Ejecuta YOLO26 y luego OCR/matching sobre cada máscara seleccionada."""
    if image is None:
        raise gr.Error("Subí una imagen o elegí un ejemplo.")
    if not path_overrides:
        raise gr.Error("Falta la ruta del checkpoint de YOLO26.")

    context = f"Confianza mínima: {confidence:.2f}."
    status = primary_checkpoint_status(path_overrides[PRIMARY_MODEL_INDEX])
    yield [], [], f"Cargando YOLO26… {context}", status, [], [], "OCR en espera."

    predictions = list(iter_predictions(
        image,
        confidence,
        path_overrides,
        [PRIMARY_MODEL_NAME],
    ))
    prediction = predictions[0]
    detection_gallery, detection_rows = [], []
    if prediction.elapsed_ms is None:
        detection_rows.append([prediction.spec.name, None, None, None, prediction.message])
        yield (
            detection_gallery, detection_rows,
            f"No se pudo ejecutar YOLO26. {context}", status,
            [], [], "OCR omitido porque no hay máscaras disponibles.",
        )
        return

    detection_gallery.append((prediction.image, f"YOLO26 · {prediction.count} lomos"))
    detection_rows.append([
        prediction.spec.name,
        prediction.count,
        round(prediction.mean_score, 3) if prediction.mean_score is not None else None,
        round(prediction.elapsed_ms, 1),
        "OK",
    ])
    detection_message = f"YOLO26 finalizó: {prediction.count} lomos encontrados. {context}"
    yield (
        detection_gallery.copy(), detection_rows.copy(), detection_message, status,
        [], [], "Preparando recortes y cargando PaddleOCR por primera vez…",
    )

    if not prediction.polygons or not prediction.scores:
        yield (
            detection_gallery, detection_rows, detection_message, status,
            [], [], "YOLO26 no produjo máscaras por encima del umbral; no hay lomos para leer.",
        )
        return

    source = ImageOps.exif_transpose(image).convert("RGB")
    ocr_gallery, ocr_rows = [], []
    try:
        for reading in iter_readings(
            source,
            prediction.polygons,
            prediction.scores,
            limit=int(max_books),
        ):
            ocr_gallery.append((reading.crop, _ocr_caption(reading)))
            ocr_rows.append(_ocr_row(reading))
            matches = sum(row[4] != "Sin coincidencia" for row in ocr_rows)
            ocr_message = (
                f"OCR {len(ocr_rows)}/{min(int(max_books), prediction.count)} · "
                f"{matches} coincidencias de catálogo."
            )
            yield (
                detection_gallery.copy(), detection_rows.copy(), detection_message, status,
                ocr_gallery.copy(), ocr_rows.copy(), ocr_message,
            )
    except Exception as exc:
        yield (
            detection_gallery, detection_rows, detection_message, status,
            ocr_gallery, ocr_rows,
            f"OCR interrumpido: {type(exc).__name__}: {exc}",
        )
        return

    matches = sum(row[4] != "Sin coincidencia" for row in ocr_rows)
    yield (
        detection_gallery, detection_rows, detection_message, status,
        ocr_gallery, ocr_rows,
        f"OCR finalizado: {len(ocr_rows)} lomos leídos y {matches} coincidencias de catálogo.",
    )


def run_comparison(image, confidence, model_names, *path_overrides):
    if image is None:
        raise gr.Error("Subí una imagen o elegí un ejemplo.")
    if not model_names:
        raise gr.Error("Elegí al menos un modelo para ejecutar.")
    known_names = {spec.name for spec in MODELS}
    if set(model_names) - known_names:
        raise gr.Error("La selección contiene un modelo desconocido.")
    names = [spec.name for spec in MODELS if spec.name in model_names]
    images, rows = [], []
    context = f"Confianza mínima: {confidence:.2f}. Modelos: {', '.join(names)}."
    status = (primary_checkpoint_status(path_overrides[PRIMARY_MODEL_INDEX])
              if names == [PRIMARY_MODEL_NAME] else checkpoint_status(*path_overrides))
    yield images.copy(), rows.copy(), f"Preparando {names[0]}… {context}", status
    predictions = iter_predictions(image, confidence, path_overrides or None, names)
    for index, prediction in enumerate(predictions):
        if prediction.elapsed_ms is None:
            rows.append([prediction.spec.name, None, None, None, prediction.message])
        else:
            images.append((prediction.image, f"{prediction.spec.name} · {prediction.count} lomos"))
            rows.append([prediction.spec.name, prediction.count,
                         round(prediction.mean_score, 3) if prediction.mean_score is not None else None,
                         round(prediction.elapsed_ms, 1), "OK"])
        done = index + 1
        if done < len(names):
            message = f"{done}/{len(names)} finalizados. Ejecutando {names[done]}…"
        else:
            message = f"Finalizado: {len(images)}/{len(names)} modelos ejecutados correctamente."
        yield images.copy(), rows.copy(), f"{message} {context}", status


def build_app():
    examples_dir = ROOT / "dataset" / "valid" / "images"
    examples = sorted(
        (path for path in examples_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda path: path.name,
    )[:3] if examples_dir.is_dir() else []
    ocr_rows, ocr_stats, ocr_gallery = ocr_experiment()

    with gr.Blocks(title="Lumen · Inventario bibliográfico", css=APP_CSS) as app:
        gr.HTML(f"""
        <header class="library-topbar">
            <div class="library-brand">
                <span class="library-mark" aria-hidden="true">
                    <svg viewBox="0 0 24 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 3h5v21H3zM9.5 6h5v18h-5zM16 2h5v22h-5z" stroke="currentColor" stroke-width="1.5"/>
                        <path d="M1.5 25.5h21" stroke="currentColor" stroke-width="1.5"/>
                        <path d="M5 8h1M11.5 11h1M18 7h1" stroke="currentColor" stroke-width="1.5"/>
                    </svg>
                </span>
                <span><strong>Lumen</strong><small>Inventario bibliográfico</small></span>
            </div>
            <div class="library-topbar-note">YOLO26 · OCR local</div>
        </header>
        <section class="library-hero">
            <div>
                <p class="library-kicker">Visión artificial para bibliotecas</p>
                <h1>Tu biblioteca, leída <em>lomo a lomo.</em></h1>
                <p class="library-hero-copy">Lumen segmenta cada libro con YOLO26, endereza su lomo,
                recupera el texto con OCR y lo contrasta con el catálogo de la colección.</p>
            </div>
            <div class="library-stats" aria-label="Resumen del laboratorio">
                <div class="library-stat"><strong>YOLO26</strong><span>modelo elegido</span></div>
                <div class="library-stat"><strong>{escape(ocr_stats['crops'])}</strong><span>lomos evaluados</span></div>
                <div class="library-stat"><strong>{escape(ocr_stats['confidence'])}</strong><span>confianza OCR</span></div>
                <div class="library-stat"><strong>29</strong><span>libros en catálogo</span></div>
            </div>
        </section>
        <div class="library-flow" aria-label="Flujo del inventario visual">
            <div class="library-flow-item"><b>01</b><span><strong>Segmenta</strong>YOLO26s-seg</span></div>
            <div class="library-flow-item"><b>02</b><span><strong>Endereza</strong>Recorte por máscara</span></div>
            <div class="library-flow-item"><b>03</b><span><strong>Lee</strong>PaddleOCR</span></div>
            <div class="library-flow-item"><b>04</b><span><strong>Identifica</strong>Catálogo local</span></div>
        </div>
        """)

        with gr.Group(elem_classes=["library-section"]):
            gr.HTML(_section_heading(
                "01 · Modelo elegido", "Una decisión, no un selector",
                "YOLO26s-seg ofrece el mejor equilibrio entre calidad de máscara y velocidad. Su recorte ajustado también es la entrada natural del OCR.",
            ))
            gr.HTML("""
                <div class="model-feature">
                    <div class="model-feature-badge">Y26</div>
                    <div><h3>YOLO26s-seg</h3>
                    <p>Segmentación de instancias en una sola etapa. Delimita el lomo a nivel de píxel,
                    reduce el fondo del recorte y opera muy por debajo de la latencia de Mask R-CNN.</p>
                    <div class="model-reasons">
                        <div class="model-reason"><strong>0.761</strong><span>mAP@0.5 máscara</span></div>
                        <div class="model-reason"><strong>~18 ms</strong><span>latencia registrada</span></div>
                        <div class="model-reason"><strong>Máscara</strong><span>recorte para OCR</span></div>
                    </div></div>
                </div>
            """)
            with gr.Accordion("Ver evidencia comparativa de segmentación", open=False, elem_classes=["advanced-paths"]):
                partition = gr.Radio(["Validación", "Test"], value="Validación", label="Partición")
                initial_rows, initial_note = metric_rows("Validación", "Máscaras")
                metrics = gr.Dataframe(
                    value=initial_rows, headers=HEADERS, datatype="str", interactive=False,
                    label="Métricas de máscara", elem_id="historical-metrics",
                )
                metric_note = gr.Markdown(initial_note, elem_classes=["metric-note"])
                reload_metrics = gr.Button("Actualizar desde informes")
            partition.change(lambda value: metric_rows(value, "Máscaras"), inputs=partition,
                             outputs=[metrics, metric_note])
            reload_metrics.click(lambda value: metric_rows(value, "Máscaras"), inputs=partition,
                                 outputs=[metrics, metric_note])

        with gr.Group(elem_classes=["library-section"]):
            gr.HTML(_section_heading(
                "02 · Pipeline bibliográfico", "De una foto a un título",
                "El flujo ya fue probado sobre dos estantes propios. Cada etapa conserva una responsabilidad clara y un artefacto verificable.",
            ))
            with gr.Row(equal_height=True):
                gr.HTML('<div class="pipeline-card"><b>01</b><strong>Segmentación</strong><span>YOLO26 encuentra cada instancia y produce su máscara.</span></div>')
                gr.HTML('<div class="pipeline-card"><b>02</b><strong>Recorte orientado</strong><span>El polígono se endereza y mejora con contraste CLAHE.</span></div>')
                gr.HTML('<div class="pipeline-card"><b>03</b><strong>Lectura OCR</strong><span>PaddleOCR prueba orientaciones y conserva la lectura más confiable.</span></div>')
                gr.HTML('<div class="pipeline-card"><b>04</b><strong>Catálogo</strong><span>RapidFuzz compara el texto contra 29 títulos con umbral 85.</span></div>')

        default_paths = [str(spec.path) for spec in MODELS]
        with gr.Group(elem_classes=["library-section"]):
            gr.HTML(_section_heading(
                "03 · Mesa de consulta", "Traé un estante",
                "Subí una imagen y ajustá la sensibilidad. Lumen ejecuta siempre YOLO26s-seg, el modelo seleccionado por el experimento.",
            ))
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    source = gr.Image(
                        label="Imagen de estantería", type="pil", sources=["upload", "clipboard", "webcam"],
                        elem_id="source-image", height=410,
                    )
                    if examples:
                        with gr.Group(elem_classes=["examples-holder"]):
                            gr.Examples(
                                examples=[[str(path)] for path in examples], inputs=source,
                                label="Ejemplos de validación",
                            )
                with gr.Column(scale=2, elem_classes=["run-card"]):
                    gr.HTML(
                        '<div class="run-intro"><h3>Analizar con YOLO26</h3>'
                        '<p>La primera ejecución puede tardar más mientras se carga el checkpoint.</p></div>'
                    )
                    status = gr.HTML(value=primary_checkpoint_status(default_paths[PRIMARY_MODEL_INDEX]))
                    confidence = gr.Slider(0.05, 0.95, value=0.50, step=0.05, label="Confianza mínima")
                    max_books = gr.Slider(
                        1, 60, value=10, step=1, label="Máximo de lomos para OCR",
                        info="Se priorizan las detecciones con mayor confianza.",
                        elem_id="max-ocr-books",
                    )
                    gr.Markdown(
                        "**0.30–0.45** · exploración con más candidatos  \n"
                        "**0.50–0.70** · lectura más conservadora", elem_classes=["fine-print"],
                    )
                    run = gr.Button("Analizar estante", variant="primary", interactive=False)
                    with gr.Accordion("Cambiar checkpoint", open=False, elem_classes=["advanced-paths"]):
                        primary_path = gr.Textbox(
                            label="Checkpoint de YOLO26s-seg",
                            value=default_paths[PRIMARY_MODEL_INDEX],
                        )
                        refresh = gr.Button("Comprobar checkpoint")
            refresh.click(primary_checkpoint_status, inputs=primary_path, outputs=status)

        path_inputs = [gr.State(path) if index != PRIMARY_MODEL_INDEX else primary_path
                       for index, path in enumerate(default_paths)]

        with gr.Group(elem_classes=["library-section"]):
            gr.HTML(_section_heading(
                "04 · Detecciones", "Los lomos encontrados",
                "Abrí la imagen para revisar las máscaras. El resumen informa cuántos lomos encontró YOLO26 y cuánto demoró la inferencia local.",
            ))
            run_status = gr.Markdown("Todavía no se analizó ningún estante.", elem_classes=["run-status"])
            gallery = gr.Gallery(
                label="Máscaras de YOLO26 — abrí la imagen para ampliarla", columns=1,
                height="auto", preview=False, object_fit="contain", interactive=False,
                elem_id="result-gallery",
            )
            table = gr.Dataframe(
                headers=["Modelo", "Lomos", "Confianza media", "Tiempo local (ms)", "Estado"],
                datatype=["str", "number", "number", "number", "str"], interactive=False,
                label="Resumen", elem_classes=["result-table"],
            )
            gr.Markdown(
                "El tiempo local excluye carga de pesos y dibujo. La confianza media describe las detecciones "
                "de esta imagen; no reemplaza las métricas del conjunto de validación.", elem_classes=["fine-print"],
            )

        with gr.Group(elem_classes=["library-section"]):
            gr.HTML(_section_heading(
                "05 · OCR en vivo", "Lecturas de este estante",
                "Cada máscara se transforma en un recorte orientado. PaddleOCR prueba 0°/90°/270°, conserva la lectura más confiable y RapidFuzz consulta el catálogo.",
            ))
            live_ocr_status = gr.Markdown(
                "El OCR se ejecutará después de segmentar los lomos.",
                elem_classes=["run-status"],
            )
            live_ocr_gallery = gr.Gallery(
                label="Lomos leídos — abrí un recorte para ampliarlo",
                columns=4,
                height="auto",
                preview=False,
                object_fit="contain",
                interactive=False,
                elem_classes=["ocr-gallery"],
            )
            live_ocr_table = gr.Dataframe(
                headers=LIVE_OCR_HEADERS,
                datatype=["number", "str", "number", "str", "str", "str", "str", "number"],
                interactive=False,
                label="OCR y coincidencias contra catalogo/catalogo.xlsx",
            )
            gr.Markdown(
                "La primera ejecución descarga y prepara los modelos de PaddleOCR. Los lomos sin similitud "
                "mínima de 85 se conservan como lecturas, pero se marcan sin coincidencia.",
                elem_classes=["fine-print"],
            )

        with gr.Group(elem_classes=["library-section"]):
            gr.HTML(_section_heading(
                "06 · Evidencia OCR", "Resultados del experimento",
                "Resultados registrados por notebooks/OCR.ipynb sobre 19 recortes de dos fotos propias. PaddleOCR fue el motor más confiable y se usa como referencia del pipeline.",
            ))
            gr.HTML(
                '<div class="ocr-kpis">'
                f'<div class="ocr-kpi"><strong>{escape(ocr_stats["crops"])}</strong><span>recortes procesados</span></div>'
                f'<div class="ocr-kpi"><strong>{escape(ocr_stats["confidence"])}</strong><span>confianza media</span></div>'
                f'<div class="ocr-kpi"><strong>{escape(ocr_stats["matches"])}</strong><span>matches ≥ 85</span></div>'
                f'<div class="ocr-kpi"><strong>{escape(ocr_stats["latency"])}</strong><span>latencia media</span></div>'
                '</div>'
            )
            if ocr_gallery:
                gr.Gallery(
                    value=ocr_gallery, label="Recortes identificados en el catálogo", columns=4,
                    height="auto", preview=False, object_fit="contain", interactive=False,
                    elem_classes=["ocr-gallery"],
                )
            gr.Dataframe(
                value=ocr_rows, headers=OCR_HEADERS, datatype="str", interactive=False,
                label="Lecturas de PaddleOCR con coincidencia de catálogo",
            )
            gr.Markdown(
                "Estos resultados pertenecen al experimento ya ejecutado, no a la fotografía cargada arriba. "
                "La notebook prueba rotaciones 0°/90°/270°, elige la lectura de mayor confianza y aplica "
                "matching `token_set_ratio` con umbral 85.", elem_classes=["fine-print"],
            )
        execution = run.click(
            run_library_pipeline,
            inputs=[source, confidence, max_books, *path_inputs],
            outputs=[gallery, table, run_status, status,
                     live_ocr_gallery, live_ocr_table, live_ocr_status],
            concurrency_limit=1,
        )
        for control in (source, confidence, max_books, primary_path):
            control.change(
                pipeline_inputs_changed,
                inputs=source,
                outputs=[gallery, table, run_status,
                         live_ocr_gallery, live_ocr_table, live_ocr_status, run],
                cancels=[execution],
                queue=False,
            )
        refresh.click(
            pipeline_inputs_changed,
            inputs=source,
            outputs=[gallery, table, run_status,
                     live_ocr_gallery, live_ocr_table, live_ocr_status, run],
            cancels=[execution],
            queue=False,
        )
        gr.HTML(
            '<div class="library-footer"><span>Lumen · Inventario bibliográfico visual</span>'
            '<span>YOLO26s-seg · PaddleOCR · catálogo local</span></div>'
        )
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", max_file_size="20mb")
