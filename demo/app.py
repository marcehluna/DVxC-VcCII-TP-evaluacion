"""Demo Gradio: compara los detectores entrenados sobre una imagen."""

from __future__ import annotations

from html import escape

import gradio as gr

from demo.inference import MODELS, ROOT, iter_predictions, selected_paths
from demo.metrics import HEADERS, metric_rows


# Gradio puede dimensionar la miniatura por su ancho y recortarla verticalmente.
# Se limita la imagen a la altura de la galería sin afectar la vista ampliada.
GALLERY_CSS = """
#result-gallery .thumbnail-item img {
    width: 100% !important;
    max-height: 500px !important;
    object-fit: contain !important;
}
"""


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
            '<div role="alert" style="padding:20px 24px;border:3px solid #b42318;'
            f'border-radius:12px;background:#fff1f0!important;{text_style};font-size:17px;line-height:1.5">'
            f'<div style="{text_style};font-size:24px;font-weight:700;margin-bottom:8px">'
            f'⚠ Faltan {len(missing)} de {len(MODELS)} checkpoints</div>'
            f'<div style="{text_style}">Copiá los pesos o editá sus rutas y pulsá '
            f'<span style="{text_style};font-weight:700">Actualizar rutas y comprobar pesos</span>.</div>'
            f'<ul style="{text_style};margin:12px 0 0;padding-left:24px;overflow-wrap:anywhere">'
            f'{rows}</ul></div>'
        )
    return (
        '<div role="status" style="padding:16px 20px;border:2px solid #17803d;'
        'border-radius:12px;background:#effaf1!important;color:#155c31!important;font-size:20px">'
        f'✓ Los {len(MODELS)} checkpoints están disponibles. Podés comparar los modelos.</div>'
    )


def choose_available(*path_overrides):
    return [spec.name for spec, path in zip(MODELS, selected_paths(path_overrides or None))
            if path.is_file()]


def clear_results():
    return [], [], "La imagen o configuración cambió. Ejecutá la selección para ver resultados nuevos."


def inputs_changed(image, model_names):
    return (*clear_results(), gr.Button(interactive=image is not None and bool(model_names)))


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
    status = checkpoint_status(*path_overrides)
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

    with gr.Blocks(title="Comparador de lomos", css=GALLERY_CSS) as app:
        gr.Markdown("# Laboratorio de detección de lomos\nConsultá las métricas y probá una foto con uno o varios modelos.")
        with gr.Accordion("1. Métricas registradas — antes de ejecutar", open=True):
            with gr.Row():
                partition = gr.Radio(["Validación", "Test"], value="Validación", label="Partición")
                geometry = gr.Radio(["Cajas", "Máscaras"], value="Cajas", label="Tipo de evaluación")
            initial_rows, initial_note = metric_rows()
            metrics = gr.Dataframe(value=initial_rows, headers=HEADERS, datatype="str",
                                   interactive=False, label="Resultados del dataset", elem_id="historical-metrics")
            metric_note = gr.Markdown(initial_note)
            reload_metrics = gr.Button("Actualizar métricas desde los informes")
            for control in (partition, geometry):
                control.change(metric_rows, inputs=[partition, geometry], outputs=[metrics, metric_note])
            reload_metrics.click(metric_rows, inputs=[partition, geometry], outputs=[metrics, metric_note])
            gr.Markdown("**mAP** resume la calidad de las detecciones. **Precisión** indica qué proporción "
                        "de las detecciones es correcta; **recall**, cuántos lomos reales se recuperan. "
                        "Una foto nueva sin etiquetas no permite calcular estas métricas.")
        gr.Markdown("## 2. Elegí los pesos y los modelos")
        default_paths = [str(spec.path) for spec in MODELS]
        status = gr.HTML(value=checkpoint_status(*default_paths))
        with gr.Accordion("Rutas de checkpoints (editables)", open=False):
            path_inputs = [gr.Textbox(label=spec.name, value=path)
                           for spec, path in zip(MODELS, default_paths)]
        refresh = gr.Button("Actualizar rutas y comprobar pesos")
        refresh.click(checkpoint_status, inputs=path_inputs, outputs=status)
        model_names = gr.CheckboxGroup(choices=[spec.name for spec in MODELS],
                                       value=choose_available(*default_paths),
                                       label="Modelos a ejecutar",
                                       info="Marcá uno para una prueba individual o varios para compararlos.")
        with gr.Row():
            select_available = gr.Button("Seleccionar disponibles")
            select_all = gr.Button("Seleccionar todos")
        select_available.click(choose_available, inputs=path_inputs, outputs=model_names)
        select_all.click(lambda: [spec.name for spec in MODELS], outputs=model_names)
        gr.Markdown("## 3. Probá una imagen")
        with gr.Row():
            source = gr.Image(label="Imagen de estantería", type="pil", sources=["upload", "clipboard", "webcam"],
                              elem_id="source-image")
            with gr.Column():
                confidence = gr.Slider(0.05, 0.95, value=0.50, step=0.05, label="Confianza mínima")
                gr.Markdown("Un umbral menor puede recuperar más lomos y sumar falsos positivos. "
                            "La primera ejecución incluye la preparación del modelo y puede tardar más.")
                run = gr.Button("Ejecutar selección", variant="primary", interactive=False)
        if examples:
            gr.Examples(examples=[[str(path)] for path in examples], inputs=source, label="Ejemplos de validación")
        gr.Markdown("## Resultados de la ejecución")
        run_status = gr.Markdown("Todavía no se ejecutó ningún modelo.")
        gallery = gr.Gallery(label="Detecciones — abrí una imagen para ampliarla o descargarla",
                             columns=2, height="auto", preview=False, object_fit="contain",
                             interactive=False, elem_id="result-gallery")
        table = gr.Dataframe(headers=["Modelo", "Lomos", "Confianza media", "Tiempo local (ms)", "Estado"],
                             datatype=["str", "number", "number", "number", "str"],
                             interactive=False, label="Resumen")
        gr.Markdown("Los tiempos locales son orientativos: excluyen carga de pesos y dibujo; pueden incluir "
                    "preparación de la primera inferencia y no son un benchmark comparable entre arquitecturas. "
                    "La confianza media no mide exactitud. "
                    "YOLO26 y Mask R-CNN muestran máscaras y cajas; YOLOv8 y Faster R-CNN muestran cajas.")
        execution = run.click(run_comparison, inputs=[source, confidence, model_names, *path_inputs],
                              outputs=[gallery, table, run_status, status], concurrency_limit=1)
        for control in (source, confidence, model_names, *path_inputs):
            control.change(inputs_changed, inputs=[source, model_names],
                           outputs=[gallery, table, run_status, run], cancels=[execution], queue=False)
        refresh.click(clear_results, outputs=[gallery, table, run_status], cancels=[execution], queue=False)
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", max_file_size="20mb")
