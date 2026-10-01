"""Inferencia de los checkpoints documentados para la demo local."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Sequence
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    default_path: Path
    env_var: str
    color: tuple[int, int, int]
    fallback_path: Path | None = None

    @property
    def path(self) -> Path:
        override = os.getenv(self.env_var)
        if override:
            return Path(override).expanduser().resolve()
        if self.default_path.is_file() or self.fallback_path is None:
            return self.default_path
        return self.fallback_path if self.fallback_path.is_file() else self.default_path

    def resolve_path(self, override: str | None = None) -> Path:
        if not override or not override.strip():
            return self.path
        path = Path(override.strip().strip('"')).expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()


MODELS = (
    ModelSpec("YOLOv8 AABB", "yolo", ROOT / "results/checkpoints/yolov8/mejor_map50.pt", "DEMO_YOLOV8_WEIGHTS", (28, 160, 204)),
    ModelSpec("Faster R-CNN", "faster_rcnn", ROOT / "results/checkpoints/faster_rcnn/mejor_map50.pt", "DEMO_FASTER_RCNN_WEIGHTS", (236, 129, 44)),
    ModelSpec("YOLO26 segmentación", "yolo_seg", ROOT / "results/checkpoints/yolo26s_seg/mejor_map50.pt", "DEMO_YOLO26_SEG_WEIGHTS", (109, 179, 95)),
    ModelSpec(
        "Mask R-CNN",
        "mask_rcnn",
        ROOT / "results/checkpoints/mask_rcnn/mejor_mask_map50.pt",
        "DEMO_MASK_RCNN_WEIGHTS",
        (170, 91, 191),
        ROOT / "results/checkpoints/faster_rcnn/mejor_mask_map50.pt",
    ),
)


@dataclass
class Prediction:
    spec: ModelSpec
    image: Image.Image | None
    count: int = 0
    mean_score: float | None = None
    elapsed_ms: float | None = None
    message: str = ""


def selected_paths(overrides: Sequence[str] | None = None) -> list[Path]:
    if overrides is not None and len(overrides) != len(MODELS):
        raise ValueError("Se necesita una ruta por modelo.")
    return [spec.resolve_path(overrides[index] if overrides is not None else None)
            for index, spec in enumerate(MODELS)]


def available_models(overrides: Sequence[str] | None = None) -> dict[str, bool]:
    return {spec.name: path.is_file() for spec, path in zip(MODELS, selected_paths(overrides))}


def device_name() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=6)
def _load_model(kind: str, path: str, mtime_ns: int, device: str):
    # El timestamp en la clave permite recargar un checkpoint reemplazado.
    if kind.startswith("yolo"):
        from ultralytics import YOLO

        return YOLO(path)

    import torch
    if kind == "mask_rcnn":
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        # Reproducir la arquitectura usada al entrenar; los pesos vienen del checkpoint.
        model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            min_size=640,
            max_size=640,
            box_detections_per_img=100,
            box_nms_thresh=0.7,
        )
        box_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
        mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, 2)

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["modelo"] if isinstance(checkpoint, dict) and "modelo" in checkpoint else checkpoint
        model.load_state_dict(state)
        return model.to(device).eval()

    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.rpn import RPNHead

    # Se evita descargar pesos COCO: el checkpoint local ya contiene el modelo.
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.transform.min_size = (640,)
    model.transform.max_size = 1333
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.40
    model.roi_heads.detections_per_img = 100
    model.rpn.nms_thresh = 0.70
    sizes = ((32,), (64,), (128,), (256,), (512,))
    ratios = (0.125, 0.25, 0.5, 1.0, 2.0)
    model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(ratios,) * 5)
    model.rpn.head = RPNHead(model.backbone.out_channels, len(ratios))

    # Los checkpoints de este repo guardan un dict con la clave "modelo".
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["modelo"] if isinstance(checkpoint, dict) and "modelo" in checkpoint else checkpoint
    model.load_state_dict(state)
    return model.to(device).eval()


def _draw(image: Image.Image, boxes: list[list[float]], scores: list[float],
          color: tuple[int, int, int], polygons: list[list[list[float]]] | None = None,
          masks: np.ndarray | None = None) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size)
    painter = ImageDraw.Draw(overlay)
    if masks is not None:
        for mask in masks:
            overlay.paste((*color, 65), (0, 0), Image.fromarray(mask.astype(np.uint8) * 255))
    if polygons:
        for polygon in polygons:
            if len(polygon) >= 3:
                painter.polygon([tuple(point) for point in polygon], fill=(*color, 65))
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)
    width = max(2, round(min(canvas.size) / 320))
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=width)
        label = f"{score:.2f}"
        label_y = max(0, y1 - 15)
        draw.rectangle((x1, label_y, x1 + 34, label_y + 14), fill=(*color, 255))
        draw.text((x1 + 2, label_y), label, fill="white")
    return canvas.convert("RGB")


def predict_one(image: Image.Image, spec: ModelSpec, confidence: float, device: str,
                path_override: str | None = None) -> Prediction:
    path = spec.resolve_path(path_override)
    if not path.is_file():
        return Prediction(spec, None, message=f"Falta el checkpoint: {path}")

    try:
        model = _load_model(spec.kind, str(path), path.stat().st_mtime_ns, device)
        masks = None
        if spec.kind.startswith("yolo"):
            import torch

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = perf_counter()
            result = model.predict(source=image, imgsz=640, conf=confidence, iou=0.40,
                                   max_det=300, device=device, verbose=False)[0]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (perf_counter() - start) * 1000
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            polygons = result.masks.xy if spec.kind == "yolo_seg" and result.masks is not None else None
        else:
            import torch
            from torchvision.transforms.functional import to_tensor

            tensor = to_tensor(image).to(device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = perf_counter()
            with torch.inference_mode():
                result = model([tensor])[0]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (perf_counter() - start) * 1000
            keep = (result["scores"] >= confidence) & (result["labels"] == 1)
            boxes = result["boxes"][keep].cpu().tolist()
            scores = result["scores"][keep].cpu().tolist()
            polygons = None
            if spec.kind == "mask_rcnn":
                masks = (result["masks"][keep, 0] >= 0.5).cpu().numpy()

        rendered = _draw(image, boxes, scores, spec.color, polygons, masks)
        return Prediction(spec, rendered, len(scores), sum(scores) / len(scores) if scores else None,
                          elapsed_ms, "OK")
    except Exception as exc:
        return Prediction(spec, None, message=f"Error al ejecutar {spec.name}: {type(exc).__name__}: {exc}")


def iter_predictions(image: Image.Image, confidence: float,
                     path_overrides: Sequence[str] | None = None,
                     model_names: Sequence[str] | None = None):
    """Ejecuta únicamente los modelos elegidos, en el orden del registro."""
    if image is None:
        raise ValueError("Subí una imagen para comparar los modelos.")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("La confianza debe estar entre 0 y 1.")
    names = [spec.name for spec in MODELS] if model_names is None else list(model_names)
    if not names:
        raise ValueError("Elegí al menos un modelo.")
    if set(names) - {spec.name for spec in MODELS}:
        raise ValueError("La selección contiene un modelo desconocido.")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    paths = selected_paths(path_overrides)
    selected = [(spec, path) for spec, path in zip(MODELS, paths) if spec.name in names]
    device = device_name() if any(path.is_file() for _, path in selected) else "cpu"
    for spec, path in selected:
        yield predict_one(rgb, spec, confidence, device, str(path))


def predict_all(image: Image.Image, confidence: float,
                path_overrides: Sequence[str] | None = None,
                model_names: Sequence[str] | None = None) -> list[Prediction]:
    return list(iter_predictions(image, confidence, path_overrides, model_names))
