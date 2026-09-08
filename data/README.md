# Datos del TP

No versionar el dataset completo ni pesos grandes.

## Book Spine 2 (Roboflow)

1. Abrir: https://universe.roboflow.com/bookspine-fxfsx/book_spine_2
2. Exportar en formato **YOLOv8** (o YOLO) con los splits del proyecto.
3. Descomprimir en `data/raw/book_spine_2/` (train / valid / test según el export).
4. Generar versión COCO/torchvision en `data/processed/` con `src/convert_annotations.py` (cuando esté implementado).

Documentar aquí la fecha de descarga y cualquier versión/split congelado (F0 del Plan).
