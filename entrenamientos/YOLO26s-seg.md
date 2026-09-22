# Entrenamientos — YOLO26s-seg

Registro de corridas del brazo de **segmentación** con *YOLO26s-seg* (*Ultralytics*).
La comparación principal es contra otro segmentador (*Mask R-CNN*). Las métricas de caja se registran solo como puente hacia los brazos AABB.
Fuente de verdad de cada experimento: notebook *notebooks/Test YOLO26s-seg.ipynb*.

---

## Experimento 1

| Campo | Valor |
| :--- | :--- |
| Estado | Completado (val, test, subconjuntos + ejemplos visuales) |
| Notebook | *notebooks/Test YOLO26s-seg.ipynb* |
| Entorno | Google Colab, GPU Tesla T4, *CUDA* (*Ultralytics* 8.4.160) |
| Dataset | *Book Spine 2* v4 en *dataset/* (export YOLOv8; labels con polígonos) |
| Checkpoint | *results/checkpoints/yolo26s_seg/mejor_map50.pt* (época 20) |

### Arquitectura

- Familia: **YOLO26** (segmentación de instancias)
- Variante: **YOLO26s-seg**
- Pesos iniciales: *yolo26s-seg.pt* (COCO, segmentación)
- Tarea: una clase (*lomo*, *single_cls*)
- Salida: cajas AABB + máscaras
- Entrada: *imgsz* = 640
- Parámetros del modelo (fused, resumen Ultralytics): ~10,4 M; ~34,3 GFLOPs

### Hiperparámetros

| Parámetro | Valor |
| :--- | ---: |
| *epochs* | 20 |
| *batch* | 16 |
| *optimizer* | SGD |
| *learning_rate* (*lr0*) | 0,01 |
| *learning_rate_final* (*lrf*) | 0,01 |
| *momentum* | 0,937 |
| *weight_decay* | 0,0005 |
| *patience* | 20 |
| *seed* | 42 |
| *iou* (NMS) | 0,40 |
| *max_det* | 300 |
| *conf_eval* | 0,50 |
| *iou_eval* | 0,50 |

Criterio de checkpoint: mayor *mAP@0.5* de **máscara** en el historial de validación (si no hay máscara, cae a caja).

### Épocas

- Entrenadas: **20**
- Checkpoint elegido: **época 20**
- Observación: la *loss* y el *mAP@0.5:0.95* (caja y máscara) siguen mejorando al final; **conviene probar más épocas** en un experimento posterior (no corrido aún).

### Resultados

Protocolo de informe: *conf* ≥ 0,5 e *IoU* ≥ 0,5. Latencia con batch=1.

#### Validación (reevaluación del mejor checkpoint; prioridad máscara)

| Métrica | Valor |
| :--- | ---: |
| *mAP@0.5* (máscara) | 0,761 |
| *mAP@0.5:0.95* (máscara) | 0,542 |
| Latencia (ms/imagen) | ~18 |
| FPS | ~55 |
| *mAP@0.5* (caja, puente) | 0,804 |
| *mAP@0.5:0.95* (caja, puente) | 0,661 |
| Precisión (caja, puente) | 0,990 |
| Recall (caja, puente) | 0,807 |

Referencia del historial Ultralytics en la época 20 (umbrales internos de val durante train, no el protocolo fijo): *mAP@0.5* caja ≈ 0,842; máscara ≈ 0,792.

#### Test (prioridad: máscara; caja = puente)

| Métrica | Validación | Test |
| :--- | ---: | ---: |
| *mAP@0.5* (máscara) | 0,761 | 0,757 |
| *mAP@0.5:0.95* (máscara) | 0,542 | 0,540 |
| *mAP@0.5* (caja, puente) | 0,804 | 0,814 |
| *mAP@0.5:0.95* (caja, puente) | 0,661 | 0,663 |
| Precisión (caja, puente) | 0,990 | 0,991 |
| Recall (caja, puente) | 0,807 | 0,816 |
| Latencia (ms/imagen) | ~18 | ~18 |

#### Subconjuntos (validación; métricas de **máscara**)

Definición geométrica de denso/inclinado igual que AABB; *mAP*/P/R por IoU de máscara ≥ 0,5.

| Subconjunto | Fotos | GT | *mAP@0.5* | *mAP@0.5:0.95* | Precisión | Recall |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Todas | 271 | 4.622 | 0,725 | 0,535 | 0,961 | 0,733 |
| Denso | 121 | 2.296 | 0,572 | 0,401 | 0,927 | 0,583 |
| Inclinado (> 45°) | 72 | 809 | 0,616 | 0,421 | 0,958 | 0,623 |

Figuras ejemplo (solo máscaras): *results/figures/yolo26s_seg/subconjuntos/*.

### Notas

- Brecha estable caja vs. máscara: el *mAP@0.5* de máscara queda ~0,04–0,05 por debajo del de caja; a *mAP@0.5:0.95* la diferencia crece (~0,12).
- Perfil similar a los brazos AABB: precisión alta, recall más limitado.
- Artefactos: *results/tables/yolo26s_seg_*.*, *results/figures/yolo26s_seg/*.
- Próximo experimento sugerido: subir *epochs* (p. ej. 40–50) manteniendo el resto de hiperparámetros, para ver si el *mAP* de máscara sigue subiendo sin degradar el de caja.
