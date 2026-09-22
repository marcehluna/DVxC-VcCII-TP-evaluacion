# Entrenamientos — YOLO8-AABB

Registro de corridas del brazo de detección AABB con *YOLOv8s* (*Ultralytics*).
Fuente de verdad de cada experimento: notebook *notebooks/Test YOLO8-AABB.ipynb*.

---

## Experimento 1

| Campo | Valor |
| :--- | :--- |
| Estado | Completado (val, test, subconjuntos) |
| Notebook | *notebooks/Test YOLO8-AABB.ipynb* |
| Entorno | Google Colab, GPU Tesla T4, *CUDA* |
| Dataset | *Book Spine 2* v4 en *dataset/* (export YOLOv8) |
| Checkpoint | *results/checkpoints/yolov8/mejor_map50.pt* (época 19) |

### Arquitectura

- Familia: **YOLOv8** (detección, cajas AABB; no OBB ni segmentación)
- Variante: **YOLOv8s**
- Pesos iniciales: *yolov8s.pt* (COCO)
- Tarea: una clase (*lomo*, *single_cls*)
- Entrada: *imgsz* = 640

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

Criterio de checkpoint: mayor *mAP@0.5* (caja) en el historial de validación.

### Épocas

- Entrenadas: **20**
- Checkpoint elegido: **época 19**

### Resultados

Protocolo de informe: *conf* ≥ 0,5 e *IoU* ≥ 0,5. Latencia con batch=1.

#### Validación (reevaluación del mejor checkpoint)

| Métrica | Valor |
| :--- | ---: |
| *mAP@0.5* | 0,822 |
| *mAP@0.5:0.95* | 0,676 |
| Precisión | 0,976 |
| Recall | 0,823 |
| Latencia (ms/imagen) | ~14 |
| FPS | ~74 |

#### Test (pasada única, sin tunear)

| Métrica | Validación | Test |
| :--- | ---: | ---: |
| *mAP@0.5* | 0,822 | 0,822 |
| *mAP@0.5:0.95* | 0,676 | 0,665 |
| Precisión | 0,976 | 0,978 |
| Recall | 0,823 | 0,825 |
| Latencia (ms/imagen) | ~14 | ~14 |

#### Subconjuntos (validación, matching *torchmetrics* + P/R guloso)

| Subconjunto | Fotos | GT | *mAP@0.5* | Precisión | Recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Todas | 271 | 4.622 | 0,76 | 0,97 | 0,76 |
| Denso (imagen, max IoU GT ≥ 0,5) | 121 | 2.296 | 0,62 | 0,96 | 0,63 |
| Inclinado (GT > 45°) | 72 | 809 | 0,64 | 0,94 | 0,66 |

### Notas

- Perfil: precisión alta, recall más bajo; el cuello de botella es dejar lomos sin detectar, sobre todo en densos e inclinados.
- Artefactos: *results/tables/yolov8_*.*, *results/figures/yolov8/*.
