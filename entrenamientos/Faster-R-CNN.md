# Entrenamientos — Faster R-CNN

Registro de corridas del brazo de detección AABB con *Faster R-CNN* (*torchvision*).
Fuente de verdad de cada experimento: notebook *notebooks/Test Faster R-CNN.ipynb*.

---

## Experimento 1

| Campo | Valor |
| :--- | :--- |
| Estado | Completado (val, test, subconjuntos) |
| Notebook | *notebooks/Test Faster R-CNN.ipynb* |
| Entorno | Google Colab, GPU Tesla T4, *CUDA* |
| Dataset | *Book Spine 2* v4 → etiquetas COCO bajo *dataset/coco/* |
| Checkpoint | *results/checkpoints/faster_rcnn/mejor_map50.pt* (época 6) |

### Arquitectura

- Familia: **Faster R-CNN**
- Backbone / cuello: **ResNet-50 + FPN v2** (*fasterrcnn_resnet50_fpn_v2*)
- Pesos iniciales: COCO (*FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT*)
- Clases: 2 (fondo + *lomo*)
- Cajas: AABB
- Resize interno: *min_size* = 640, *max_size* = 1333
- Ánclas RPN: tamaños (32, 64, 128, 256, 512); aspectos (0,125, 0,25, 0,5, 1,0, 2,0)

### Hiperparámetros

| Parámetro | Valor |
| :--- | ---: |
| *epocas* | 12 |
| *tamano_lote* | 2 |
| *learning_rate* | 0,005 |
| *momentum* | 0,9 |
| *weight_decay* | 0,0005 |
| *learning_rate_step_size* | 8 |
| *learning_rate_gamma* | 0,1 |
| *semilla* | 42 |
| *box_score_thresh* (entrenamiento / modelo) | 0,05 |
| *box_nms_thresh* | 0,40 |
| *rpn_nms_thresh* | 0,70 |
| *box_detections_per_img* | 100 |
| Score de informe (*SCORE_EVAL*) | 0,50 |
| IoU de matching (*IOU_EVAL*) | 0,50 |

Criterio de checkpoint: mayor *mAP@0.5* en validación.

### Épocas

- Entrenadas: **12**
- Checkpoint elegido: **época 6**

### Resultados

Protocolo de informe: *score* ≥ 0,5 e *IoU* ≥ 0,5. Latencia con batch=1.

#### Validación (mejor checkpoint)

| Métrica | Valor |
| :--- | ---: |
| *mAP@0.5* | 0,800 |
| *mAP@0.5:0.95* | 0,606 |
| Precisión | 0,960 |
| Recall | 0,788 |
| Latencia (ms/imagen) | ~88 |
| FPS | ~11 |

#### Test (pasada única, sin tunear)

| Métrica | Validación | Test |
| :--- | ---: | ---: |
| *mAP@0.5* | 0,800 | 0,820 |
| *mAP@0.5:0.95* | 0,602 | 0,616 |
| Precisión | 0,961 | 0,952 |
| Recall | 0,792 | 0,814 |
| Latencia (ms/imagen) | ~96 | ~101 |

#### Subconjuntos (validación)

| Subconjunto | Fotos | GT | *mAP@0.5* | Precisión | Recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Todas | 271 | 4.622 | 0,80 | 0,96 | 0,79 |
| Denso (imagen, max IoU GT ≥ 0,5) | 121 | 2.296 | 0,68 | 0,94 | 0,66 |
| Inclinado (GT > 45°) | 72 | 809 | 0,65 | 0,91 | 0,66 |

### Notas

- Tras la época 6 el *mAP@0.5* se estanca; el *mAP@0.5:0.95* sigue subiendo con el *StepLR*.
- Perfil: precisión alta, recall más bajo; cae en densos e inclinados.
- Artefactos: *results/tables/faster_rcnn_*.*, *results/figures/faster_rcnn/*.
