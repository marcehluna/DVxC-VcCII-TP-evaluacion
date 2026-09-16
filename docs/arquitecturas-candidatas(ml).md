# Arquitecturas candidatas — detección de lomos

Nota interna. El paper compara **un** YOLO vs. **un** Faster R-CNN (AABB, una clase, Colab T4, `imgsz` ≥ 640). Acá están las variantes de cada familia que alcanzan para cerrar ese núcleo, y cuáles no conviene abrir.

El dataset (Book Spine 2 v4) empuja a: lomos altos y angostos (alto/ancho mediano ~7), estantes densos (NMS ~0,5 mata TP), ~18 % muy inclinados, AABB holgado. Hasta ~67 lomos en una foto de validación.

## Recomendación para el experimento

| Brazo | Modelo | Por qué |
| :--- | :--- | :--- |
| A — one-stage | **YOLOv8s** (Ultralytics, detección) | Cabe en T4 a 640, hay paper/docs (`06` Yaseen), latencia competitiva. `n` queda corto si el NMS y los lomos finos importan; `m` es plan B si `s` se queda corto en mAP. |
| B — two-stage | **Faster R-CNN ResNet-50 FPN v2** (torchvision) | Implementación lista, pretrained COCO, FPN para escalas distintas. Es el two-stage “de libro” (`04` Ren) sin armar RPN a mano. |

Smoke / debug (pocas épocas, no es el número del paper): **YOLOv8n** y, si hace falta recortar cómputo, el mismo Faster R-CNN con menos `max_size` **no** — no bajar de 640. Mejor menos épocas.

No son un tercer brazo: YOLOv8-seg (solo para recortar OCR), YOLOv8-obb / Oriented R-CNN (contexto en el paper, `01` Yang).

---

## Familia YOLO (brazo A)

Stack fijo del Plan: **YOLOv8 detección**, paquete Ultralytics, pesos COCO. Todas las de abajo resuelven el TP; cambian el trade-off mAP / VRAM / ms.

### Detección AABB (lo que entra al experimento)

| Variante | Parámetros (aprox.) | Encaje T4 @ 640 | Uso |
| :--- | ---: | :--- | :--- |
| `yolov8n` | 3,2 M | Holgado | Baseline de pocas épocas; verificar mAP y figuras. Débil en lomos finos/densos. |
| **`yolov8s`** | 11 M | Cómodo | **Candidato principal A.** |
| `yolov8m` | 26 M | Justo (batch 8–16) | Si `s` deja recall flojo en estantes densos. Un run extra, no dos papers. |
| `yolov8l` | 44 M | Apretado | Solo si `m` no alcanza y hay cuota. Poco margen para Faster R-CNN el mismo día. |
| `yolov8x` | 68 M | Malo en T4 | Descartar. |

Anchor-free + cabeza desacoplada + FPN/PAN (`06`, `05`). Conviene frente a lomos de aspecto extremo: no depende de ánclas 1:1 / 1:2.

**Hiperparámetros que importan más que pasar de `s` a `l`:** `imgsz=640` (o 800 si entra), NMS IoU más permisivo que 0,5, `max_det` ≥ 100 (hay fotos con 50–67 lomos).

### Variantes YOLO que resuelven el *problema* pero no el *diseño* del TP

| Variante | Qué aporta | Por qué no es el brazo A |
| :--- | :--- | :--- |
| YOLOv8-seg (`n/s/m-seg`) | Máscara → recorte más limpio para OCR (ocupación AABB ~0,54) | Tercer experimento. Útil **después**, sobre el modelo demo, no vs. Faster R-CNN. |
| YOLOv8-obb | Caja orientada; el ~18 % muy torcido deja de inflar el AABB | Cambia la métrica (hay que definir IoU orientado). El brazo B sería Oriented R-CNN, no Faster R-CNN. |
| YOLO11 / YOLOv9 / v10 | El paper de lomos `02` usa YOLOv11 + OCR | El Plan congela **v8** (cita `06`). Cambiar de generación ensucia la comparación y la bibliografía. |
| YOLO-NAS, YOLOX | Otras ramas one-stage (`05`) | Otro código, otro paper. Fuera. |

---

## Familia Faster R-CNN (brazo B)

Stack fijo: **torchvision.models.detection**, una clase (+ background), mismas imágenes/splits que YOLO (cajas COCO derivadas del polígono).

### Detectores listos en torchvision

| Variante | Constructor | Encaje T4 @ ≥ 640 | Uso |
| :--- | :--- | :--- | :--- |
| **ResNet-50 FPN v2** | `fasterrcnn_resnet50_fpn_v2` | Pesado pero viable (batch 2–4) | **Candidato principal B.** Mejor AP COCO que v1; FPN ayuda con lomos de distinto tamaño. |
| ResNet-50 FPN v1 | `fasterrcnn_resnet50_fpn` | Similar / un poco más liviano | Plan B si v2 pincha por API/pesos. Misma historia para el paper. |
| MobileNetV3 Large FPN | `fasterrcnn_mobilenet_v3_large_fpn` | Holgado | Si ResNet-50 no entra en tiempo/cuota. Se discute como two-stage *rápido*, no como “el” Faster R-CNN clásico. |
| MobileNetV3 320 FPN | `fasterrcnn_mobilenet_v3_large_320_fpn` | Entrada ~320 | **No.** El EDA pide no bajar resolución. |

Backbones tipo ResNet-101 no vienen armados en torchvision para Faster R-CNN; armarlos a mano no vale el tiempo del curso.

### Configuración (mismo modelo, distinto comportamiento)

Esto no es otra arquitectura, pero en este dataset mueve más el resultado que cambiar de MobileNet a ResNet:

| Palanca | Default típico | Ajuste por el EDA |
| :--- | :--- | :--- |
| Aspectos de ánclas RPN | 0,5 / 1 / 2 | Lomo mediano alto/ancho ~7 → agregar ratios chicos (p. ej. 1/4, 1/8) o el AABB se propone mal. |
| `box_nms_thresh` / RPN NMS | 0,5 | ~23 % de cajas GT en valid ya tienen IoU ≥ 0,5 con un vecino. Bajar NMS o documentar *soft-NMS*. |
| `box_detections_per_img` | 100 | Máximo observado ~67; 100 alcanza. No bajar a 20–50. |
| `min_size` / `max_size` | 800 / 1333 | No bajar el lado corto de 640. Subir de 800 es opcional y caro en T4. |

### Variantes two-stage que no son el brazo B

| Variante | Relación | Uso en el TP |
| :--- | :--- | :--- |
| Mask R-CNN (`07`) | Faster R-CNN + máscara | Mismo argumento que YOLOv8-seg: recorte OCR, no comparación de detección. |
| Oriented R-CNN (`01`) | Two-stage con caja rotada | Trabajos relacionados: por qué AABB duele en lomos torcidos. No hay receta corta en torchvision. |
| RetinaNet / FCOS (torchvision) | One-stage, no Faster R-CNN | No sustituyen el brazo B. |

---

## Cómo elegir sin abrir un abanico

Orden de corridas (siempre el mismo split v4, misma semilla, T4, batch=1 para latencia):

1. **Smoke:** YOLOv8n, pocas épocas → figuras y que el mAP no sea 0.
2. **Brazo A:** YOLOv8s, protocolo completo.
3. **Brazo B:** Faster R-CNN ResNet-50 FPN v2, mismos splits, `imgsz` comparable.
4. **Solo si A o B fallan de forma evidente:** YOLOv8m **o** ánclas/NMS en Faster R-CNN (un cambio a la vez).
5. **Tabla del paper:** un YOLO + un Faster R-CNN. Lo demás va a apéndice o a “intentos descartados”.

Criterio del modelo demo (pipeline OCR): mejor balance **mAP@0.5 en validación / ms por imagen**, no el mAP más alto a cualquier costo. Medir también el subconjunto de valid **sin** la misma foto fuente en train (leakage del export Roboflow).

---

## Encaje con las referencias

| Paper | Cómo alimenta la elección |
| :--- | :--- |
| `04` Ren 2015 | Define Faster R-CNN (RPN + Fast R-CNN). Justifica el brazo two-stage. |
| `06` Yaseen 2024 + Ultralytics | Define YOLOv8 (C2f, FPN/PAN, n/s/m/l/x). Justifica el brazo one-stage. |
| `05` Terven 2023 | Contexto one-stage vs. two-stage y la línea YOLO. No obliga a v5/v7/NAS. |
| `01` Yang 2024 | Lomos inclinados y AABB flojo → limitaciones del diseño, no un tercer modelo. |
| `02` Li 2025 | Pipeline detección → OCR (ellos: YOLO11 + PaddleOCR). Nosotros: v8 + EasyOCR. |
| `07` He 2017 | Máscaras si se discute el recorte; fuera del núcleo A vs. B. |
| `09` COCO | mAP@0.5 y @0.5:0.95; no interpretar COCO-large como “lomo fácil”. |

---

## Fuera de lista (no cierran el TP como está definido)

- Detectores que no sean YOLO ni Faster R-CNN como *los dos* brazos (EfficientDet, DETR, etc.).
- Entrenar from scratch sin COCO: más épocas de las que hay en Colab.
- Una escala `x` / ResNet-101 “porque es más grande”.
- Cambiar YOLOv8 → YOLO11 a mitad de F1.
