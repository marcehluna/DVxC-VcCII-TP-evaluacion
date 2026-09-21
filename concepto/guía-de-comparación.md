# Guía: brazo YOLO comparable a Faster R-CNN

Objetivo: entrenar y evaluar **YOLOv8** (brazo A) con el **mismo protocolo** que el notebook *Test Faster R-CNN.ipynb* (brazo B), para poder llenar la tabla A vs. B sin trampas.

**Estado:** ambos brazos corridos. Notebooks de referencia:

- Brazo A: *notebooks/Test YOLO8-AABB.ipynb* (*YOLOv8s*, AABB)
- Brazo B: *notebooks/Test Faster R-CNN.ipynb*

Dataset: *Book Spine 2* v4 en *dataset/* (export YOLOv8). No re-descargar ni re-partir.

---

## Camino rápido

1. Notebook *Test YOLO8-AABB.ipynb*: markdown impersonal + código, *cursiva* en nombres.
2. Setup → dataset (skip si ya está) → **no** hace falta conversión COCO.
3. Entrenar *YOLOv8s* (`imgsz` ≥ 640, Colab T4).
4. Evaluar en **validación**: *[mAP@0.5](mailto:mAP@0.5)*, *[mAP@0.5](mailto:mAP@0.5):0.95*, precisión, recall, latencia.
5. Figuras cualitativas (GT vs pred).
6. Chequeo único en **test** (sin tunear).
7. Subconjuntos **denso** e **inclinado** con las **mismas** definiciones que Faster R-CNN.
8. Completar la tabla del final de este documento.

Listo cuando: hay pesos, métricas en val/test/subconjuntos, figuras y la tabla A vs. B armable fila a fila. *(Cumplido en la corrida documentada abajo.)*

---

## Qué tiene que ser igual (comparación justa)


| Tema                    | Valor a respetar                                                                                   |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| Splits                  | Los de Roboflow en *dataset/* (train / valid / test)                                               |
| Clase                   | Una sola: *lomo*                                                                                   |
| Cajas                   | AABB (no OBB, no seg como brazo del paper)                                                         |
| Entrada                 | `imgsz` ≥ **640**                                                                                  |
| Selección de checkpoint | Mejor ***[mAP@0.5](mailto:mAP@0.5)* en validación**                                                |
| Precisión / recall      | *score* / *conf* ≥ **0,5** e *IoU* ≥ **0,5** (igual que FRCNN)                                     |
| Hardware de latencia    | El mismo que se declare (p. ej. Colab T4), batch=1                                                 |
| Test                    | Solo chequeo final; **no** elegir hiperparámetros mirando test                                     |
| Denso                   | Imagen con algún GT cuyo *max IoU* con otra caja ≥ **0,5**                                         |
| Inclinado               | Solo GT con inclinación del polígono > **45°**; preds que ya explican un GT vertical **no** son FP |


Si YOLO reporta *mAP* con otros umbrales por defecto, **recalcular** P/R (y, si hace falta, *mAP*) con 0,5/0,5 para alinear con el brazo B.

---

## Qué cambia respecto de Faster R-CNN


| Faster R-CNN                          | YOLO                                                              |
| ------------------------------------- | ----------------------------------------------------------------- |
| Export YOLO → conversión COCO         | Usa labels YOLO nativos (*data.yaml*)                             |
| *torchvision* + DataLoader propio     | Ultralytics (`YOLO(...).train` / `.val`)                          |
| Ánclas / RPN / NMS en código          | NMS / `iou` / `conf` / `max_det` en args de Ultralytics           |
| Checkpoint `.pt` propio               | `best.pt` → copiado a *results/checkpoints/yolov8/mejor_map50.pt* |
| Salidas en *results/.../faster_rcnn/* | Paralelizar: *results/.../yolov8/*                                |


No copiar la conversión COCO. Sí copiar el **orden de secciones** y las **lecturas** (train, test, subconjuntos).

---

## Estructura del notebook (brazo A)

### 1. Setup

Instalar *ultralytics*, *torch*, etc. Anotar versiones.

### 2. Dataset

Misma lógica que FRCNN: si *dataset/* ya tiene *data.yaml* e imágenes, no descargar. API key solo por entorno / Colab secret.

### 3. Modelo e hiperparámetros

Candidato del Plan: **YOLOv8s**.

Corrida documentada:


| Parámetro                     | Valor               |
| ----------------------------- | ------------------- |
| `model`                       | *yolov8s.pt* (COCO) |
| `imgsz`                       | 640                 |
| `epochs`                      | 20                  |
| `batch`                       | 16                  |
| `optimizer`                   | SGD                 |
| `learning_rate` (`lr0`)       | 0,01                |
| `learning_rate_final` (`lrf`) | 0,01                |
| `iou` (NMS)                   | 0,40                |
| `conf` / *SCORE_EVAL* (P/R)   | 0,50                |
| `max_det`                     | 300                 |
| Hardware                      | Colab Tesla T4      |


### 4. Entrenamiento + métricas en validación

- Historial CSV, curvas, resumen JSON.
- Checkpoint = mejor *[mAP@0.5](mailto:mAP@0.5)* en val → **época 19**.
- Latencia batch=1 en el mismo tipo de GPU que FRCNN.
- Celda markdown de **Lectura**.

### 5. Visualización cualitativa

GT vs predicción; aleatorias + más FN + más FP; semilla fija. Figuras en *results/figures/yolov8/cualitativo/*.

### 6. Chequeo en test

Una pasada con el mejor checkpoint. Tabla val vs test. Lectura corta. Sin retocar HP.

### 7. Subconjuntos denso / inclinado

Misma geometría que el EDA / FRCNN (labels YOLO de *valid*):

1. Por caja: *max IoU* con vecinos e inclinación del polígono.
2. **Denso:** subset de imágenes → métricas completas.
3. **Inclinado:** solo GT > 45°; no contar como FP las preds que matchean un GT no inclinado.
4. Tabla global vs denso vs inclinado + Lectura.

---

## Salidas mínimas (archivos)


| Artefacto                      | Destino sugerido                                    |
| ------------------------------ | --------------------------------------------------- |
| Mejor peso                     | *results/checkpoints/yolov8/*                       |
| Historial / resúmenes JSON-CSV | *results/tables/* (`yolov8_*.csv/json`)             |
| Curvas y cualitativas          | *results/figures/yolov8/*                           |
| Subconjuntos                   | *results/tables/yolov8_subconjuntos_validacion.csv* |


Nombres en español donde se pueda; no commitear pesos ni imágenes.

---

## Checklist antes de comparar

- [x] Mismo *dataset/* y mismos splits que FRCNN
- [x] `imgsz` ≥ 640
- [x] Mejor checkpoint elegido por *[mAP@0.5](mailto:mAP@0.5)* en **validación** (época 19)
- [x] P/R con *conf*≥0,5 e *IoU*≥0,5
- [x] Latencia en el **mismo** hardware declarado (Colab T4)
- [x] Test corrido una vez, sin tunear
- [x] Denso / inclinado con umbrales 0,5 / 45° idénticos
- [x] Figuras cualitativas guardadas
- [x] Lecturas en el notebook (train, test, subconjuntos)
- [x] Tabla A vs. B completada

---

## Fuera de alcance de esta guía

- OCR / matching (F2–F3)
- YOLOv8-seg u OBB como tercer brazo del paper
- Retocar Faster R-CNN para “empatar” a YOLO
- Validación limpia por leakage (opcional; si se hace, hacerlo en **ambos** brazos)

---

## Tabla de métricas para la comparativa A vs. B

Números de *Test YOLO8-AABB.ipynb* (A) y *Test Faster R-CNN.ipynb* (B). Detección en **validación**, salvo el bloque *Test*. Latencia en Colab T4, batch=1.


| Bloque                 | Métrica                                   | YOLOv8 (A)                                            | Faster R-CNN (B)                    | Notas                              |
| ---------------------- | ----------------------------------------- | ----------------------------------------------------- | ----------------------------------- | ---------------------------------- |
| **Validación (todas)** | *[mAP@0.5](mailto:mAP@0.5)*               | 0,822                                                 | 0,800                               | Criterio de checkpoint             |
|                        | *[mAP@0.5](mailto:mAP@0.5):0.95*          | 0,676                                                 | 0,602                               |                                    |
|                        | Precisión (*conf*/*score*≥0,5, *IoU*≥0,5) | 0,976                                                 | 0,961                               |                                    |
|                        | Recall (*conf*/*score*≥0,5, *IoU*≥0,5)    | 0,823                                                 | 0,792                               |                                    |
|                        | Latencia (ms/imagen)                      | ~14                                                   | ~96                                 | Mismo tipo de GPU (T4)             |
|                        | FPS (batch=1)                             | ~74                                                   | ~10,4                               |                                    |
|                        | Época del mejor ckpt                      | 19                                                    | 6                                   |                                    |
| **Test (chequeo)**     | *[mAP@0.5](mailto:mAP@0.5)*               | 0,822                                                 | 0,820                               | No tunear con esto                 |
|                        | *[mAP@0.5](mailto:mAP@0.5):0.95*          | 0,665                                                 | 0,616                               |                                    |
|                        | Precisión                                 | 0,978                                                 | 0,952                               |                                    |
|                        | Recall                                    | 0,825                                                 | 0,814                               |                                    |
|                        | Latencia (ms/imagen)                      | ~14                                                   | ~101                                |                                    |
| **Denso (val)**        | *[mAP@0.5](mailto:mAP@0.5)*               | 0,620                                                 | 0,679                               | Imagen con GT *max IoU*≥0,5        |
|                        | *[mAP@0.5](mailto:mAP@0.5):0.95*          | 0,519                                                 | 0,492                               |                                    |
|                        | Precisión                                 | 0,956                                                 | 0,939                               |                                    |
|                        | Recall                                    | 0,629                                                 | 0,661                               |                                    |
|                        | Imágenes / GT                             | 121 / 2.296                                           | 121 / 2.296                         | Mismo criterio geométrico          |
| **Inclinado (val)**    | *[mAP@0.5](mailto:mAP@0.5)*               | 0,645                                                 | 0,650                               | Solo GT ángulo>45°                 |
|                        | *[mAP@0.5](mailto:mAP@0.5):0.95*          | 0,530                                                 | 0,478                               |                                    |
|                        | Precisión                                 | 0,943                                                 | 0,908                               | FP: no contar match a GT vertical  |
|                        | Recall                                    | 0,656                                                 | 0,656                               | Empate en recall                   |
|                        | Imágenes / GT                             | 72 / 809                                              | 72 / 809                            |                                    |
| **Extras de informe**  | Tiempo de entrenamiento                   | ~0,27 h (20 ép.)                                      | (ver notebook B)                    | Colab T4                           |
|                        | Hiperparámetros clave                     | *yolov8s*, imgsz 640, batch 16, lr 0,01, NMS iou 0,40 | ResNet-50 FPN v2, batch 2, lr 0,005 |                                    |
|                        | Figuras cualitativas                      | sí                                                    | sí                                  | Val, mismos criterios de selección |


### Lectura rápida de la comparación

- En el **promedio** (val/test) YOLOv8 gana en *mAP*, precisión, recall y, sobre todo, en **velocidad** (~7× más rápido en esta medición).
- En **densos** Faster R-CNN queda mejor en *[mAP@0.5](mailto:mAP@0.5)* (0,679 vs 0,620) y en recall; en **inclinados** casi empatan (*mAP* 0,645 vs 0,650; recall idéntico 0,656).
- Ambos comparten el mismo perfil: **precisión alta, recall más bajo**, y el recall cae en escenas difíciles.
- **Criterio demo (pipeline OCR):** si el demo prioriza fluidez y métricas globales, el brazo A (YOLOv8) encaja mejor; si prioriza no perder lomos en estantes apretados, el brazo B sigue compitiendo en ese corte. Dejar el criterio escrito al elegir A o B.

