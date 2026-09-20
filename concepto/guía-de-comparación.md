# Guía: brazo YOLO comparable a Faster R-CNN

Objetivo: entrenar y evaluar **YOLOv8** (brazo A) con el **mismo protocolo** que el notebook *Test Faster R-CNN.ipynb* (brazo B), para poder llenar la tabla A vs. B sin trampas.

Referencia de lo ya hecho: `notebooks/Test Faster R-CNN.ipynb`.  
Dataset: *Book Spine 2* v4 en `dataset/` (export YOLOv8). No re-descargar ni re-partir.

---

## Camino rápido

1. Notebook nuevo (p. ej. `notebooks/Test YOLOv8.ipynb`), mismo estilo: markdown impersonal + código, *cursiva* en nombres.
2. Setup → dataset (skip si ya está) → **no** hace falta conversión COCO.
3. Entrenar YOLOv8s (`imgsz` ≥ 640, Colab T4).
4. Evaluar en **validación**: *mAP@0.5*, *mAP@0.5:0.95*, precisión, recall, latencia.
5. Figuras cualitativas (GT vs pred).
6. Chequeo único en **test** (sin tunear).
7. Subconjuntos **denso** e **inclinado** con las **mismas** definiciones que Faster R-CNN.
8. Completar la tabla del final de este documento.

Listo cuando: hay pesos, métricas en val/test/subconjuntos, figuras y la tabla A vs. B armable fila a fila.

---

## Qué tiene que ser igual (comparación justa)

| Tema | Valor a respetar |
| :--- | :--- |
| Splits | Los de Roboflow en `dataset/` (train / valid / test) |
| Clase | Una sola: *lomo* |
| Cajas | AABB (no OBB, no seg como brazo del paper) |
| Entrada | `imgsz` ≥ **640** |
| Selección de checkpoint | Mejor ***mAP@0.5* en validación** |
| Precisión / recall | *score* ≥ **0,5** e *IoU* ≥ **0,5** (igual que FRCNN) |
| Hardware de latencia | El mismo que se declare (p. ej. Colab T4), batch=1 |
| Test | Solo chequeo final; **no** elegir hiperparámetros mirando test |
| Denso | Imagen con algún GT cuyo *max IoU* con otra caja ≥ **0,5** |
| Inclinado | Solo GT con inclinación del polígono > **45°**; preds que ya explican un GT vertical **no** son FP |

Si YOLO reporta *mAP* con otros umbrales por defecto, **recalcular** P/R (y, si hace falta, *mAP*) con 0,5/0,5 para alinear con el brazo B.

---

## Qué cambia respecto de Faster R-CNN

| Faster R-CNN (ya hecho) | YOLO (a hacer) |
| :--- | :--- |
| Export YOLO → conversión COCO | Usa labels YOLO nativos (`data.yaml`) |
| *torchvision* + DataLoader propio | Ultralytics (`YOLO(...).train` / `.val`) |
| Ánclas / RPN / NMS en código | NMS / `iou` / `conf` / `max_det` en args de Ultralytics |
| Checkpoint `.pt` propio | `best.pt` / `last.pt` del run Ultralytics |
| Salidas en `results/.../faster_rcnn/` | Paralelizar: `results/.../yolov8/` (o similar) |

No copiar la conversión COCO. Sí copiar el **orden de secciones** y las **lecturas** (train, test, subconjuntos).

---

## Estructura sugerida del notebook

### 1. Setup
Instalar *ultralytics*, *torch*, etc. Anotar versiones.

### 2. Dataset
Misma lógica que FRCNN: si `dataset/` ya tiene *data.yaml* e imágenes, no descargar. API key solo por entorno / Colab secret.

### 3. Modelo e hiperparámetros
Candidato del Plan: **YOLOv8s**. Smoke: *yolov8n* pocas épocas.

Documentar al menos:

- `model` (p. ej. `yolov8s.pt`)
- `imgsz` (≥ 640)
- `epochs`, `batch`, `lr0` / optimizer
- `iou` (NMS; conviene **más permisivo que 0,5**, p. ej. ~0,4, alineado al EDA)
- `conf` (umbral al inferir; para P/R del paper usar **0,5** o reportar ambos si Ultralytics usa otro)
- `max_det` (≥ 100; hay fotos con muchos lomos)

### 4. Entrenamiento + métricas en validación
- Guardar historial (CSV) y curvas (*loss*, *mAP*, P/R si están).
- Checkpoint = mejor *mAP@0.5* en val.
- Latencia: ms/imagen y FPS, batch=1, mismo dispositivo que FRCNN.
- Celda markdown de **Lectura** (simple): mejor época, si P ≫ R, efecto del schedule, límites.

### 5. Visualización cualitativa
Misma idea: GT vs predicción, casos aleatorios + peores (más FN / más FP), semilla fija. Guardar figuras bajo `results/figures/yolov8/` (o el prefijo que usen).

### 6. Chequeo en test
Una pasada con el mejor checkpoint. Tabla val vs test. Lectura corta. Sin retocar HP.

### 7. Subconjuntos denso / inclinado
Reutilizar la geometría del EDA (labels YOLO de *valid*):

1. Calcular por caja: *max IoU* con vecinos e inclinación del polígono.
2. **Denso:** subset de imágenes → métricas completas.
3. **Inclinado:** solo GT > 45°; no contar como FP las preds que matchean un GT no inclinado.
4. Tabla global vs denso vs inclinado + Lectura en lenguaje simple.

Ideal: **mismo script/criterio** que en FRCNN (copiar helpers, cambiar solo el forward del modelo).

---

## Salidas mínimas (archivos)

| Artefacto | Destino sugerido |
| :--- | :--- |
| Mejor peso | `results/checkpoints/yolov8/` |
| Historial / resúmenes JSON-CSV | `results/tables/` (`yolov8_*.csv/json`) |
| Curvas y cualitativas | `results/figures/yolov8/` |
| Subconjuntos | `results/tables/yolov8_subconjuntos_validacion.csv` |

Nombres en español donde se pueda; no commitear pesos ni imágenes.

---

## Checklist antes de comparar

- [ ] Mismo `dataset/` y mismos splits que FRCNN  
- [ ] `imgsz` ≥ 640  
- [ ] Mejor checkpoint elegido por *mAP@0.5* en **validación**  
- [ ] P/R con *score*≥0,5 e *IoU*≥0,5 (o documentar equivalencia Ultralytics)  
- [ ] Latencia en el **mismo** hardware declarado  
- [ ] Test corrido una vez, sin tunear  
- [ ] Denso / inclinado con umbrales 0,5 / 45° idénticos  
- [ ] Figuras cualitativas guardadas  
- [ ] Lecturas en el notebook (train, test, subconjuntos)  

---

## Fuera de alcance de esta guía

- OCR / matching (F2–F3)
- YOLOv8-seg u OBB como tercer brazo del paper
- Retocar Faster R-CNN para “empatar” a YOLO
- Validación limpia por leakage (opcional; si se hace, hacerlo en **ambos** brazos)

---

## Tabla de métricas para la comparativa A vs. B

Completar con los números de cada notebook. Todas las filas de detección son sobre **validación**, salvo el bloque *Test*. Latencia en el hardware declarado (batch=1).

| Bloque | Métrica | YOLOv8 (A) | Faster R-CNN (B) | Notas |
| :--- | :--- | ---: | ---: | :--- |
| **Validación (todas)** | *mAP@0.5* | | 0,800 | Criterio de checkpoint |
| | *mAP@0.5:0.95* | | 0,602 | |
| | Precisión (*score*≥0,5, *IoU*≥0,5) | | 0,961 | |
| | Recall (*score*≥0,5, *IoU*≥0,5) | | 0,792 | |
| | Latencia (ms/imagen) | | ~96 | Mismo GPU |
| | FPS (batch=1) | | ~10,4 | |
| | Época del mejor ckpt | | 6 | |
| **Test (chequeo)** | *mAP@0.5* | | 0,820 | No tunear con esto |
| | *mAP@0.5:0.95* | | 0,616 | |
| | Precisión | | 0,952 | |
| | Recall | | 0,814 | |
| | Latencia (ms/imagen) | | ~101 | |
| **Denso (val)** | *mAP@0.5* | | 0,679 | Imagen con GT *max IoU*≥0,5 |
| | *mAP@0.5:0.95* | | 0,492 | |
| | Precisión | | 0,939 | |
| | Recall | | 0,661 | |
| | Imágenes / GT | | 121 / 2.296 | |
| **Inclinado (val)** | *mAP@0.5* | | 0,650 | Solo GT ángulo>45° |
| | *mAP@0.5:0.95* | | 0,478 | |
| | Precisión | | 0,908 | FP: no contar match a GT vertical |
| | Recall | | 0,656 | |
| | Imágenes / GT | | 72 / 809 | |
| **Extras de informe** | Tiempo de entrenamiento | | | Mismo tipo de máquina si se puede |
| | Hiperparámetros clave | | | `imgsz`, NMS/iou, epochs, batch |
| | Figuras cualitativas | sí/no | sí | Val, mismos criterios de selección |

**Criterio sugerido para el modelo demo (pipeline OCR):** mejor balance ***mAP@0.5* en validación / latencia**, no solo el *mAP* más alto. Dejar el criterio escrito al elegir A o B.
