# Datos del TP

El destino canónico es **`dataset/`** en la raíz del repo (no `data/raw/`). No versionar imágenes, labels ni pesos.

## Book Spine 2 (Roboflow) — freeze F0

| Campo | Valor |
| :--- | :--- |
| Proyecto | [`bookspine-fxfsx/book_spine_2`](https://universe.roboflow.com/bookspine-fxfsx/book_spine_2) |
| Versión | **4** ([export](https://universe.roboflow.com/bookspine-fxfsx/book_spine_2/dataset/4)) |
| Fecha de descarga | 2026-09-11 |
| Formato | YOLOv8 (polígonos; una clase `1` = lomo) |
| Licencia | CC BY 4.0 |
| Resolución | 640×640 |
| Destino | `dataset/` |

### Splits congelados (no re-partir)

| Partición | Imágenes | Etiquetas | Lomos | % imágenes |
| :--- | ---: | ---: | ---: | ---: |
| Entrenamiento (`train`) | 1.265 | 1.265 | 20.354 | 76,4 % |
| Validación (`valid`) | 271 | 271 | 4.622 | 16,4 % |
| Prueba (`test`) | 120 | 120 | 1.747 | 7,2 % |
| **Total** | **1.656** | **1.656** | **26.723** | 100 % |

La comparación A vs. B se mide en **validación**. Test es chequeo final. Análisis geométrico (ocupación, aspecto, inclinación, solape): `notebooks/Dataset_y_EDA.ipynb` y `concepto/Plan.md` §4.1.

No hay transcripciones de títulos: el GT de OCR se arma a mano en F2.

## Descarga

Descarga idempotente desde `notebooks/Dataset_y_EDA.ipynb`:

1. Crea `dataset/` si no existe.
2. Si ya hay `data.yaml` e imágenes, **no vuelve a descargar**.
3. Si falta, exporta **YOLOv8** desde el proyecto de Universe.
4. Conversiones a COCO (Faster R-CNN) van bajo `dataset/` (p. ej. `dataset/coco/`).

La API key de Roboflow no va en el repo: `ROBOFLOW_API_KEY` por env, `.env` o secret de Colab.
