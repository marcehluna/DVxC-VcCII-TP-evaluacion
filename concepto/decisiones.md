# Decisiones de diseño

Registro de elecciones que hay que justificar en el informe (metodología / datos). El alcance está en [`definicion.md`](definicion.md); el plan operativo, en [`Plan.md`](Plan.md).

Cada entrada: qué se decidió, por qué, qué se descartó y cómo se dice en el paper.

---

## D1 — Una sola copia del dataset (export YOLO → COCO)

**Fecha:** 2026-09-19  
**Estado:** vigente  
**Afecta:** datos, comparación YOLOv8 vs. Faster R-CNN

### Decisión

*Book Spine 2* se congela **una vez**, en formato **YOLOv8**, en `dataset/` (versión 4 de Roboflow). Faster R-CNN no descarga un export COCO aparte: usa **las mismas** imágenes y splits, con anotaciones convertidas a cajas COCO bajo `dataset/coco/`.

El AABB de cada lomo es el envolvente del polígono YOLO. No se re-parte ni se vuelve a bajar para el brazo two-stage.

### Por qué

El núcleo del trabajo es una comparación **justa**: mismo conjunto de train/valid/test, mismas fotos. Roboflow permite exportar COCO, pero un segundo download es otra copia (otra fecha, otra versión, otro zip). No hay garantía de que coincida con el freeze YOLO ya usado en el EDA.

La conversión polígono → caja es determinista y barata. El riesgo de dos formatos independientes no lo es.

### Alternativa descartada

Bajar COCO directo desde Roboflow para Faster R-CNN (más simple en esa notebook). Se rechaza porque:

- duplica imágenes y puede desfasar la versión respecto de `dataset/`;
- obliga a auditar a mano que COCO y YOLO sean el mismo v4 y el mismo corte;
- `dataset/` no puede ser a la vez layout YOLO y layout COCO.

### Cómo decirlo en el informe

Se utiliza el export YOLOv8 de *Book Spine 2* v4 como fuente única. El detector two-stage se entrena sobre cajas COCO derivadas de esos polígonos (AABB), de modo que ambos modelos evalúan la misma partición. No se emplean dos descargas Roboflow con formatos distintos.
