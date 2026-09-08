# Resumen — A Comprehensive Review of YOLO Architectures (v1 → v8 and YOLO-NAS)

**Archivo:** `05_Terven2023_YOLO_Architectures_Review.pdf`  
**Cita:** Terven, J.; Córdova-Esparza, D. M. *Machine Learning and Knowledge Extraction* (arXiv:2304.00501). https://arxiv.org/abs/2304.00501

## Problema

YOLO evolucionó en muchas versiones con cambios de arquitectura, pérdidas, ánclas y “training tricks”. Falta una visión unificada de **qué cambió en cada generación** y qué lecciones quedan para detección en tiempo real.

## Aporte

Revisión amplia de la familia YOLO desde v1 hasta **YOLOv8**, **YOLO-NAS** y variantes con transformers; incluye métricas/postproceso, arquitecturas y dirección futura.

## Contenido útil (mapa mental)

- Contraste one-stage (YOLO/SSD) vs. two-stage (R-CNN / Faster R-CNN).  
- Línea temporal de innovaciones: grillas, anchors, backbone CSP, FPN/PAN, label assignment, etc.  
- Discusión explícita del trade-off **velocidad ↔ precisión**.  
- Cubren también YOLOX, PP-YOLO y otras ramas que otras reviews cortan en v3/v4.

## Resultados

No es un paper de un modelo nuevo: el “resultado” es un **marco conceptual** para ubicar YOLOv8 (u otra variante) respecto a predecesores y a detectores two-stage.

## Relevancia para nuestro TP

Sirve como **referencia de contexto** en Metodología cuando justifiquemos elegir una arquitectura one-stage tipo YOLO. Evita reinventar la historia de YOLO en el paper y da vocabulario estándar (NMS, mAP, anchors, neck FPN/PAN) alineado a lo que evaluaremos.
