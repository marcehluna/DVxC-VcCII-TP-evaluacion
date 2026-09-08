# Resumen — Microsoft COCO: Common Objects in Context

**Archivo:** `09_Lin2014_COCO_MicrosoftCOCO.pdf`  
**Cita:** Lin, T.-Y. et al. ECCV 2014 (arXiv:1405.0312). https://arxiv.org/abs/1405.0312

## Problema

Los datasets clásicos (PASCAL VOC, ImageNet detection) limitaban el progreso: pocas categorías en contexto rico, o reconocimiento sin suficiente énfasis en **localización precisa** y objetos en escenas naturales no icónicas.

## Aporte

Presentan **MS COCO**: dataset masivo de detección, segmentación y captioning, con objetos en contexto cotidiano, múltiples instancias por imagen y anotaciones detalladas (cajas y máscaras).

## Ideas / diseño del dataset

- Énfasis en objetos no centrados / no “poseídos” para la cámara (non-iconic views).  
- Muchas instancias por imagen y relaciones contextuales.  
- Protocolo de evaluación que popularizó el uso de **AP promedio sobre IoU** (mAP estilo COCO: varios umbrales IoU, no solo 0.5).

## Impacto

Se convirtió en el benchmark de referencia de detección/segmentación; casi todo detector moderno (incl. YOLO y Faster/Mask R-CNN) reporta COCO mAP.

## Relevancia para nuestro TP

No entrenamos en COCO como tarea final, pero **sí** usamos el lenguaje de evaluación que COCO consolidó: mAP, precisión/recall a IoU dados. Citarlo (o el protocolo COCO vía herramientas tipo Ultralytics) legitima cómo reportamos el núcleo experimental sobre *Book Spine 2*, aunque el dominio sea lomos y no las 80 clases COCO.
