# Resumen — Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

**Archivo:** `04_Ren2015_FasterRCNN.pdf`  
**Cita:** Ren, S.; He, K.; Girshick, R.; Sun, J. *IEEE TPAMI* (arXiv:1506.01497). https://arxiv.org/abs/1506.01497

## Problema

Tras Fast R-CNN, el cuello de botella dejó de ser la clasificación de regiones y pasó a ser la **generación de propuestas** (Selective Search ~2 s/imagen en CPU). Hacía falta propuestas baratas y de calidad, preferentemente en GPU y compartiendo cómputo con el detector.

## Aporte

Introducen la **Region Proposal Network (RPN)**: una red fully convolutional que predice cajas y objectness sobre el feature map, entrenada end-to-end y **compartiendo** convoluciones con Fast R-CNN. El sistema unificado se conoce como **Faster R-CNN**.

## Método (ideas clave)

- RPN desliza una ventana sobre el mapa de features y usa **anchors** multi-escala / multi-aspecto.  
- Entrenamiento alternado (o aproximaciones joint) RPN ↔ Fast R-CNN con features compartidos.  
- En inferencia, ~**300 propuestas**/imagen bastan para SOTA de la época.

## Resultados destacados

Con VGG-16: ~**5 fps** en GPU (pipeline completo) y precisión SOTA en PASCAL VOC 2007/2012 y MS COCO. RPN + Faster R-CNN fueron base de varios 1.er puestos en ILSVRC/COCO 2015.

## Relevancia para nuestro TP

Candidato natural como **arquitectura two-stage** en la comparación experimental del núcleo (frente a un YOLO one-stage). Citarlo permite justificar: más preciso en muchos escenarios clásicos, más lento; útil para discutir trade-off **mAP vs. tiempo de inferencia**, exactamente las métricas que pide el profesor.
