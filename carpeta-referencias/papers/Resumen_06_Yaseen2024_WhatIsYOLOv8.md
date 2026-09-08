# Resumen — What is YOLOv8: An In-Depth Exploration of the Internal Features

**Archivo:** `06_Yaseen2024_WhatIsYOLOv8.pdf`  
**Cita:** Yaseen, M. arXiv:2408.15857, 2024. https://arxiv.org/abs/2408.15857  
**Nota:** Ultralytics no publicó paper formal de YOLOv8; este análisis complementa la cita del software.

## Problema

YOLOv8 se usa mucho en la práctica, pero su documentación “viva” (repo/docs) no sustituye un análisis académico de arquitectura, entrenamiento y rendimiento frente a YOLOv5 y otros detectores.

## Aporte

Exploración en profundidad de **YOLOv8**: backbone, neck, cabeza anchor-free, técnicas de entrenamiento y desempeño en COCO / Roboflow 100; también comenta el ecosistema (paquete Python / CLI).

## Método / ideas de arquitectura destacadas

- Backbone tipo **CSPNet** / bloques C2f para features más ricos.  
- Neck **FPN + PAN** para detección multiescala.  
- Enfoque **anchor-free** con cabeza desacoplada (objectness / clase / regresión).  
- Variantes por tamaño (n/s/m/l/x) y trade-off accuracy vs. latencia.

## Resultados (enfoque del paper)

Posiciona YOLOv8 como solución SOTA práctica: alta precisión con capacidad de tiempo real en hardware diverso; enfatiza usabilidad para entrenamiento y despliegue.

## Relevancia para nuestro TP

Si el brazo one-stage del experimento es **YOLOv8** ( Ultralytics ), este paper + la cita de software cubren la justificación técnica sin inventar arquitectura. Apoya la sección de comparación: por qué esperar buen mAP con latencia competitiva frente a Faster R-CNN en Colab.
