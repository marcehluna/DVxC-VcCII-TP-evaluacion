# Resumen — An Accurate Book Spine Detection Network Based on Improved Oriented R-CNN

**Archivo:** `01_Yang2024_BookSpine_OrientedRCNN_Sensors.pdf`  
**Cita:** Ma, H.; Wang, C.; Li, A.; Xu, A.; Han, D. *Sensors* 2024, 24, 7996. https://doi.org/10.3390/s24247996

## Problema

En bibliotecas, el inventario visual depende de detectar **lomos** en fotos de estanterías. Los detectores convencionales (cajas axis-aligned) sufren con lomos **inclinados**, relaciones de aspecto extremas y deformaciones geométricas.

## Aporte

Proponen un **Oriented R-CNN mejorado** para detección de lomos con cajas orientadas, pensado para inventario inteligente sin RFID costoso.

## Método (ideas clave)

1. **Convoluciones deformables (DCN)** en bloques residuales de ResNet50, para modelar mejor la geometría del lomo.  
2. **PAFPN** en el cuello, para fusión multiescala más rica.  
3. **Ánclas adaptadas** vía K-median con inicialización de centros adaptativa, alineadas al dataset de lomos.

## Resultados destacados

Reportan **mAP 90.22%** en su dataset de lomos, **+4.47 pp** respecto al baseline de Oriented R-CNN. El dataset propio incluye miles de instancias (promedio ~23 lomos/imagen) con muchos casos inclinados.

## Relevancia para nuestro TP

Caso de aplicación **reciente (&lt; 8 años)** casi idéntico al problema: detección de lomos para auditoría. Sirve para Introducción / trabajos relacionados y para motivar métricas de detección (mAP). Nosotros comparamos arquitecturas axis-aligned (p. ej. YOLO vs. Faster R-CNN); este paper muestra por qué la **orientación** del lomo importa y qué fallas esperar en estanterías reales.
