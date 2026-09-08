# Resumen — Mask R-CNN

**Archivo:** `07_He2017_MaskRCNN.pdf`  
**Cita:** He, K.; Gkioxari, G.; Dollár, P.; Girshick, R. ICCV 2017 (arXiv:1703.06870). https://arxiv.org/abs/1703.06870

## Problema

La detección da cajas; muchas aplicaciones necesitan **máscaras de instancia** (píxeles del objeto). Extender Faster R-CNN a segmentación de instancias de forma simple y precisa era un desafío abierto.

## Aporte

**Mask R-CNN**: añade una rama paralela de predicción de máscara binaria por RoI, sobre Faster R-CNN. Introduce **RoIAlign** (sin cuantización agresiva) para alinear features y máscaras.

## Método (ideas clave)

- Multi-task: clasificación + box regression + **mask** por instancia.  
- RoIAlign mejora AP de máscara de forma notable vs. RoIPool.  
- Flexible: también se adapta a pose estimation.

## Resultados destacados

Resultados top en tracks de COCO (detección, segmentación, keypoints en la época). Ejemplo citado: ~**35.7 mask AP** con ResNet-101; ~5 fps en configuraciones reportadas.

## Relevancia para nuestro TP

No es obligatorio usarlo: el núcleo acordado es **detección** (cajas). Sirve si queremos mencionar segmentación de lomos (como en papers 01/02) o justificar por qué nos quedamos en bounding boxes (más simple, suficiente para OCR por recorte). También conecta históricamente Faster R-CNN → Mask R-CNN.
