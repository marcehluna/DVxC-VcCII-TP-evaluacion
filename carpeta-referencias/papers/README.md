# Papers de referencia — TP auditoría bibliográfica

Selección alineada al alcance del TP: **detección de lomos (núcleo)**, comparación de arquitecturas, **OCR** y **matching** simple.

Cada PDF tiene un resumen one-page: `Resumen_<mismo_nombre_del_paper>.md`.

| Archivo | Año | Para qué sirve en el TP |
| :--- | :--- | :--- |
| `01_Yang2024_BookSpine_OrientedRCNN_Sensors.pdf` | 2024 | **Caso de aplicación reciente:** detección de lomos en bibliotecas (oriented R-CNN, mAP, lomos inclinados). Ideal para Introducción / trabajos relacionados. |
| `02_BookSpine_YOLOv11_PaddleOCR_Electronics2025.pdf` | 2025 | **Pipeline completo cercano al nuestro:** detección (YOLO) + OCR (PaddleOCR) en lomos. Referencia directa para metodología y métricas de texto. |
| `03_Ioli2024_ImageTextMatching_BookCollections.pdf` | 2024 | Matching imagen–texto / inventario a escala (SAM + OCR + matching). Útil para discutir la etapa C *sin* complejizarla: qué hacen ellos vs. nuestra similitud de texto. |
| `04_Ren2015_FasterRCNN.pdf` | 2015 | Paper fundacional de **Faster R-CNN** (candidato two-stage en la comparación). |
| `05_Terven2023_YOLO_Architectures_Review.pdf` | 2023 | Revisión de arquitecturas YOLO hasta v8 / YOLO-NAS. Contexto para justificar one-stage. |
| `06_Yaseen2024_WhatIsYOLOv8.pdf` | 2024 | Análisis de **YOLOv8** (arquitectura, pérdidas, rendimiento). Complementa la cita del software Ultralytics. |
| `07_He2017_MaskRCNN.pdf` | 2017 | **Mask R-CNN** (detección + máscaras). Alternativa / contraste si se menciona segmentación de lomos. |
| `08_Shi2017_CRNN_OCR.pdf` | 2017 | **CRNN** clásico para reconocimiento de texto en imagen. Base teórica de muchos OCR de secuencia. |
| `09_Lin2014_COCO_MicrosoftCOCO.pdf` | 2014 | Dataset/protocolo **COCO**; contexto de mAP@IoU usado en evaluación de detección. |

## Enlaces canónicos

1. Yang et al., *Sensors* 2024 — https://doi.org/10.3390/s24247996  
2. YOLO + PaddleOCR, *Electronics* 2025 — https://doi.org/10.3390/electronics14234689  
3. Ioli et al., arXiv:2407.19812 — https://arxiv.org/abs/2407.19812  
4. Ren et al., Faster R-CNN — https://arxiv.org/abs/1506.01497  
5. Terven et al., YOLO review — https://arxiv.org/abs/2304.00501  
6. Yaseen, What is YOLOv8 — https://arxiv.org/abs/2408.15857  
7. He et al., Mask R-CNN — https://arxiv.org/abs/1703.06870  
8. Shi et al., CRNN — https://arxiv.org/abs/1507.05717  
9. Lin et al., COCO — https://arxiv.org/abs/1405.0312  

## Sugerencia de uso en el paper (≥5 referencias)

Priorizar **01** y **02** como casos reales &lt; 8 años; **04** + **06** (o **05**) para la comparación experimental; **08** o **02** para OCR; **09** para métricas; **03** solo si se discute matching.
