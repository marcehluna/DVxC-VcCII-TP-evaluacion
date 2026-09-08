# Resumen — Recognizing Text on Book Spines with Improved YOLOv11 and Optimized PaddleOCR

**Archivo:** `02_BookSpine_YOLOv11_PaddleOCR_Electronics2025.pdf`  
**Cita:** Li, Z.; Guo, B.; Mu, D. *Electronics* 2025, 14, 4689. https://doi.org/10.3390/electronics14234689

## Problema

Las bibliotecas “inteligentes” necesitan reconocer texto en lomos densos: solapamiento, distorsión tipográfica y errores de estantería (reportan tasas altas de mal ubicados). RFID es caro y frágil; la visión debe detectar el lomo **y** leer el texto.

## Aporte

Pipeline **detección/segmentación + OCR** de punta a punta: YOLOv11 mejorado + PaddleOCR optimizado para lomos.

## Método (ideas clave)

1. **YOLOv11 + CBAM** (atención canal-espacial) para resaltar textura del lomo y atenuar vecinos.  
2. **PaddleOCR** con ajustes de tarea: RecAug, peso de pérdida para texto curvo, diccionario ampliado y tamaño de entrada adaptado.  
3. Flujo: imagen de estantería → segmentar/detectar lomos → recortar → OCR → texto estructurado.

## Resultados destacados

- Segmentación/detección mejorada: **97.4%** de accuracy reportada (+2.1% vs. baseline), con menos cómputo/parámetros.  
- OCR: **CER** de 8.6% → **3.2%**.  
- Sistema end-to-end: **96.8%** de reconocimiento por libro en escenarios reales de estantería; recall de detección de lomos hasta ~94.7% en sus cifras.

## Relevancia para nuestro TP

Es el paper **más alineado** al alcance sugerido por el profesor: núcleo de detección (familia YOLO) + OCR como segunda etapa. Útil para Metodología (diseño del pipeline) y para comparar cómo evalúan texto (CER) vs. cómo nosotros mediremos recuperación de texto de forma más simple. No hace falta copiar CBAM ni YOLOv11: el valor está en la **estructura experimental** detección → OCR.
