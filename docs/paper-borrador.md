# Borrador del paper — Auditoría visual de lomos en estanterías

Documento de trabajo en español. Formato final: IEEE *conference-template-letter.docx*.  
Extensión: se apunta a ~4 páginas como **referencia**, pero la prioridad es la **claridad del contenido** (no recortar argumentos ni tablas esenciales solo por espacio).  
Redacción continua de secciones; la **revisión de contenido** (ajustes de redacción, cifras y tablas) se hace en un pase posterior sobre el borrador completo.

**Decisiones de alcance:**

- Idioma: español (todo el contenido).
- Alcance experimental: evaluación y comparación de **seis** modelos sobre *Book Spine 2* v4:
  1. **YOLOv8s** — detección AABB  
  2. **Faster R-CNN** ResNet-50 FPN — detección AABB  
  3. **YOLOv8s-OBB** — detección con cajas orientadas  
  4. **YOLO26s-OBB** — detección con cajas orientadas  
  5. **YOLO26s-seg** — segmentación de instancias  
  6. **Mask R-CNN** ResNet-50 FPN — segmentación de instancias  
- **OCR** sobre lomos localizados: **incluido en el alcance**; sección propia **VII** (experimentación y conclusiones desde `notebooks/OCR.ipynb`). Contenido de esa sección: **pendiente** hasta cerrar la notebook.
- Organización de resultados: tablas por **tipo de tarea** (AABB / OBB / máscara) más una **tabla resumen** de los seis modelos; OCR en sección/tabla propia. No se mezclan *mAP* de caja AABB, OBB y máscara como si fueran idénticos.
- Matching de catálogo: pendiente de confirmar si entra al cuerpo o queda como trabajo futuro.

---

## Título

**[CONFIRMADO — se puede revisar después]**

Evaluación experimental de YOLOv8, YOLO26 y familias R-CNN para detección y segmentación de lomos de libros

*Descartada (por ahora):*  
Localización de lomos en estanterías: comparación de detectores AABB, OBB y segmentadores de instancias

---

## Resumen

**[CONFIRMADO — se puede revisar después]**

Este trabajo evalúa la localización de lomos en imágenes de estanterías y la recuperación de texto mediante OCR sobre el conjunto *Book Spine 2* (v4). Se comparan seis modelos: YOLOv8s y Faster R-CNN (cajas alineadas a los ejes, AABB), YOLOv8s-OBB y YOLO26s-OBB (cajas orientadas, OBB), y YOLO26s-seg y Mask R-CNN (máscaras de instancia). Se reportan *mAP*, precisión, recall y latencia por tipo de tarea, junto con OCR aplicado a los lomos localizados. Los resultados evidencian compromisos entre calidad de localización, recall en escenas densas o inclinadas y velocidad de inferencia, en función de la representación empleada.

---

## Palabras clave

**[CONFIRMADO — se puede revisar después]**

detección de lomos, cajas orientadas, segmentación de instancias, OCR, YOLOv8, YOLO26, Faster R-CNN, Mask R-CNN, inventario de libros

---

## Tabla comparativa de resultados (material para la sección V)

**[BORRADOR — revisión de contenido diferida]**

### Notas de lectura (importante)

- **No es un único ranking absoluto:** AABB, OBB y máscara optimizan representaciones distintas; el *mAP@0.5* no es estrictamente comparable entre filas de distinto tipo.
- **AABB y segmentación (máscara):** protocolo de informe de las notebooks de test con *conf*/*score* ≥ 0,5 e *IoU* ≥ 0,5 (reevaluación del mejor checkpoint en validación), salvo que se indique lo contrario.
- **OBB:** cifras del *best.pt* validadas por Ultralytics al cierre del entrenamiento en `Model_Test_Yolo8-OBB.ipynb` (25 épocas; val interna Ultralytics sobre 270 imágenes). No hay aún la misma pasada de test/subconjuntos densos-inclinados que en los otros brazos.
- **Latencia OBB:** suma preprocess + inference + postprocess del log de validación Ultralytics (batch de evaluación del trainer; orden de magnitud, no el mismo script batch=1 de las otras notebooks).
- Fuentes AABB/seg: lecturas actualizadas de notebooks / `entrenamientos/*.md` / `results/tables` (Mask); YOLO8-AABB según registro en `entrenamientos/YOLO8-AABB.md` (corrida documentada de 20 épocas).

### Tabla resumen — los seis modelos (métrica principal en validación)

| Modelo | Tarea | *mAP@0.5* | *mAP@0.5:0.95* | Precisión | Recall | Latencia (ms/im.) | Observación |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| YOLOv8s | AABB (caja) | 0,822 | 0,676 | 0,976 | 0,823 | ~14 | Mejor checkpoint época 19 |
| Faster R-CNN | AABB (caja) | 0,799 | 0,603 | 0,965 | 0,791 | ~119 | Mejor checkpoint época 6 (corrida 30 ép.) |
| YOLOv8s-OBB | OBB (caja orientada) | 0,728 | 0,351 | 0,759 | 0,677 | ~31 | Ultralytics val *best.pt*; 25 ép. |
| YOLO26s-OBB | OBB (caja orientada) | 0,848 | 0,532 | 0,835 | 0,761 | ~18 | Ultralytics val *best.pt*; 25 ép. |
| YOLO26s-seg | Máscara | 0,780 | 0,562 | — | — | ~26 | Mejor checkpoint época 24 (corrida 30 ép.) |
| Mask R-CNN | Máscara | 0,785 | 0,507 | 0,785 | 0,834 | ~117 | Mejor checkpoint época 12 |

*Para segmentadores, la columna *mAP* es de **máscara**. Mask R-CNN incluye P/R de máscara del resumen de validación; YOLO26s-seg en la lectura principal prioriza *mAP* de máscara (P/R de máscara en subconjuntos: ver tabla abajo).*

### Detalle AABB (caja) — validación vs test

| Modelo | Partición | *mAP@0.5* | *mAP@0.5:0.95* | Precisión | Recall | Latencia (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s | Val | 0,822 | 0,676 | 0,976 | 0,823 | ~14 |
| YOLOv8s | Test | 0,822 | 0,665 | 0,978 | 0,825 | ~14 |
| Faster R-CNN | Val | 0,799 | 0,603 | 0,965 | 0,791 | ~119 |
| Faster R-CNN | Test | 0,829 | 0,614 | 0,966 | 0,818 | ~124 |

### Detalle OBB (caja orientada) — validación Ultralytics (*best.pt*)

| Modelo | *mAP@0.5* | *mAP@0.5:0.95* | Precisión | Recall | Latencia aprox. (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s-OBB | 0,728 | 0,351 | 0,759 | 0,677 | ~31 |
| YOLO26s-OBB | 0,848 | 0,532 | 0,835 | 0,761 | ~18 |

### Detalle segmentación (máscara) — validación / test / subconjuntos

| Modelo | Corte | *mAP@0.5* | *mAP@0.5:0.95* | Precisión | Recall | Latencia (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| YOLO26s-seg | Val | 0,780 | 0,562 | — | — | ~26 |
| YOLO26s-seg | Test | 0,776 | 0,558 | — | — | ~25 |
| YOLO26s-seg | Val todas (subconj.) | 0,735 | 0,545 | 0,958 | 0,743 | — |
| YOLO26s-seg | Denso | 0,584 | 0,413 | 0,925 | 0,594 | — |
| YOLO26s-seg | Inclinado | 0,620 | 0,437 | 0,962 | 0,627 | — |
| Mask R-CNN | Val | 0,785 | 0,507 | 0,785 | 0,834 | ~117 |
| Mask R-CNN | Test | 0,776 | 0,518 | 0,757 | 0,839 | ~113 |
| Mask R-CNN | Denso | 0,648 | 0,341 | 0,661 | 0,728 | — |
| Mask R-CNN | Inclinado | 0,647 | 0,335 | 0,657 | 0,746 | — |

### Pendientes para homogeneizar la tabla del paper

1. Reevaluar OBB con el mismo script de informe (*conf* ≥ 0,5, test, densos/inclinados) si se quiere paridad con AABB.  
2. Confirmar si YOLOv8s-AABB se reentrenó a 30 épocas (hoy el MD de entrenamientos documenta 20).  
3. Decidir qué figura/tabla entra al IEEE (resumen de seis filas + 1–2 tablas de detalle).

---

## I. Introducción

**[CONFIRMADO — se puede revisar después]**

El control de inventario en bibliotecas y librerías depende con frecuencia de inspecciones visuales por estantería. La automatización a partir de fotografías exige, como mínimo, localizar cada lomo y recuperar el texto visible para contrastarlo con un catálogo. Las cajas alineadas a los ejes (AABB) constituyen un punto de partida habitual; no obstante, el análisis exploratorio de *Book Spine 2* revela lomos inclinados, solapes densos y AABB holgados respecto del polígono anotado, lo que motiva también el uso de cajas orientadas (OBB) y de máscaras de instancia.

En este trabajo se comparan seis modelos —YOLOv8s y Faster R-CNN (AABB), YOLOv8s-OBB y YOLO26s-OBB (OBB), YOLO26s-seg y Mask R-CNN (máscaras)— y se evalúa OCR sobre los lomos localizados. La contribución consiste en una comparación experimental acotada, con el mismo conjunto de datos y criterios de informe por tipo de tarea, orientada al compromiso entre calidad de localización, comportamiento en escenas densas o inclinadas, latencia y utilidad del recorte para OCR.

El resto del artículo describe el conjunto de datos y el protocolo experimental (Secciones III–IV), presenta los resultados de localización (Sección V), discute limitaciones y la elección del localizador (Sección VI), reporta la experimentación con OCR (Sección VII) y cierra con conclusiones y trabajo futuro (Sección VIII).

---

## II. Trabajo relacionado

**[CONFIRMADO — se puede revisar después]**

### A. Inventario visual de lomos

La detección automática de lomos en estanterías se ha propuesto como alternativa a inventarios manuales o a sistemas RFID de elevado costo. Ma *et al.* [1] abordan lomos inclinados y de aspecto extremo mediante un Oriented R-CNN mejorado, y muestran que las cajas alineadas a los ejes degradan la localización cuando el libro no se encuentra en posición vertical. Li *et al.* [2] integran detección y segmentación basadas en YOLO con OCR (PaddleOCR) para recuperar texto en estanterías densas, en un flujo de procesamiento afín al del presente trabajo. A mayor escala, Llabrés *et al.* [3] formulan el inventario como emparejamiento *many-to-many* entre detecciones y un catálogo bajo OCR ruidoso; dicho problema excede el emparejamiento por similitud textual simple y se menciona aquí únicamente como horizonte, no como método adoptado.

### B. Detección: one-stage, two-stage y orientación

Las arquitecturas *one-stage* de la familia YOLO priorizan velocidad e inferencia densa. Terven y Córdova-Esparza [5] revisan esa línea hasta YOLOv8 y YOLO-NAS, y Yaseen [6] detalla la arquitectura de YOLOv8 (backbone tipo CSP, cuello FPN/PAN, cabeza *anchor-free*). En el extremo *two-stage*, Faster R-CNN [4] unifica la generación de propuestas (RPN) y la detección compartiendo cómputo convolucional, con un compromiso clásico entre mayor costo computacional y propuestas explícitas. El protocolo de evaluación en detección moderna (AP/*mAP* sobre umbrales de IoU) está consolidado por MS COCO [9], marco métrico que también se emplea sobre *Book Spine 2*. Las cajas orientadas (OBB), motivadas por [1], buscan reducir el área de fondo que un AABB holgado incorpora en lomos inclinados.

### C. Segmentación de instancias y OCR

Mask R-CNN [7] extiende Faster R-CNN con una rama de máscara por instancia y RoIAlign, lo que permite delimitar el lomo a nivel de píxel en lugar de mediante una caja. Dicha representación resulta pertinente cuando el AABB excede sustancialmente el polígono del lomo y el OCR se alimenta del recorte. En reconocimiento de texto, Shi *et al.* [8] introdujeron CRNN (CNN + RNN + CTC) como modelo extremo a extremo de secuencias; numerosos sistemas OCR prácticos, incluidos los empleados en flujos de lomos [2], heredan la idea de leer el recorte como secuencia. En este trabajo, la comparación experimental abarca AABB, OBB y máscaras, y el OCR se evalúa sobre los lomos localizados.

---

## III. Problema y conjunto de datos

**[BORRADOR — revisión de contenido diferida]**

### A. Problema de localización y lectura

El objetivo consiste en obtener, a partir de una imagen de estantería, lomos localizados y, en la medida de lo posible, texto recuperable por OCR. Ello impone dos requisitos: (i) no omitir instancias en estantes densos ni en libros inclinados, y (ii) que el recorte enviado al OCR contenga predominantemente el lomo correcto, sin incluir de forma significativa al vecino ni al fondo. Un *mAP* de caja elevado no garantiza, por sí solo, un recorte legible.

### B. Conjunto de datos *Book Spine 2*

Se emplea *Book Spine 2* (Roboflow, espacio *bookspine-fxfsx*), versión **v4**, exportación **YOLOv8** con etiquetas en **polígono** (una clase: lomo) y resolución 640×640. Se conserva el particionado oficial: **1.265 / 271 / 120** imágenes (~76 / 16 / 7 %), con aproximadamente 20.354 / 4.622 / 1.747 lomos y densidad similar entre particiones (~15–17 lomos por imagen). La partición de validación se utiliza para la comparación entre modelos; la de prueba se reserva como evaluación final sin ajuste de hiperparámetros. El volumen resulta adecuado para *fine-tuning*; no obstante, el conjunto de prueba es reducido en número de imágenes, por lo que su *mAP* presenta mayor variabilidad.

### C. Análisis exploratorio: variables medidas

Previamente al entrenamiento se realizó un análisis exploratorio centrado en la **geometría de las etiquetas**, en lugar de en estadísticas de color o en proyecciones globales de la imagen. Histogramas RGB o incrustaciones tipo t-SNE mezclan fondo, condiciones de captura y del orden de 16 lomos por imagen, y no predicen de forma directa fallos de detección ni de recorte. Dado que YOLOv8 AABB y Faster R-CNN operan sobre **cajas alineadas a los ejes** derivadas del polígono, el análisis cuantificó:

1. **Ocupación** = área(polígono) / área(AABB).  
2. **Tamaño y aspecto** del AABB (área relativa, razón alto/ancho, percentiles).  
3. **Inclinación** del polígono respecto de la vertical.  
4. **Solape** entre AABB vecinos (máximo IoU por caja).  
5. **Fuga entre particiones (*leakage*)** por imagen fuente en el particionado de Roboflow (mismo nombre base, distinto sufijo `.rf.<hash>`).

### D. Hallazgos del análisis exploratorio (validación, salvo indicación contraria)

**AABB holgado.** Sobre 4.622 lomos de validación, la ocupación media es **0,54** (mediana 0,53; p10 = 0,18; p90 = 0,92). En promedio, el área de la caja es casi el **doble** de la del polígono: la holgura corresponde en gran medida a vecino o fondo. Este resultado desaconseja alimentar el OCR con el AABB sin postprocesado y motiva el uso de OBB y/o máscaras para el recorte.

**Forma extrema.** Las particiones son homogéneas en tamaño: mediana de aproximadamente 44–47 px de ancho y 350–380 px de alto, con razón alto/ancho mediana cercana a **7**. Aunque el criterio COCO [9] clasifica la mayoría como objetos “grandes” por área, la geometría es **alta y estrecha**; el área relativa no refleja la dificultad del objeto. Aproximadamente un 10 % de las cajas son más anchas que altas (aspecto bajo), típicamente asociado a holgura o a disposición en filas.

**Inclinación bimodal.** La mediana de inclinación es de aproximadamente 3,5–3,7° (lomos casi verticales), mientras que la media (~18°) y el percentil 90 (~78°) evidencian una cola pesada: cerca del **18 %** supera 45°. El conjunto no está compuesto exclusivamente por instancias inclinadas; coexisten lomos verticales y fuertemente rotados. Ello motiva el subconjunto *inclinado* y la inclusión del brazo OBB.

**Estanterías densas.** El máximo IoU mediano entre AABB vecinos es **0,27**; el **76 %** de las cajas solapa otra con IoU ≥ 0,1 y el **23 %** alcanza IoU ≥ 0,5. Un NMS agresivo en 0,5 puede eliminar verdaderos positivos. De ahí la definición del subconjunto *denso* (imagen con al menos un GT cuyo máximo IoU con otro GT ≥ 0,5) y la adopción de un NMS más permisivo en detección.

**Fuga por archivo, no por escena original.** El particionado se realiza a nivel de archivo Roboflow. Existen 1.218 fuentes distintas, de las cuales 104 cruzan particiones: el **27 %** de validación y el **33 %** de prueba tienen una imagen hermana en entrenamiento. El *mAP* absoluto puede resultar optimista frente a estanterías nunca observadas; la comparación relativa entre modelos sobre el **mismo** particionado permanece válida. Este sesgo se declara explícitamente.

### E. Implicaciones para el diseño experimental

El análisis exploratorio condiciona el protocolo posterior: (i) no reducir `imgsz` por debajo de 640; (ii) reportar métricas globales y en los subconjuntos denso e inclinado; (iii) comparar AABB, OBB y segmentación como respuestas distintas al mismo diagnóstico geométrico; (iv) tratar el OCR como etapa sensible a la calidad del recorte (ajuste de caja, rotación o máscara), y no como lectura directa del AABB sin postprocesado.

---

## IV. Metodología

**[BORRADOR — revisión de contenido diferida]**

### A. Diseño experimental

Se comparan seis modelos bajo tres representaciones de localización, motivadas por el análisis exploratorio (Sección III): **AABB** (YOLOv8s, Faster R-CNN ResNet-50 FPN), **OBB** (YOLOv8s-OBB, YOLO26s-OBB) y **máscara de instancia** (YOLO26s-seg, Mask R-CNN ResNet-50 FPN). Se conserva el particionado oficial de *Book Spine 2* v4, sin reasignación de imágenes. La validación se emplea para la selección de *checkpoint* y la comparación; la prueba constituye una evaluación única sin ajuste de umbrales. Adicionalmente se reportan dos subconjuntos de validación definidos por geometría de etiquetas: *denso* (imagen con al menos un GT cuyo máximo IoU AABB con otro GT ≥ 0,5) e *inclinado* (GT con inclinación > 45°).

### B. Entrenamiento por familia

Todos los experimentos parten de pesos preentrenados en COCO y realizan *fine-tuning* a una clase (*lomo*). La resolución efectiva de entrada se mantiene en al menos 640 px (`imgsz` / `min_size`), de acuerdo con el análisis exploratorio. Se fija semilla 42 cuando el marco de trabajo lo permite. El hardware de referencia es Google Colab con GPU Tesla T4.

**Familia YOLO (Ultralytics).** Se entrenan YOLOv8s (AABB), YOLO26s-seg y las variantes OBB YOLOv8s-OBB y YOLO26s-OBB. Hiperparámetros típicos en AABB y segmentación: *imgsz* = 640, *batch* = 16, SGD (*lr0* = 0,01, *lrf* = 0,01, *momentum* = 0,937), NMS con *iou* = 0,40 (más permisivo que el valor por defecto ~0,7, motivado por el solape observado), *max_det* = 300. Corridas de referencia: YOLOv8s-AABB, 20 épocas (mejor *checkpoint* en la época 19); YOLO26s-seg, hasta 30 épocas con *patience* = 5 (mejor época 24 según *mAP* de máscara); OBB, 25 épocas, SGD con *lr0* = 1×10⁻³ y *batch* = 32. El criterio de selección del mejor modelo es el *fitness* de Ultralytics / *mAP@0.5* de la salida principal (caja o máscara, según el brazo).

**Familia R-CNN (torchvision).** Se entrenan Faster R-CNN y Mask R-CNN con ResNet-50 FPN (variante v2 en Faster), con etiquetas convertidas al formato COCO. Parámetros: *min_size* = 640, *max_size* = 1333; *batch* = 2; SGD con *lr* = 0,005 y *momentum* = 0,9; NMS de cajas 0,40 en Faster (alineado a YOLO). El *checkpoint* se selecciona por el mayor *mAP@0.5* (caja o máscara) en validación. En las corridas de referencia, Faster alcanza su mejor *mAP@0.5* en la época 6 (extender el entrenamiento a 30 épocas no mejora dicha métrica); Mask lo hace en la época 12.

### C. Protocolo de evaluación

Se reportan *mAP@0.5*, *mAP@0.5:0.95*, precisión, recall y latencia media (ms/imagen, *batch* = 1 en AABB y segmentación). En AABB y segmentación, los umbrales de informe son *conf*/*score* ≥ 0,5 e *IoU* ≥ 0,5 al reevaluar el mejor *checkpoint*. Los *mAP* de caja AABB, OBB y máscara **no** se agregan en un *ranking* único: se tabulan por tipo de tarea y, de forma complementaria, en un resumen de seis filas con la correspondiente advertencia de comparabilidad. Las cifras OBB corresponden a la validación Ultralytics del *best.pt* al cierre del entrenamiento; en la presente versión no incluyen la misma pasada de prueba y subconjuntos que AABB y segmentación, lo que se declara como limitación del protocolo.

### D. OCR sobre lomos localizados

Tras la detección, el flujo de OCR no se alimenta del AABB sin postprocesado: el recorte se obtiene a partir del polígono o la máscara (o de la caja orientada cuando corresponde), se alinea el texto mediante rotación y se aplica un preprocesado previo a la lectura con motores de OCR de uso práctico (p. ej., EasyOCR o PaddleOCR). Los umbrales de detección se alinean al resto del experimento (*conf* = 0,5, NMS *iou* = 0,40). La experimentación y las conclusiones de OCR se desarrollan en la Sección VII. El emparejamiento fino contra un catálogo queda fuera del cuerpo experimental o se contempla como trabajo futuro (Sección VIII).

---

## V. Experimentos y resultados

**[BORRADOR — revisión de contenido diferida]**

*(Nota de borrador: las tablas numéricas canónicas figuran al inicio de este documento; en la versión IEEE se insertarán una o dos tablas de detalle junto con el resumen de seis filas.)*

### A. Detección AABB

En validación, YOLOv8s obtiene *mAP@0.5* = 0,822 y Faster R-CNN 0,799, ambos con precisión elevada (~0,97) y recall inferior (~0,79–0,82). La latencia favorece a YOLO (~14 ms frente a ~119 ms por imagen). En prueba, Faster R-CNN alcanza 0,829 en *mAP@0.5*; dado el tamaño reducido de esa partición, dicho valor no se emplea para la selección de modelo. En los subconjuntos denso e inclinado, ambos detectores degradan su rendimiento: la limitación principal es el recall, en coherencia con el solape entre cajas y con el NMS.

### B. Detección OBB

Bajo el protocolo de validación Ultralytics al cierre del entrenamiento, YOLO26s-OBB (*mAP@0.5* = 0,848) supera a YOLOv8s-OBB (0,728) y presenta menor latencia (~18 ms frente a ~31 ms). El *mAP@0.5:0.95* resulta inferior al de AABB (0,53 frente a ~0,60–0,68), lo cual es esperable al exigir ajuste angular. Estas cifras no corresponden aún a una reevaluación con *conf* ≥ 0,5 ni a las particiones de prueba y subconjuntos; se interpretan exclusivamente dentro del brazo OBB.

### C. Segmentación de instancias

YOLO26s-seg y Mask R-CNN exhiben *mAP@0.5* de máscara próximos en validación (0,780 frente a 0,785) y en prueba (~0,776 en ambos). Mask R-CNN obtiene mayor recall (0,834) a costa de una latencia cercana a 117 ms; YOLO26s-seg opera en torno a 26 ms. En los subconjuntos denso e inclinado, el *mAP@0.5* desciende a aproximadamente 0,58–0,65; Mask R-CNN conserva mejor recall en esos cortes, mientras que YOLO26s-seg mantiene mayor precisión. La brecha entre métricas de caja y de máscara, cuando ambas se miden, confirma que localizar el AABB no equivale a segmentar el polígono.

### D. Remisión a OCR

Los resultados cuantitativos y las conclusiones del flujo de OCR (localización con YOLO26s-seg, recortes y motores de lectura) se presentan en la Sección VII, una vez cerrada la notebook correspondiente.

---

## VI. Discusión

**[BORRADOR — revisión de contenido diferida]**

El análisis exploratorio anticipa los patrones de error observados: una ocupación AABB media de ~0,54 implica inclusión de fondo o vecino en el recorte; el solape denso explica caídas de recall bajo NMS; la cola de inclinación justifica el brazo OBB sin que el conjunto esté compuesto únicamente por instancias rotadas. En cuanto a velocidad, las variantes YOLO superan a las R-CNN en un factor aproximado de 5–8×, con *mAP@0.5* comparable por tipo de tarea. Mask R-CNN no supera de forma consistente a YOLO26s-seg en *mAP* de máscara global; no obstante, su mayor recall en escenas densas o inclinadas puede resultar preferible cuando el objetivo es un inventario exhaustivo frente a un requisito de tiempo real.

Se identifican las siguientes limitaciones: (i) fuga del 27–33 % por imagen fuente entre particiones —el *mAP* absoluto puede ser optimista—, aunque la comparación relativa entre modelos sobre el mismo particionado permanece útil; (ii) la partición de prueba (120 imágenes) introduce variabilidad en las métricas; (iii) el protocolo OBB aún no está homogeneizado con el de AABB y segmentación; (iv) OCR y emparejamiento de catálogo no alcanzan el mismo nivel de reporte tabular que la detección.

A partir de estos resultados se adopta **YOLO26s-seg** como localizador para el flujo de OCR de este trabajo. La decisión se sustenta en tres criterios alineados al diagnóstico del análisis exploratorio y al uso previsto del recorte: (i) la representación por máscara mitiga la holgura del AABB (ocupación media ~0,54), al delimitar el lomo a nivel de píxel y reducir fondo o vecino en el recorte enviado al OCR; (ii) el *mAP@0.5* de máscara en validación (0,780) es comparable al de Mask R-CNN (0,785), sin incurrir en su latencia (~26 ms frente a ~117 ms por imagen); (iii) frente a los detectores AABB, ofrece un recorte más ajustado sin sacrificar el compromiso calidad–velocidad característico de la familia YOLO. Aunque Mask R-CNN puede conservar mayor recall en subconjuntos densos o inclinados, y aunque YOLO26s-OBB alcanza un *mAP@0.5* elevado bajo su protocolo propio, YOLO26s-seg concentra en un único modelo la representación geométrica adecuada para OCR, una calidad de máscara competitiva y una latencia compatible con un pipeline de inventario visual.

---

## VII. Experimentación con OCR

**[PENDIENTE — rellenar al cerrar `notebooks/OCR.ipynb`]**

*(Placeholder. Fuente de verdad: `notebooks/OCR.ipynb` y artefactos asociados — p. ej. `resultados_ocr/`, `notebooks/data/gt_ocr.csv`. No redactar cifras ni conclusiones hasta que la notebook esté cerrada.)*

### A. Protocolo experimental

*(Pendiente: fotos propias / GT; localizador YOLO26s-seg; recorte desde máscara; preprocesado; motores EasyOCR / PaddleOCR; métricas de lectura o similitud.)*

### B. Resultados

*(Pendiente: tablas y observaciones cuantitativas / cualitativas.)*

### C. Conclusiones de OCR

*(Pendiente: síntesis de hallazgos del brazo OCR y su vínculo con la calidad del recorte.)*

---

## VIII. Conclusiones y trabajo futuro

**[BORRADOR — revisión de contenido diferida]**

Sobre *Book Spine 2* v4 se compararon seis modelos en AABB, OBB y máscara de instancia, junto con un flujo de OCR sensible a la calidad del recorte (Sección VII). No existe un único modelo óptimo en todos los criterios: las variantes YOLO ofrecen el mejor compromiso entre calidad y latencia; OBB y máscaras responden al diagnóstico geométrico del análisis exploratorio cuando el AABB holgado degrada el recorte; las escenas densas e inclinadas permanecen como el principal punto débil en recall.

Como trabajo futuro se plantea homogeneizar la evaluación OBB (prueba y subconjuntos bajo el mismo umbral), ampliar la cuantificación de OCR según el tipo de localizador, explorar NMS y umbrales adaptados a la densidad, y —si se incorpora emparejamiento de catálogo— tratarlo como etapa independiente del *mAP* de detección [3].

---

## Referencias

*(Lista alineada a `carpeta-referencias/papers/`; se irá citando el resto del artículo con estos números.)*

[1] H. Ma, C. Wang, A. Li, A. Xu y D. Han, “An Accurate Book Spine Detection Network Based on Improved Oriented R-CNN,” *Sensors*, vol. 24, no. 24, art. 7996, 2024, doi: 10.3390/s24247996.

[2] Z. Li, B. Guo y D. Mu, “Recognizing Text on Book Spines with Improved YOLOv11 and Optimized PaddleOCR,” *Electronics*, vol. 14, no. 23, art. 4689, 2025, doi: 10.3390/electronics14234689.

[3] A. Llabrés, A. U. Dey, D. Karatzas y E. Valveny, “Image-text matching for large-scale book collections,” arXiv:2407.19812, 2024.

[4] S. Ren, K. He, R. Girshick y J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” arXiv:1506.01497, 2015.

[5] J. Terven y D. M. Córdova-Esparza, “A Comprehensive Review of YOLO Architectures in Object Detection: From YOLOv1 to YOLOv8 and YOLO-NAS,” arXiv:2304.00501, 2023.

[6] M. Yaseen, “What is YOLOv8: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector,” arXiv:2408.15857, 2024.

[7] K. He, G. Gkioxari, P. Dollár y R. Girshick, “Mask R-CNN,” en *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, arXiv:1703.06870.

[8] B. Shi, X. Bai y C. Yao, “An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition,” *IEEE Trans. Pattern Anal. Mach. Intell.*, 2017, arXiv:1507.05717.

[9] T.-Y. Lin *et al.*, “Microsoft COCO: Common Objects in Context,” en *ECCV*, 2014, arXiv:1405.0312.
