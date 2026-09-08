# Ideas seleccionadas — VxC 2

## 1. Analítica táctica en fútbol + equipos sin supervisión

**Contexto y problema a resolver**

En videos de partidos transmitidos, la cámara se mueve casi todo el tiempo y los jugadores se cruzan o se tapan entre sí. Hace falta no solo detectar quién está en la cancha (jugadores, arquero, árbitro, pelota), sino también seguir a cada uno a lo largo del tiempo y separar automáticamente a los dos equipos. Sin eso, cualquier análisis táctico (formación, ocupación del campo, interacciones) queda incompleto o hay que etiquetar a mano.

**Link al dataset**

- Detección principal (Roboflow — *football-players-detection-3zvbc*): https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
- Complemento de segmentación (Kaggle): https://www.kaggle.com/datasets/ihelon/football-player-segmentation
- Secuencias / tracking (SoccerTrack): https://github.com/AtomScott/SoccerTrack · v2: https://github.com/AtomScott/SoccerTrack-v2

**Solución propuesta**

Entrenar un detector/segmentador (p. ej. YOLO con máscaras) para las clases del juego y acoplarlo a un rastreador pensado para cámara en movimiento (BoT-SORT con compensación del paneo). Sobre la región de la camiseta de cada jugador, agrupar colores de forma no supervisada (*K-Means* en un espacio de color estable a la luz) para asignar equipo, con votación a lo largo de varios cuadros para estabilizar la etiqueta.

**Potenciales riesgos**

La pelota y los jugadores lejos se detectan a ratos, y eso hace que las identidades “salten” de un cuadro a otro. Cuando la cámara se mueve rápido, hace zoom o hay un corte de transmisión, corregir ese movimiento suele fallar y el seguimiento se degrada. Separar equipos solo por color de camiseta es frágil si los colores se parecen, si entra el árbitro en la mezcla o si cambia la luz del estadio. En jugadas apretadas es fácil confundir quién es quién. Además, al juntar varias etapas pesadas, en una GPU típica de Colab el prototipo puede quedar demasiado lento para una demo cómoda.

---

## 2. Seguimiento aéreo multiclase (VisDrone)

**Contexto y problema a resolver**

Las secuencias tomadas desde drones muestran vehículos y personas desde arriba, a menudo muy chicos en la imagen, con cambios de altura, ángulo e iluminación. El movimiento de la propia cámara rompe el supuesto de “fondo quieto” que usan muchos rastreadores. El problema es clasificar varias categorías a la vez y mantener la identidad de cada objeto a lo largo del video, midiendo si compensar el movimiento del dron realmente mejora el seguimiento.

**Link al dataset**

- VisDrone-MOT / AISKYEYE (hub y descargas): https://github.com/VisDrone/VisDrone-Dataset

**Solución propuesta**

Entrenar un detector multiclase adaptado a objetos pequeños (p. ej. YOLO u otro modelo con buena resolución de entrada) sobre VisDrone-MOT y acoplarlo a un rastreador con compensación de movimiento global (BoT-SORT + flujo óptico disperso). Comparar de forma explícita variantes con y sin esa compensación (y, si alcanza el alcance, otros rastreadores) sobre un subconjunto manejable de secuencias, reportando métricas de seguimiento además de la detección.

**Potenciales riesgos**

Desde el aire mucha gente y vehículos se ven como puntos: el modelo puede dar números aceptables en promedio y aun así fallar en peatones o bicis lejanos. Bajar y entrenar el dataset completo consume mucho tiempo, disco y cuotas de Colab o Kaggle. Si no se compensa bien el movimiento del dron, las trayectorias se parten una y otra vez. Con muchas clases desiguales, en un subconjunto chico las categorías raras casi no aparecen y el prototipo no las aprende. Comparar varios rastreadores sin una evaluación de seguimiento bien armada deja la comparación a ojo y difícil de defender.

---

## 3. Sistema inteligente de auditoría de inventario bibliográfico (detección + OCR)

**Contexto y problema a resolver**

Las auditorías de stock en bibliotecas universitarias y librerías comerciales suelen ser manuales, lentas y propensas a errores: hay que revisar ejemplar por ejemplar para saber qué hay en cada estantería. El problema es pasar de una foto (o varias) de una estantería llena a un listado usable de títulos presentes, con la menor intervención humana posible.

**Link al dataset**

- Roboflow — *Book Spine 2*: https://universe.roboflow.com/bookspine-fxfsx/book_spine_2

**Solución propuesta**

Construir un prototipo que, a partir de imágenes de estanterías, detecte cada lomo con un modelo de detección entrenado sobre *Book Spine 2* (comparando arquitecturas, p. ej. YOLOv8 vs. Faster R-CNN, según mAP y latencia en Colab). Sobre cada caja detectada, aplicar OCR para leer el texto del lomo, limpiarlo/normalizarlo y cruzarlo con una base de datos de inventario para marcar presentes, faltantes o no reconocidos.

**Potenciales riesgos**

Los lomos chicos, torcidos, con poca luz o tapados por otros libros hacen que la detección falle o recorte mal el texto. El OCR se degrada con tipografías decorativas, idiomas mezclados o títulos verticales. Si el dataset de entrenamiento no se parece a las estanterías reales (ángulo, densidad, calidad de foto), el prototipo “anda” en el set de prueba y falla en el caso de uso. Cruzar el texto leído con el catálogo exige normalización cuidadosa: un error de un carácter puede marcar un libro como ausente. Por último, comparar dos detectores y además integrar OCR puede dejar poco tiempo para una demo cerrada y evaluada con números claros.
