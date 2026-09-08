# Concepto del TP — Auditoría de inventario bibliográfico

**Estado:** opción elegida (recomendación del profesor).  
**Materia:** Visión por Computadora II — CEIA.

## Definición en una frase

Prototipo que detecta lomos de libros en imágenes de estanterías, compara experimentalmente dos arquitecturas de detección y extiende el pipeline con OCR y un matching sencillo contra un catálogo de demostración.

## Orientación del profesor (alcance)

De las tres ideas, esta es la más factible para el tiempo del curso: combina detección, comparación de arquitecturas y posprocesamiento con OCR en un pipeline completo, **si se acota el alcance**.

Prioridad acordada:

1. **Núcleo (objetivo principal):** detección de lomos + comparación entre dos arquitecturas, con métricas de detección y tiempo de inferencia.  
2. **Segunda etapa:** OCR sobre los lomos detectados y evaluación de qué tan bien se recupera el texto.  
3. **Etapa final (simple):** matching contra un catálogo mediante similitud de texto, sin convertirlo en otro problema complejo.

OCR y matching son **extensiones** del sistema; no deben competir con el rigor del experimento de detección.

## Problema del mundo real

Las auditorías de stock en bibliotecas y librerías suelen ser manuales, lentas y propensas a error. El caso de uso es pasar de fotos de estanterías a indicios de qué títulos están presentes, con intervención humana reducida.

## Encaje con el TP

| Requisito del curso | Cómo se cubre |
| :--- | :--- |
| Eje central (obligatorio) | **Detección de objetos** sobre lomos (núcleo evaluado con rigor) |
| Capacidad adicional (valorable) | **OCR** + matching simple por similitud de texto |
| Justificación técnica | Comparar dos arquitecturas (p. ej. YOLOv8 vs. Faster R-CNN) con evidencia experimental |
| Prototipo funcional | Pipeline: imagen → detección → OCR → matching de catálogo |
| Experimentación | Fine-tuning e inferencia en Google Colab (u GPU equivalente) |

## Dataset de partida

- **Roboflow — Book Spine 2:** https://universe.roboflow.com/bookspine-fxfsx/book_spine_2

## Alcance por etapas

### Etapa A — Núcleo (obligatoria)

- Entrenar / ajustar **dos** detectores de lomos sobre el dataset.  
- Evaluar y comparar al menos: **mAP**, **precisión**, **recall** y **tiempo de inferencia**.  
- Documentar por qué se eligió cada arquitectura y qué trade-offs se observan (calidad vs. velocidad).  
- Entregar visualizaciones de detecciones sobre imágenes de validación / demo.

### Etapa B — Extensión OCR

- Recortar cada lomo detectado y aplicar OCR.  
- Evaluar la calidad de recuperación de texto (p. ej. exactitud / similitud respecto a una referencia en un subconjunto etiquetado a mano, o análisis cualitativo + cuantitativo acotado).  
- Normalización básica del texto (mayúsculas, signos) solo lo necesario para la etapa C.

### Etapa C — Extensión matching de catálogo

- Catálogo de demostración (lista de títulos, no integración con ILS/OPAC real).  
- Matching por **similitud de texto** (umbral configurable).  
- Salida interpretable: presente / no encontrado / baja confianza.  
- Sin formulaciones complejas (no retrieval avanzado, no embeddings obligatorios, no sistema de inventario completo).

## Fuera de alcance

- App móvil o despliegue en producción.  
- Integración con sistemas bibliotecarios reales.  
- Matching como problema de investigación (solo similitud de texto suficiente para la demo).  
- Seguimiento en video o reconstrucción 3D de la estantería.  
- Explorar muchas arquitecturas: el compromiso es **dos**, bien medidas.

## Tareas del TP (orden sugerido)

1. Preparar dataset (*Book Spine 2*), splits y protocolo de evaluación de detección.  
2. Entrenar arquitectura A y arquitectura B; registrar hiperparámetros y cómputo.  
3. Comparar mAP, precisión, recall y tiempo de inferencia; elegir modelo “principal” para el pipeline demo.  
4. Integrar OCR sobre cajas del modelo elegido; medir / ilustrar calidad de texto recuperado.  
5. Armar catálogo demo y matching por similitud; cerrar demo de punta a punta.  
6. Redactar paper IEEE (énfasis en resultados del núcleo) + README reproducible + video de demo.

## Riesgos a vigilar

- Subestimar el tiempo del núcleo (dos entrenamientos + métricas) y adelantar OCR/matching sin evidencia sólida de detección.  
- Lomos chicos, torcidos o mal iluminados → detecciones débiles y recortes malos para OCR.  
- Domain gap Roboflow vs. estanterías reales.  
- OCR frágil con tipografías decorativas o títulos verticales.  
- Matching sensible a errores de OCR: mantenerlo simple y documentar fallos, no “parchearlo” con complejidad extra.

## Criterio de éxito

El TP está bien encaminado si:

1. **Núcleo cumplido:** hay comparación experimental clara entre dos detectores con mAP, precisión, recall y tiempo de inferencia.  
2. **Extensión OCR:** se recupera texto de lomos detectados y se discute su calidad con evidencia.  
3. **Extensión matching:** un catálogo demo y similitud de texto alcanzan para una demo de auditoría interpretable, con limitaciones explicitadas.
