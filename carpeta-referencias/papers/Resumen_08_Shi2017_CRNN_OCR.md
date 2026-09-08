# Resumen — An End-to-End Trainable Neural Network for Image-Based Sequence Recognition (CRNN)

**Archivo:** `08_Shi2017_CRNN_OCR.pdf`  
**Cita:** Shi, B.; Bai, X.; Yao, C. *IEEE TPAMI* 2017 (arXiv:1507.05717). https://arxiv.org/abs/1507.05717

## Problema

Reconocer texto en imagen (scene text / sequence recognition) solía partirse en detección de caracteres + clasificación, con pipelines frágiles. Se necesitaba un modelo **end-to-end** que lean secuencias de longitud variable sin segmentar caracteres a mano.

## Aporte

Arquitectura **CRNN**: CNN (features) + RNN bidireccional (contexto de secuencia) + **CTC** (alineación con la etiqueta sin recortar caracteres). Entrenamiento extremo a extremo.

## Método (ideas clave)

1. CNN extrae un mapa de features → secuencia de columnas (“Map-to-Sequence”).  
2. BiLSTM modela dependencias izquierda-derecha.  
3. CTC permite entrenar con strings completos pese a longitudes distintas.  
4. Modelo compacto y rápido respecto a pipelines clásicos de muchas etapas.

## Resultados destacados

Supera o iguala métodos previos en benchmarks de scene text recognition de la época; muestra robustez y generalidad (incluso sin léxico cerrado en algunos settings).

## Relevancia para nuestro TP

Base teórica de muchos OCR modernos (incl. componentes tipo secuencia en PaddleOCR/EasyOCR). Para la Etapa B podemos citar CRNN al explicar *por qué* el OCR sobre el recorte del lomo es un problema de **secuencia**, y qué errores (CER, confusiones de caracteres) esperamos sin necesidad de reimplementar CRNN.
