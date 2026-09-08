# Resumen — Image-text matching for large-scale book collections

**Archivo:** `03_Ioli2024_ImageTextMatching_BookCollections.pdf`  
**Cita:** Llabrés, A.; Dey, A. U.; Karatzas, D.; Valveny, E. arXiv:2407.19812, 2024. https://arxiv.org/abs/2407.19812  
**Código/datos:** https://github.com/llabres/library-dataset

## Problema

Inventariar una biblioteca a partir de muchas fotos de estanterías implica **detectar** libros y **mapearlos** a un catálogo. Tratar cada libro como retrieval independiente es subóptimo: el problema real es un matching **many-to-many** (conjunto de detecciones ↔ conjunto de títulos), con OCR ruidoso y catálogos incompletos o enormes.

## Aporte

Formulan el inventario como matching many-to-many; publican un dataset de una biblioteca pública en España; proponen un pipeline de dos etapas de matching.

## Método (ideas clave)

1. **SAM** para segmentar lomos; **OCR comercial** para extraer texto.  
2. Matching en dos fases: primero **CLIP** (rápido); luego refinamiento con **Hungarian matching** o un modelo **BERT** robusto a OCR ruidoso / matches parciales.  
3. Evaluación en *closed-set* (~15k títulos del inventario) y *open-set* (~2.3M títulos), más un setting matching-only vs. detección+matching.

## Resultados / hallazgos

Hungarian y BERT superan un baseline de **fuzzy string matching**. El desempeño se degrada cuando el catálogo crece o cuando faltan libros en una de las dos listas (detecciones o inventario). Subrayan límites inherentes del matching a gran escala.

## Relevancia para nuestro TP

Justifica por qué el profesor pide **matching simple** (similitud de texto) como extensión: el matching “serio” (CLIP, BERT, Hungarian many-to-many) es otro problema de investigación. Podemos citar este trabajo en Alcance/Limitaciones: “existen enfoques avanzados; nosotros acotamos a similitud de texto sobre un catálogo demo”.
