# DVxC-VcCII-TP-evaluacion

Repo del TP de evaluación de Visión por Computadora II (CEIA).

**Tema:** auditoría de inventario bibliográfico (detección de lomos + OCR + matching simple).

## Estructura

| Carpeta | Uso |
| :--- | :--- |
| `concepto/` | Definición y plan del TP |
| `data/` | Dataset (raw/processed; no se versiona el contenido pesado) |
| `src/` | Scripts de train / eval / OCR / matching |
| `notebooks/` | Experimentos Colab |
| `results/` | Tablas, figuras y predicciones |
| `demo/` | Pipeline punta a punta y `catalog.csv` |
| `docs/` | Notas de experimento |
| `carpeta-referencias/` | Pautas, papers y template IEEE |

## Setup rápido

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dataset: ver [`data/README.md`](data/README.md). Alcance: [`concepto/definicion.md`](concepto/definicion.md). Plan: [`concepto/Plan.md`](concepto/Plan.md).
