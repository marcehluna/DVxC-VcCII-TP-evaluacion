# DVxC-VcCII-TP-evaluacion

Repo del TP de evaluación de Visión por Computadora II (CEIA).

**Tema:** auditoría de inventario bibliográfico (detección de lomos + OCR + matching simple).

## Estructura

| Carpeta | Uso |
| :--- | :--- |
| `concepto/` | Definición y plan del TP |
| `dataset/` | Book Spine 2 (no se versiona; descarga idempotente) |
| `data/` | Instrucciones y versión/splits congelados |
| `src/` | Scripts de train / eval / OCR / matching |
| `notebooks/` | Experimentos Colab |
| `results/` | Tablas, figuras y predicciones |
| `demo/` | Front local del pipeline YOLO26s-seg → OCR → catálogo |
| `docs/` | Notas de experimento |
| `carpeta-referencias/` | Pautas, papers y template IEEE |

## Preparar el entorno

El proyecto requiere Python 3.12 o posterior. Desde la raíz del repo, creá el entorno e instalá las dependencias.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Si ya existe `.venv` y no tiene `pip` (por ejemplo, si se creó con `uv`), instalá con `uv pip install --link-mode copy --python .venv\Scripts\python.exe -r requirements.txt` en Windows o `uv pip install --python .venv/bin/python -r requirements.txt` en Linux/macOS. El modo `copy` evita errores de hardlinks cuando el repo está dentro de OneDrive. Si el entorno ya tiene las dependencias, no hace falta reinstalarlas.

Dataset: ver [`data/README.md`](data/README.md). Alcance: [`concepto/definicion.md`](concepto/definicion.md). Plan: [`concepto/Plan.md`](concepto/Plan.md).

## Demo: inventario visual de biblioteca

La interfaz local usa **Gradio**. Para levantar el front desde la raíz del repo:

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe -m demo.app
```

**Linux / macOS:**

```bash
./.venv/bin/python -m demo.app
```

Abrí la dirección que imprime la terminal (normalmente `http://127.0.0.1:7860`). Para detener la app, presioná `Ctrl+C` en esa terminal.

### Cómo usarla

1. Revisá la ficha de **YOLO26s-seg**, elegido por su balance entre calidad de máscara, velocidad y utilidad del recorte para OCR. La comparación contra Mask R-CNN queda disponible como evidencia secundaria.
2. Subí una foto de estantería (hasta 20 MB), pegá una imagen, usá la cámara o elegí un ejemplo de validación.
3. Ajustá **Confianza mínima** (valor inicial: `0.50`) y el máximo de lomos que querés leer con OCR. Pulsá **Analizar estante**.
4. Revisá las máscaras y, a medida que termina cada recorte, la sección **OCR en vivo**. Allí aparecen el texto, la confianza, la rotación elegida y la coincidencia contra el catálogo.
5. Consultá **Evidencia OCR** para contrastar la nueva ejecución con los 19 recortes previamente procesados por la notebook.

La app busca YOLO26 primero en `results/checkpoints/yolo26s_seg/mejor_map50.pt` y, como ruta alternativa, en `pesos/yolo26s_seg/mejor_map50.pt`. También admite `DEMO_YOLO26_SEG_WEIGHTS` o una ruta editada desde la interfaz.

### OCR y catálogo

La demo ejecuta en vivo YOLO26s-seg → recorte orientado → CLAHE → PaddleOCR → matching con RapidFuzz. La implementación reutilizable está en `demo/ocr.py`; `notebooks/OCR.ipynb` no se modifica ni se importa. Como en el experimento, cada recorte se prueba en 0°/90°/270° y se conserva la lectura de mayor confianza. Los modelos móviles `PP-OCRv3_mobile_det` y `latin_PP-OCRv3_mobile_rec`, seleccionados por PaddleOCR para español en la notebook, se descargan en `.cache/paddlex/` durante la primera ejecución y se conservan para las siguientes.

Como referencia experimental, la notebook procesó 19 recortes de las dos fotos de `fotos_propias/`. PaddleOCR obtuvo una confianza media de `0.949`, una latencia media registrada de `649 ms` y 8 coincidencias sobre 19 con umbral de similitud 85 contra los 29 títulos de `catalogo/catalogo.xlsx`.

Los CSV de `resultados_ocr/` alimentan solamente la sección de evidencia histórica; la tabla **OCR en vivo** se calcula desde la fotografía recién cargada.

### Testing del front

Las pruebas de lógica, callbacks y lectura de informes usan `unittest`, incluido en Python, y las dependencias de la app. Desde PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Cubren ejecución selectiva, rutas editadas, pesos faltantes o corruptos, limpieza de resultados, orientación EXIF y separación de métricas de cajas/máscaras y validación/test. No descargan checkpoints. Las pruebas de navegador e inferencia real se omiten por defecto.

Para probar una inferencia real de Mask R-CNN en CPU, con su checkpoint y el dataset de validación presentes:

```powershell
$env:DEMO_TEST_REAL = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_demo.py -v
Remove-Item Env:DEMO_TEST_REAL
```

La prueba de navegador requiere **Microsoft Edge**, el paquete opcional **Playwright**, el checkpoint de YOLO26s-seg y las imágenes de validación:

```powershell
uv pip install --python .venv\Scripts\python.exe playwright
$env:DEMO_TEST_BROWSER = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_browser.py -v
Remove-Item Env:DEMO_TEST_BROWSER
```

Esta prueba abre Edge sin ventana, consulta las métricas y la evidencia OCR, ejecuta YOLO26s-seg más un recorte OCR real, comprueba un peso faltante, verifica las galerías y revisa el contraste del aviso en tema oscuro. Guarda capturas en `.test-artifacts/` (no se versionan). Si usás `pip`, podés instalar Playwright con `.\.venv\Scripts\python.exe -m pip install playwright`.

Verificación local del 09/10/2026: pruebas de lógica aprobadas y recorrido de navegador validado con inferencia real de YOLO26s-seg, evidencia OCR, limpieza de resultados, manejo de pesos faltantes y contraste en modo oscuro. Las dependencias pueden emitir avisos de deprecación de Gradio/pandas y de recursos de asyncio; no impiden completar las pruebas.
