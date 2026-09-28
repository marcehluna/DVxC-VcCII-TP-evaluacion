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
| `demo/` | Front local para comparar modelos y `catalog.csv` |
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

Si ya existe `.venv` y no tiene `pip` (por ejemplo, si se creó con `uv`), instalá con `uv pip install --python .venv\Scripts\python.exe -r requirements.txt` en Windows o `uv pip install --python .venv/bin/python -r requirements.txt` en Linux/macOS. Si el entorno ya tiene las dependencias, no hace falta reinstalarlas.

Dataset: ver [`data/README.md`](data/README.md). Alcance: [`concepto/definicion.md`](concepto/definicion.md). Plan: [`concepto/Plan.md`](concepto/Plan.md).

## Demo: comparar modelos entrenados

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

1. Consultá **Métricas registradas — antes de ejecutar**. Elegí **Validación** o **Test**, y **Cajas** o **Máscaras** para ver mAP, precisión, recall y latencia de las corridas documentadas. Esta sección funciona sin cargar pesos. El botón **Actualizar métricas desde los informes** vuelve a leer los archivos locales.
2. Revisá el aviso de checkpoints. Los pesos no se versionan: copiá los archivos `.pt` a las rutas de la tabla de abajo o abrí **Rutas de checkpoints (editables)**. Se aceptan rutas absolutas, rutas relativas a la raíz del repo y rutas entre comillas copiadas desde Windows.
3. Si agregaste o moviste pesos con la app abierta, pulsá **Actualizar rutas y comprobar pesos**. No hace falta reiniciar. El aviso comprueba que el archivo exista; la compatibilidad del checkpoint se verifica al ejecutarlo.
4. En **Modelos a ejecutar**, marcá uno para una prueba individual o varios para comparar. Al abrir la app se seleccionan los archivos disponibles. Los botones **Seleccionar disponibles** y **Seleccionar todos** facilitan cambiar la selección; actualizar rutas no modifica tu elección. Los modelos desmarcados no se cargan ni se ejecutan.
5. Subí una foto de estantería (hasta 20 MB), pegá una imagen, usá la cámara o elegí un ejemplo de validación. Los ejemplos solo aparecen si existe `dataset/valid/images`.
6. Ajustá **Confianza mínima** (valor inicial: `0.50`) y pulsá **Ejecutar selección**. El botón se habilita cuando termina de cargarse la imagen y hay al menos un modelo elegido. La app usa las rutas actuales de los campos. Los modelos elegidos se ejecutan secuencialmente y sus resultados aparecen a medida que terminan. CUDA se usa si está disponible; en caso contrario, CPU.
7. Compará las imágenes en la galería, que permite ampliarlas y descargarlas, y la tabla **Resumen**: número de lomos, confianza media, tiempo local y estado de cada modelo. YOLOv8 AABB y Faster R-CNN muestran cajas; YOLO26 segmentación y Mask R-CNN muestran máscaras y cajas. Si un modelo falla o le faltan pesos, muestra su error y continúan los demás.

Cambiar la imagen, la confianza, la selección o las rutas limpia los resultados anteriores y cancela la publicación de resultados de esa ejecución. Una inferencia ya iniciada puede tardar en terminar. Ejecutá nuevamente para obtener resultados con la nueva configuración. La primera carga de un modelo puede demorar más; los pesos se conservan en memoria para las siguientes ejecuciones.

### Cómo interpretar las métricas

- La tabla histórica cita su fuente en cada fila: `entrenamientos/*.md` para YOLOv8, Faster R-CNN y YOLO26; `results/tables/mask_rcnn_resumen_*.json` para Mask R-CNN. Si una fuente falta o no se puede leer, la app lo informa sin impedir el uso de los modelos.
- Las métricas corresponden al dataset **Book Spine 2 v4** y a los checkpoints documentados. Cambiar la ruta de un modelo no recalcula la tabla histórica. Los protocolos pueden diferir entre informes; consultá la fuente antes de comparar.
- Cajas y máscaras se consultan por separado. `—` significa que no hay un dato disponible: por ejemplo, las tablas globales de YOLO26 no informan precisión y recall de máscara. Los valores entre 0 y 1 se conservan como proporciones; `~` indica una latencia aproximada.
- El tiempo local de una ejecución no incluye la carga de pesos ni el dibujo, puede incluir preparación inicial y depende del hardware y del modelo. No reemplaza las latencias de las corridas en Tesla T4 ni constituye un benchmark uniforme entre arquitecturas.
- La confianza media de una foto no es una medida de exactitud. Una imagen nueva sin etiquetas no permite calcular mAP, precisión o recall.

La app busca los checkpoints documentados en:

| Modelo | Ruta |
| :--- | :--- |
| YOLOv8 AABB | `results/checkpoints/yolov8/mejor_map50.pt` |
| Faster R-CNN | `results/checkpoints/faster_rcnn/mejor_map50.pt` |
| YOLO26 segmentación | `results/checkpoints/yolo26s_seg/mejor_map50.pt` |
| Mask R-CNN | `results/checkpoints/mask_rcnn/mejor_mask_map50.pt` |

Si falta la ruta principal de Mask R-CNN, la app también busca `results/checkpoints/faster_rcnn/mejor_mask_map50.pt` para admitir el checkpoint ya generado. Como valores iniciales alternativos se admiten las variables de entorno `DEMO_YOLOV8_WEIGHTS`, `DEMO_FASTER_RCNN_WEIGHTS`, `DEMO_YOLO26_SEG_WEIGHTS` y `DEMO_MASK_RCNN_WEIGHTS`. Para Faster R-CNN y Mask R-CNN usá checkpoints con la clave `modelo` y la arquitectura correspondiente. Los checkpoints deben provenir de una fuente confiable.

OCR y matching del catálogo todavía son etapas pendientes en `src/`; esta demo compara visualmente los modelos disponibles. Los tiempos medidos localmente dependen del hardware y no sustituyen las métricas de validación de `entrenamientos/`.

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

La prueba de navegador requiere **Microsoft Edge**, el paquete opcional **Playwright**, el checkpoint de Mask R-CNN y las imágenes de validación:

```powershell
uv pip install --python .venv\Scripts\python.exe playwright
$env:DEMO_TEST_BROWSER = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_browser.py -v
Remove-Item Env:DEMO_TEST_BROWSER
```

Esta prueba abre Edge sin ventana, consulta métricas, ejecuta Mask R-CNN, comprueba un peso faltante, verifica que la miniatura entre completa en la galería y revisa el contraste del aviso en tema oscuro. Guarda capturas en `.test-artifacts/` (no se versionan). Si usás `pip`, podés instalar Playwright con `.\.venv\Scripts\python.exe -m pip install playwright`.

Verificación local del 28/09/2026: **13 pruebas de lógica aprobadas** y **1 prueba de navegador aprobada**, incluyendo una inferencia real de Mask R-CNN, actualización de métricas, selección individual, limpieza de resultados, manejo de pesos faltantes y contraste en modo oscuro. Las inferencias reales de YOLOv8, Faster R-CNN y YOLO26 quedan pendientes de disponer de sus checkpoints locales. Las dependencias emitieron avisos de deprecación de Gradio/pandas y avisos de recursos de asyncio; no impidieron completar las pruebas.
