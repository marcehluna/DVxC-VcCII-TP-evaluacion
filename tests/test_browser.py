"""Prueba opcional con Edge: DEMO_TEST_BROWSER=1 y paquete playwright instalado."""

import os
from contextlib import ExitStack
from pathlib import Path
import socket
import re
import unittest


@unittest.skipUnless(os.getenv("DEMO_TEST_BROWSER") == "1", "Activar DEMO_TEST_BROWSER=1 para probar Edge")
class BrowserTest(unittest.TestCase):
    def test_yolo26_inference_ocr_evidence_and_dark_contrast(self):
        from playwright.sync_api import sync_playwright
        from demo.app import build_app
        from demo.inference import MODELS, ROOT

        artifacts = ROOT / ".test-artifacts"
        artifacts.mkdir(exist_ok=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        app = build_app()
        app.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                   inbrowser=False, quiet=True)
        self.addCleanup(app.close)
        with sync_playwright() as playwright, ExitStack() as cleanup:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            cleanup.callback(browser.close)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            cleanup.callback(page.screenshot, path=str(artifacts / "last-browser-state.png"), full_page=True)
            page.goto(f"http://127.0.0.1:{port}/?__theme=light")
            page.get_by_text("Ver evidencia comparativa de segmentación", exact=True).click()
            page.get_by_role("radio", name="Test", exact=True).check()
            page.locator('#historical-metrics span[data-editable="false"]:visible').filter(
                has_text=re.compile(r"^0\.776$")).first.wait_for()
            page.get_by_text("Resultados del experimento", exact=True).wait_for()
            page.get_by_text("8/19", exact=True).wait_for()
            page.screenshot(path=str(artifacts / "demo-light.png"), full_page=True)

            # El checkpoint elegido permite verificar el recorrido principal en CPU/GPU.
            yolo26 = next(spec for spec in MODELS if spec.name == "YOLO26 segmentación")
            self.assertTrue(yolo26.path.is_file(), "Falta el checkpoint real de YOLO26")
            sample = next((ROOT / "dataset/valid/images").glob("*.jpg"))
            page.locator('#source-image input[type="file"]').first.set_input_files(str(sample))
            page.locator('#max-ocr-books input[type="range"]').evaluate(
                "el => { el.value = '1'; el.dispatchEvent(new Event('input', {bubbles: true})); "
                "el.dispatchEvent(new Event('change', {bubbles: true})); }")
            page.get_by_role("button", name="Analizar estante", exact=True).click()
            page.get_by_text("YOLO26 finalizó:", exact=False).wait_for(timeout=180000)
            page.get_by_text("OCR finalizado: 1 lomos leídos", exact=False).wait_for(timeout=240000)
            page.get_by_text("YOLO26 ·", exact=False).first.wait_for()
            sizes = page.locator("#result-gallery").evaluate(
                "el => ({height: el.getBoundingClientRect().height, "
                "imageHeight: el.querySelector('img').getBoundingClientRect().height})")
            self.assertLessEqual(sizes["imageHeight"], sizes["height"])
            page.screenshot(path=str(artifacts / "demo-result.png"), full_page=True)

            # Cambiar el checkpoint limpia resultados y un peso ausente no rompe la app.
            page.get_by_text("Cambiar checkpoint", exact=True).click()
            page.get_by_role("textbox", name="Checkpoint de YOLO26s-seg", exact=True).fill(
                str(artifacts / "missing.pt"))
            page.get_by_text("La imagen o configuración cambió.", exact=False).wait_for()
            page.get_by_role("button", name="Analizar estante", exact=True).click()
            page.get_by_text("No se pudo ejecutar YOLO26.", exact=False).wait_for()

            page.goto(f"http://127.0.0.1:{port}/?__theme=dark")
            page.get_by_text("Cambiar checkpoint", exact=True).click()
            page.get_by_role("textbox", name="Checkpoint de YOLO26s-seg", exact=True).fill(
                str(artifacts / "missing.pt"))
            page.get_by_role("button", name="Comprobar checkpoint", exact=True).click()
            alert = page.get_by_role("alert").filter(has_text="Falta el checkpoint").first
            alert.wait_for()
            colors = alert.evaluate("el => ({background: getComputedStyle(el).backgroundColor, "
                                    "texts: [...el.querySelectorAll('span')].map(x => getComputedStyle(x).color)})")
            self.assertEqual(colors["background"], "rgb(255, 241, 240)")
            self.assertTrue(all(color == "rgb(122, 23, 14)" for color in colors["texts"]))
            page.screenshot(path=str(artifacts / "demo-dark.png"), full_page=True)


if __name__ == "__main__":
    unittest.main()
