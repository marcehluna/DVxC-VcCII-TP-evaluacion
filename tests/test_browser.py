"""Prueba opcional con Edge: DEMO_TEST_BROWSER=1 y paquete playwright instalado."""

import os
from contextlib import ExitStack
from pathlib import Path
import socket
import re
import unittest


@unittest.skipUnless(os.getenv("DEMO_TEST_BROWSER") == "1", "Activar DEMO_TEST_BROWSER=1 para probar Edge")
class BrowserTest(unittest.TestCase):
    def test_metrics_selection_real_inference_and_dark_contrast(self):
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
            page.get_by_role("radio", name="Máscaras", exact=True).check()
            page.locator('#historical-metrics span[data-editable="false"]:visible').filter(
                has_text=re.compile(r"^0\.785$")).first.wait_for()
            page.get_by_role("radio", name="Test", exact=True).check()
            page.locator('#historical-metrics span[data-editable="false"]:visible').filter(
                has_text=re.compile(r"^0\.776$")).first.wait_for()
            page.screenshot(path=str(artifacts / "demo-light.png"), full_page=True)

            # El checkpoint real disponible permite verificar todo el recorrido en CPU/GPU.
            self.assertTrue(MODELS[3].path.is_file(), "Falta el checkpoint real de Mask R-CNN")
            for spec in MODELS:
                page.get_by_role("checkbox", name=spec.name, exact=True).uncheck()
            page.get_by_role("checkbox", name="Mask R-CNN", exact=True).check()
            sample = next((ROOT / "dataset/valid/images").glob("*.jpg"))
            page.locator('#source-image input[type="file"]').first.set_input_files(str(sample))
            page.get_by_role("button", name="Ejecutar selección", exact=True).click()
            completion = page.get_by_text("Finalizado:", exact=False)
            completion.wait_for(timeout=180000)
            self.assertIn("Finalizado: 1/1 modelos ejecutados correctamente.", completion.inner_text())
            page.get_by_text("Mask R-CNN ·", exact=False).first.wait_for()
            sizes = page.locator("#result-gallery").evaluate(
                "el => ({height: el.getBoundingClientRect().height, "
                "imageHeight: el.querySelector('img').getBoundingClientRect().height})")
            self.assertLessEqual(sizes["imageHeight"], sizes["height"])
            page.screenshot(path=str(artifacts / "demo-result.png"), full_page=True)

            # Cambiar la selección limpia resultados y un peso ausente no rompe la app.
            page.get_by_role("checkbox", name="Mask R-CNN", exact=True).uncheck()
            page.get_by_text("La imagen o configuración cambió.", exact=False).wait_for()
            page.get_by_text("Rutas de checkpoints (editables)", exact=True).click()
            page.get_by_role("textbox", name="YOLOv8 AABB", exact=True).fill(str(artifacts / "missing.pt"))
            page.get_by_role("checkbox", name="YOLOv8 AABB", exact=True).check()
            page.get_by_role("button", name="Ejecutar selección", exact=True).click()
            page.get_by_text("Finalizado: 0/1 modelos ejecutados correctamente.", exact=False).wait_for()

            page.goto(f"http://127.0.0.1:{port}/?__theme=dark")
            alert = page.get_by_role("alert").filter(has_text="Faltan").first
            alert.wait_for()
            colors = alert.evaluate("el => ({background: getComputedStyle(el).backgroundColor, "
                                    "texts: [...el.querySelectorAll('span')].map(x => getComputedStyle(x).color)})")
            self.assertEqual(colors["background"], "rgb(255, 241, 240)")
            self.assertTrue(all(color == "rgb(122, 23, 14)" for color in colors["texts"]))
            page.screenshot(path=str(artifacts / "demo-dark.png"), full_page=True)


if __name__ == "__main__":
    unittest.main()
