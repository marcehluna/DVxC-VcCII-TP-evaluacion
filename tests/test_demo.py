"""Pruebas sin descargar pesos: python -m unittest discover -s tests -v."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from demo import inference
from demo.app import (checkpoint_status, choose_available, clear_results, inputs_changed,
                      ocr_experiment, primary_checkpoint_status, run_comparison)
from demo.metrics import metric_rows


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [str(self.root / f"{i}.pt") for i in range(len(inference.MODELS))]
        self.image = Image.new("RGB", (80, 60), "white")

    def test_only_selected_model_runs_with_its_own_path(self):
        Path(self.paths[3]).touch()
        with patch.object(inference, "device_name", return_value="cpu"), \
             patch.object(inference, "predict_one") as predict:
            inference.predict_all(self.image, .5, self.paths, [inference.MODELS[3].name])
        predict.assert_called_once()
        self.assertEqual(predict.call_args.args[1], inference.MODELS[3])
        self.assertEqual(predict.call_args.args[4], self.paths[3])

    def test_unselected_existing_weights_do_not_trigger_device_import(self):
        Path(self.paths[3]).touch()
        with patch.object(inference, "device_name", side_effect=AssertionError("No cargar torch")):
            result = inference.predict_all(self.image, .5, self.paths, [inference.MODELS[0].name])
        self.assertEqual(len(result), 1)
        self.assertIn("Falta el checkpoint", result[0].message)

    def test_invalid_inputs_are_rejected(self):
        for names in ([], ["desconocido"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                inference.predict_all(self.image, .5, self.paths, names)
        for confidence in (-1, 2, float("nan")):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                inference.predict_all(self.image, confidence, self.paths)
        with self.assertRaises(ValueError):
            inference.predict_all(None, .5, self.paths)

    def test_corrupt_weights_are_isolated_per_model(self):
        for path in self.paths:
            Path(path).touch()
        with patch.object(inference, "device_name", return_value="cpu"), \
             patch.object(inference, "_load_model", side_effect=RuntimeError("checkpoint inválido")):
            predictions = inference.predict_all(self.image, .5, self.paths)
        self.assertEqual(len(predictions), 4)
        self.assertTrue(all("checkpoint inválido" in p.message for p in predictions))

    def test_status_refresh_detects_new_weights_and_escapes_paths(self):
        self.assertIn("Faltan 4 de 4", checkpoint_status(*self.paths))
        Path(self.paths[2]).touch()
        self.assertIn("Faltan 3 de 4", checkpoint_status(*self.paths))
        self.assertEqual(choose_available(*self.paths), [inference.MODELS[2].name])
        html = checkpoint_status(*([str(self.root / '<script>.pt')] * 4))
        self.assertNotIn("<script>", html)
        self.assertIn("color:#7a170e!important", html)

    def test_paths_support_windows_copy_as_path_and_relative_paths(self):
        spec = inference.MODELS[0]
        self.assertEqual(spec.resolve_path(f'  "{self.paths[0]}"  '), Path(self.paths[0]))
        self.assertEqual(spec.resolve_path("results/custom.pt"), inference.ROOT / "results/custom.pt")

    def test_streaming_results_and_errors_do_not_leave_previous_images(self):
        successful = inference.Prediction(inference.MODELS[0], self.image, 0, None, 12.34, "OK")
        failed = inference.Prediction(inference.MODELS[1], None, message="Falta el checkpoint")
        names = [p.spec.name for p in (successful, failed)]
        with patch("demo.app.iter_predictions", return_value=iter([successful, failed])):
            frames = list(run_comparison(self.image, .5, names, *self.paths))
        self.assertEqual(frames[0][0], [])
        self.assertEqual(len(frames[-1][0]), 1)
        self.assertEqual(len(frames[-1][1]), 2)
        self.assertIsNone(frames[-1][1][0][2])  # Sin detecciones: no hay confianza media.
        self.assertIn("1/2", frames[-1][2])
        self.assertEqual(clear_results()[:2], ([], []))

    def test_exif_orientation_is_applied(self):
        self.image.getexif()[274] = 6
        with patch.object(inference, "predict_one") as predict:
            inference.predict_all(self.image, .5, self.paths, [inference.MODELS[0].name])
        self.assertEqual(predict.call_args.args[0].size, (60, 80))

    def test_run_button_requires_uploaded_image_and_selection(self):
        self.assertFalse(inputs_changed(None, ["Mask R-CNN"])[-1].interactive)
        self.assertFalse(inputs_changed(self.image, [])[-1].interactive)
        self.assertTrue(inputs_changed(self.image, ["Mask R-CNN"])[-1].interactive)

    def test_primary_status_uses_yolo26_checkpoint(self):
        self.assertIn("Falta el checkpoint de YOLO26", primary_checkpoint_status(self.paths[2]))
        Path(self.paths[2]).touch()
        self.assertIn("YOLO26 listo", primary_checkpoint_status(self.paths[2]))

    def test_ocr_artifacts_are_summarized(self):
        output = self.root / "resultados_ocr"
        crops = output / "recortes"
        crops.mkdir(parents=True)
        (output / "comparativa_ocr.csv").write_text(
            "crop_id,paddleocr_confianza,paddleocr_latencia_ms\n"
            "book_001,0.9,500\nbook_002,1.0,700\n", encoding="utf-8")
        (output / "comparativa_catalogo.csv").write_text(
            "crop_id,paddleocr_texto,paddleocr_match_titulo,paddleocr_match_score,paddleocr_coincide\n"
            "book_001,texto,EL PRINCIPITO,90,True\n"
            "book_002,otro,SIN MATCH,20,False\n", encoding="utf-8")
        (crops / "book_001.jpg").touch()

        rows, stats, gallery = ocr_experiment(self.root)

        self.assertEqual(stats, {"crops": "2", "confidence": "0.950", "matches": "1/2",
                                 "latency": "600 ms"})
        self.assertEqual(rows[0][-1], "90.0%")
        self.assertEqual(len(gallery), 1)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "entrenamientos").mkdir()
        (self.root / "results/tables").mkdir(parents=True)
        (self.root / "entrenamientos/YOLO26s-seg.md").write_text(
            '#### Validación\n| Métrica | Valor |\n'
            '| *mAP@0.5* (máscara) | 0,761 |\n| *mAP@0.5* (caja, puente) | 0,804 |\n'
            '| Precisión (caja, puente) | 0,990 |\n| Latencia (ms/imagen) | ~18 |\n'
            '#### Test\n| Métrica | Validación | Test |\n'
            '| *mAP@0.5* (máscara) | 0,761 | 0,757 |\n'
            '| *mAP@0.5* (caja, puente) | 0,804 | 0,814 |\n', encoding="utf-8")

    def test_mask_metrics_never_use_box_precision(self):
        rows, _ = metric_rows("Validación", "Máscaras", self.root)
        self.assertEqual(rows[0][2], "0.761")
        self.assertEqual(rows[0][4], "—")
        self.assertEqual(rows[0][6], "~18")
        self.assertEqual(len(rows), 2)

    def test_test_partition_uses_test_column(self):
        rows, _ = metric_rows("Test", "Máscaras", self.root)
        self.assertEqual(rows[0][2], "0.757")

    def test_json_global_and_flat_test_schema(self):
        values = {"mask_map50": .8, "box_map50": .9, "latencia_ms": 110}
        for partition, suffix, results in (("Validación", "validacion", {"global": values}),
                                            ("Test", "test", values)):
            path = self.root / f"results/tables/mask_rcnn_resumen_{suffix}.json"
            path.write_text(json.dumps({"resultados": results}), encoding="utf-8")
            rows, _ = metric_rows(partition, "Máscaras", self.root)
            self.assertEqual(rows[-1][2], "0.800")
            self.assertEqual(rows[-1][4], "—")

    def test_missing_or_invalid_reports_show_missing_values(self):
        (self.root / "results/tables/mask_rcnn_resumen_test.json").write_text("{", encoding="utf-8")
        rows, note = metric_rows("Test", "Cajas", self.root)
        self.assertEqual(rows[-1][2], "—")
        self.assertIn("Fuentes faltantes", note)


@unittest.skipUnless(os.getenv("DEMO_TEST_REAL") == "1", "Activar DEMO_TEST_REAL=1 para inferencia real")
class RealCheckpointTest(unittest.TestCase):
    def test_mask_rcnn_on_validation_image(self):
        spec = inference.MODELS[3]
        self.assertTrue(spec.path.is_file(), "Falta el checkpoint de Mask R-CNN")
        sample = next((inference.ROOT / "dataset/valid/images").glob("*.jpg"))
        with Image.open(sample) as opened:
            image = opened.convert("RGB")
        result = inference.predict_one(image, spec, .5, "cpu")
        self.assertEqual(result.message, "OK", result.message)
        self.assertEqual(result.image.size, image.size)
        self.assertGreater(result.elapsed_ms, 0)
        print(f"\nMask R-CNN real: {result.count} lomos, {result.elapsed_ms:.1f} ms")


if __name__ == "__main__":
    unittest.main()
