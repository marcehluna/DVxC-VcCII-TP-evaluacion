"""Pruebas del pipeline OCR en vivo sin cargar modelos externos."""

from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from demo import ocr
from demo.app import run_library_pipeline
from demo.inference import MODELS, Prediction


class OCRTests(unittest.TestCase):
    def test_uses_the_same_paddle_models_as_the_notebook(self):
        self.assertEqual(ocr.PADDLE_DETECTION_MODEL, "PP-OCRv3_mobile_det")
        self.assertEqual(ocr.PADDLE_RECOGNITION_MODEL, "latin_PP-OCRv3_mobile_rec")

    def test_polygon_crop_is_oriented_and_preprocessed(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[20:100, 60:90] = (220, 220, 220)
        polygon = [[60, 20], [90, 20], [90, 100], [60, 100]]

        crop = ocr.crop_from_polygon(image, polygon)
        processed = ocr.preprocess_crop(crop)

        self.assertGreater(crop.size, 0)
        self.assertGreaterEqual(processed.shape[1], 80)
        self.assertEqual(processed.shape[2], 3)

    def test_catalog_matching_applies_threshold(self):
        entries = (
            ocr.CatalogEntry("Desde el ojo del pez", "Pablo De Santis"),
            ocr.CatalogEntry("El príncipe", "Maquiavelo"),
        )
        title, author, score = ocr.match_catalog(
            "DESDE EL OJO DEL PEZ Pablo De Santis", entries, threshold=85)
        self.assertEqual(title, "Desde el ojo del pez")
        self.assertEqual(author, "Pablo De Santis")
        self.assertGreaterEqual(score, 85)

        title, author, score = ocr.match_catalog("texto sin relación", entries, threshold=85)
        self.assertIsNone(title)
        self.assertIsNone(author)
        self.assertLess(score, 85)

    def test_all_notebook_rotations_are_evaluated(self):
        image = np.zeros((20, 80, 3), dtype=np.uint8)
        readings = [("un título", .95), ("otro", .55), ("", 0.0)]
        with patch.object(ocr, "read_paddle", side_effect=readings) as reader:
            text, confidence, rotation, _, _ = ocr.read_best_rotation(image)
        self.assertEqual((text, confidence, rotation), ("un título", .95, 0))
        self.assertEqual(reader.call_count, 3)

    def test_live_pipeline_streams_ocr_rows(self):
        image = Image.new("RGB", (100, 80), "white")
        prediction = Prediction(
            MODELS[2], image, 1, .9, 12.0, "OK",
            polygons=[[[10, 10], [30, 10], [30, 70], [10, 70]]],
            scores=[.9],
        )
        reading = ocr.OCRReading(
            1, image, .9, "El príncipe", .96, 90, 120.0,
            "EL PRINCIPE", "Maquiavelo", 100.0,
        )
        paths = [str(Path("missing") / f"{index}.pt") for index in range(len(MODELS))]
        with patch("demo.app.iter_predictions", return_value=iter([prediction])), \
             patch("demo.app.iter_readings", return_value=iter([reading])):
            frames = list(run_library_pipeline(image, .5, 10, *paths))

        self.assertIn("OCR finalizado", frames[-1][6])
        self.assertEqual(frames[-1][5][0][4], "EL PRINCIPE")
        self.assertEqual(len(frames[-1][4]), 1)


if __name__ == "__main__":
    unittest.main()
