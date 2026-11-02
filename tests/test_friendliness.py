"""Input checks, remembered settings, model readiness and the finish summary."""
import importlib
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lowering_factor_validator as validator
import lowering_model
from ems_toolkit_common import SettingsStore

lowerer = importlib.import_module("Predicted_Seabed_lowering_tool_V8")


def _write_layer(path: Path, layer: str, columns: dict, crs="EPSG:27700") -> None:
    count = len(next(iter(columns.values())))
    frame = gpd.GeoDataFrame(
        columns, geometry=[box(i * 250, 0, i * 250 + 250, 250) for i in range(count)], crs=crs
    )
    frame.to_file(path, layer=layer, driver="GPKG")


class InputCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_vector_layer_and_field(self):
        path = self.root / "grid.gpkg"
        _write_layer(path, "other", {"name": ["x"]})
        _write_layer(path, "grid_250m", {"GridID": ["a", "b", "c"]})
        good = validator.check_input("calibration_grid", {"calibration_grid": str(path), "grid_id_field": "GridID"})
        self.assertTrue(good.ok)
        self.assertEqual(good.recommended_layer, "grid_250m")
        self.assertIn("3 cells", good.message)
        wrong_field = validator.check_input(
            "calibration_grid",
            {"calibration_grid": str(path), "calibration_grid_layer": "other", "grid_id_field": "GridID"},
        )
        self.assertFalse(wrong_field.ok)
        self.assertIn("no 'GridID' field", wrong_field.message)
        missing = validator.check_input("licence_layer", {"licence_layer": str(self.root / "no.gpkg")})
        self.assertFalse(missing.ok)

    def test_ems_folder_and_output_folder(self):
        tracks = self.root / "tracks" / "2022" / "vessel"
        tracks.mkdir(parents=True)
        (tracks / "19-01-2022.txt").write_text("x", encoding="utf-8")
        (tracks / "notes.docx").write_text("x", encoding="utf-8")
        result = validator.check_input("ems_source", {"ems_source": str(self.root / "tracks")})
        self.assertTrue(result.ok)
        self.assertIn("1 EMS track files", result.message)
        empty = self.root / "empty"
        empty.mkdir()
        self.assertFalse(validator.check_input("ems_source", {"ems_source": str(empty)}).ok)
        self.assertIn("created", validator.check_input("output_folder", {"output_folder": str(self.root / "new")}).message)
        self.assertFalse(validator.check_input("output_folder", {"output_folder": str(self.root / "a" / "b")}).ok)
        self.assertIsNone(validator.check_input("output_folder", {"output_folder": ""}).ok)

    def test_model_reports_blank_factors(self):
        path = self.root / "model.gpkg"
        _write_layer(path, "areas", {"area_id": ["137", "522"], "lowering_factor": [0.0169, None]})
        result = validator.check_input(
            "model_path", {"model_path": str(path), "model_layer": "areas", "model_id_field": "area_id"}
        )
        self.assertTrue(result.ok)
        self.assertIn("2 areas, 1 with a blank lowering_factor", result.message)


class ModelReadinessTests(unittest.TestCase):
    def _model(self, factors):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.gpkg"
            keys = list(factors)
            gpd.GeoDataFrame(
                {"area_id": keys, "area_name": [f"Area {k}" for k in keys],
                 "lowering_factor": list(factors.values()), "active": [1] * len(keys)},
                geometry=[box(i * 1000, 0, i * 1000 + 250, 250) for i in range(len(keys))],
                crs="EPSG:27700",
            ).to_file(path, layer="areas", driver="GPKG")
            gpd.GeoDataFrame(
                {"grid_id": [f"{k}_1" for k in keys], "area_id": keys},
                geometry=[box(i * 1000, 0, i * 1000 + 250, 250) for i in range(len(keys))],
                crs="EPSG:27700",
            ).to_file(path, layer="grid_250m", driver="GPKG")
            model = lowering_model.load_lowering_model(path)
        return model, lowering_model.validate_lowering_model(model)

    def test_all_blank_blocks_and_says_what_to_do(self):
        level, lines = lowering_model.factor_readiness(*self._model({"137": None, "522": None}))
        self.assertEqual(level, "block")
        self.assertIn("137, 522", lines[0])
        self.assertIn("write-back", lines[1])

    def test_some_blank_warns_and_selected_area_blocks(self):
        model, validation = self._model({"137": 0.0169, "522": None})
        level, lines = lowering_model.factor_readiness(model, validation)
        self.assertEqual(level, "warn")
        self.assertIn("will be skipped: 522", lines[0])
        self.assertEqual(lowering_model.factor_readiness(model, validation, "137")[0], "ok")
        self.assertEqual(lowering_model.factor_readiness(model, validation, "522")[0], "block")

    def test_all_filled_is_ok(self):
        self.assertEqual(lowering_model.factor_readiness(*self._model({"137": 0.0169}))[0], "ok")


class FinishSummaryTests(unittest.TestCase):
    def test_rows_and_totals(self):
        summary = pd.DataFrame([
            {"summary_scope": "area", "area_id": "137", "lowering_factor": 0.016912,
             "areas_successfully_processed": 1, "dredge_hours": 120.5, "volume_m3": 127_000.4,
             "max_lowering_m": 0.4123, "250m_grid_cells_affected": 40, "warnings_errors": None},
            {"summary_scope": "area", "area_id": "522", "lowering_factor": None,
             "areas_successfully_processed": 0, "dredge_hours": 0, "volume_m3": 0,
             "max_lowering_m": 0, "250m_grid_cells_affected": 0,
             "warnings_errors": "lowering_factor must be a finite, non-negative number."},
            {"summary_scope": "run", "area_id": "", "dredge_hours": 120.5, "volume_m3": 127_000.4},
        ])
        rows, totals = lowerer.finish_summary_rows(summary)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][:7], ("137", "Done", "0.016912", "120.5", "127,000", "0.412", "40"))
        self.assertEqual(rows[1][1], "Skipped")
        self.assertEqual(rows[1][2], "")
        self.assertEqual((totals["processed"], totals["skipped"]), (1, 1))
        self.assertAlmostEqual(totals["volume"], 127_000.4)


def _display_available():
    if sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@unittest.skipUnless(_display_available(), "needs a display")
class ValidatorFormTests(unittest.TestCase):
    def test_settings_are_remembered_and_checks_shown(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SettingsStore("unused.json")
            store.path = Path(folder) / "remembered.json"
            app = validator.App(settings_store=store)
            try:
                app.vars["licence_filter"].set("137")
                app.vars["calibration_grid"].set(str(Path(folder) / "missing.gpkg"))
                app.vars["output_folder"].set(str(Path(folder) / "out"))
                deadline = time.time() + 5
                while time.time() < deadline and "Calibration grid: File not found." not in app.input_problems():
                    app.update()
                    time.sleep(0.05)
                self.assertIn("Calibration grid: File not found.", app.input_problems())
                self.assertFalse(app.advanced.winfo_ismapped())
                app._toggle_advanced()
            finally:
                app._on_close()
            again = validator.App(settings_store=store)
            try:
                self.assertEqual(again.vars["licence_filter"].get(), "137")
                self.assertTrue(again._advanced_open)
            finally:
                again.destroy()


if __name__ == "__main__":
    unittest.main()
