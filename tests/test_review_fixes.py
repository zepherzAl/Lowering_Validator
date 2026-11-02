import collections
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
import pandas as pd
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lowering_factor_validator as validator


class ModelWriteBackSettingTests(unittest.TestCase):
    def test_write_back_is_off_by_default(self):
        settings = validator.parse_cli_settings([])
        self.assertFalse(settings.write_model_factors)

    def test_cli_flag_switches_write_back_on(self):
        settings = validator.parse_cli_settings(["--write-model-factors"])
        self.assertTrue(settings.write_model_factors)

    def test_settings_file_value_must_be_boolean(self):
        settings = validator.Settings(
            difference_raster="d.flt",
            calibration_grid="g.gpkg",
            licence_layer="l.gpkg",
            ems_source="ems",
            output_folder="out",
            write_model_factors="yes",
        )
        with self.assertRaisesRegex(ValueError, "write_model_factors"):
            settings.validate()


class LoggingIntervalTests(unittest.TestCase):
    def test_counter_median(self):
        self.assertEqual(
            validator._counter_median(collections.Counter({60: 5, 120: 2, 3600: 1})),
            60.0,
        )
        self.assertIsNone(validator._counter_median(collections.Counter()))

    def test_warns_when_logging_is_slower_than_sample_interval(self):
        messages = []
        validator.check_ems_logging_interval(120.0, 60, messages.append)
        self.assertTrue(any("WARNING" in line and "2.0x" in line for line in messages))

    def test_no_warning_when_interval_matches(self):
        messages = []
        validator.check_ems_logging_interval(60.0, 60, messages.append)
        self.assertFalse(any("WARNING" in line for line in messages))


class DocumentedSignTableTests(unittest.TestCase):
    """Each README sign-table row should give positive ObsLower_m for lowering."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.proj_data = str(Path(rasterio.__file__).parent / "proj_data")
        self.grid = gpd.GeoDataFrame(
            {"GridID": ["cell"]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:27700"
        )

    def _obs_lower(self, raw_value, convention, sign):
        path = self.root / f"{convention}-{sign}-{raw_value}.tif"
        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            with rasterio.open(
                path, "w", driver="GTiff", width=2, height=2, count=1,
                dtype="float32", crs="EPSG:27700",
                transform=from_origin(0, 2, 1, 1),
            ) as dataset:
                dataset.write(np.full((2, 2), raw_value, dtype=np.float32), 1)
            stats = validator.zonal_change(
                self.grid, str(path), sign, lambda _: None,
                raster_value_convention=convention,
            )
        return stats.iloc[0]["ObsLower_m"]

    def test_readme_sign_table(self):
        # 1 m of lowering expressed as each documented raster type.
        cases = [
            (1.0, "ELEVATION", "positive_is_lowering"),        # elevation, earlier - later
            (-1.0, "ELEVATION", "negative_is_lowering"),       # elevation, later - earlier
            (1.0, "POSITIVE_DEPTH", "negative_is_lowering"),   # depth, later - earlier
            (-1.0, "POSITIVE_DEPTH", "positive_is_lowering"),  # depth, earlier - later
        ]
        for raw, convention, sign in cases:
            with self.subTest(convention=convention, sign=sign):
                self.assertAlmostEqual(self._obs_lower(raw, convention, sign), 1.0)



class BackgroundAndPooledRateTests(unittest.TestCase):
    """Six 250 m cells: A dredged 1 h (1.0 m), B dredged 0.5 h (0.15 m), four undredged (0.1 m)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.proj_data = str(Path(rasterio.__file__).parent / "proj_data")
        self.raster = self.root / "difference.tif"
        values = np.array([[0.1, 0.1, 0.1], [1.0, 0.15, 0.1]], dtype=np.float32)
        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            with rasterio.open(
                self.raster, "w", driver="GTiff", width=3, height=2, count=1,
                dtype="float32", crs="EPSG:27700",
                transform=from_origin(0, 500, 250, 250), nodata=-9999.0,
            ) as dataset:
                dataset.write(values, 1)
        cells = {
            "A": box(0, 0, 250, 250), "B": box(250, 0, 500, 250),
            "C": box(500, 0, 750, 250), "D": box(0, 250, 250, 500),
            "E": box(250, 250, 500, 500), "F": box(500, 250, 750, 500),
        }
        self.grid = gpd.GeoDataFrame(
            {"GridID": list(cells)}, geometry=list(cells.values()), crs="EPSG:27700"
        )
        self.licences = gpd.GeoDataFrame(
            {"LicenceNo": ["L1"]}, geometry=[box(0, 0, 750, 500)], crs="EPSG:27700"
        )
        points = [Point(125, 125)] * 60 + [Point(375, 125)] * 30
        self.ems = gpd.GeoDataFrame(
            {"_datetime": [pd.Timestamp("2024-01-01")] * len(points)},
            geometry=points, crs="EPSG:27700",
        )

    def _run(self, **overrides):
        output = self.root / ("-".join(f"{k}{v}" for k, v in overrides.items()) or "default")
        options = {"background_min_cells": 4, **overrides}
        settings = validator.Settings(
            difference_raster=str(self.raster), calibration_grid="g.gpkg",
            licence_layer="l.gpkg", ems_source="ems.csv", output_folder=str(output),
            minimum_ems_hours=0.5, minimum_coverage_percent=80.0,
            minimum_detectable_change_m=0.2, maximum_rate_tph=1e9,
            sample_interval_minutes=1.0, bulk_density=1.0, **options,
        )
        messages = []
        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            summary = validator._run_validation_single(
                settings, messages.append, grid_override=self.grid,
                licences_override=self.licences, ems_override=self.ems,
                difference_raster_override=str(self.raster),
            )
        return summary.iloc[0], "\n".join(messages)

    def test_background_is_removed_and_light_cells_stay_in_pooled_rate(self):
        row, log = self._run()
        self.assertAlmostEqual(row["Background_m"], 0.1, places=5)
        self.assertEqual(row["BackgroundCells"], 4)
        # (0.9 + 0.05) m x 62,500 m2 over 1.5 h
        self.assertAlmostEqual(row["ValidatedRate_m3ph"], (56_250 + 3_125) / 1.5, places=0)
        self.assertEqual(row["AcceptedCells"], 2)
        self.assertAlmostEqual(row["UncorrectedRate_tph"], (62_500 + 9_375) / 1.5, places=0)
        self.assertIn("Background lowering for licence L1: 0.100 m", log)

    def test_cell_filters_mode_rejects_low_change_cell(self):
        row, _ = self._run(rate_method="CELL_FILTERS")
        self.assertEqual(row["AcceptedCells"], 1)
        self.assertAlmostEqual(row["ValidatedRate_m3ph"], 56_250, places=0)

    def test_background_correction_can_be_switched_off(self):
        row, _ = self._run(background_correction="NONE")
        self.assertEqual(row["Background_m"], 0.0)
        self.assertAlmostEqual(row["ValidatedRate_m3ph"], (62_500 + 9_375) / 1.5, places=0)

    def test_too_few_undredged_cells_skips_correction_with_warning(self):
        row, log = self._run(background_min_cells=10)
        self.assertEqual(row["Background_m"], 0.0)
        self.assertEqual(row["BackgroundSource"], "insufficient")
        self.assertIn("no background correction was applied", log)


class RegressionRateTests(unittest.TestCase):
    def test_slope_ignores_constant_offset(self):
        hours = pd.Series([1.0, 2.0, 4.0])
        volumes = 5_000 + 1_000 * hours
        self.assertAlmostEqual(validator._regression_rate_m3ph(volumes, hours), 1_000)
        self.assertTrue(np.isnan(validator._regression_rate_m3ph(volumes[:2], hours[:2])))


if __name__ == "__main__":
    unittest.main()
