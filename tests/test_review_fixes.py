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
from shapely.geometry import box

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


if __name__ == "__main__":
    unittest.main()
