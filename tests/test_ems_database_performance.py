import json
import sqlite3
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ems_database
from ems_database import EMSDatabase, EMSRecord
import lowering_factor_validator as validator
import pandas as pd
from resolution_grid import create_aligned_child_grid, inspect_master_grid


class ValidationMetadataTests(unittest.TestCase):
    def test_non_finite_metadata_values_are_serialized_as_json_null(self):
        metadata = {
            "summary": [
                {
                    "rate": float("nan"),
                    "difference": float("inf"),
                    "coverage": pd.NA,
                    "count": 3,
                }
            ]
        }

        serialized = json.dumps(validator._json_safe(metadata), allow_nan=False)

        self.assertEqual(
            json.loads(serialized),
            {
                "summary": [
                    {
                        "rate": None,
                        "difference": None,
                        "coverage": None,
                        "count": 3,
                    }
                ]
            },
        )


class ChangeSignConventionTests(unittest.TestCase):
    def test_change_classes_follow_positive_lowering_convention(self):
        cases = (
            (0.30, "DEEPENING"),
            (0.01, "DEEPENING"),
            (0.00, "NO_SIGNIFICANT_CHANGE"),
            (-0.01, "ACCRETION"),
            (-0.30, "ACCRETION"),
        )

        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(validator.classify_change(value, 0.0), expected)

    def test_change_tolerance_still_marks_small_changes_insignificant(self):
        self.assertEqual(
            validator.classify_change(0.01, 0.05),
            "NO_SIGNIFICANT_CHANGE",
        )
        self.assertEqual(
            validator.classify_change(-0.01, 0.05),
            "NO_SIGNIFICANT_CHANGE",
        )

    def test_change_distribution_uses_correct_sign_for_area_and_volume(self):
        summary = validator.summarise_change_distribution(
            pd.Series([0.30, -0.10, 0.01]),
            pd.Series([10.0, 20.0, 30.0]),
            tolerance_m=0.05,
        )

        self.assertEqual(summary["deepening_area_m2"], 10.0)
        self.assertEqual(summary["accretion_area_m2"], 20.0)
        self.assertEqual(summary["stable_area_m2"], 30.0)
        self.assertEqual(summary["gross_deepening_volume_m3"], 3.0)
        self.assertEqual(summary["gross_accretion_volume_m3"], 2.0)
        self.assertEqual(summary["net_lowering_volume_m3"], 1.0)


class ValidationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.settings = SimpleNamespace(
            minimum_ems_hours=0.5,
            minimum_coverage_percent=80.0,
            minimum_detectable_change_m=0.2,
            maximum_rate_tph=1500.0,
            sample_interval_minutes=1.0,
        )

    def _diagnostic_row(
        self,
        cell_id,
        *,
        raster_pixels=1,
        ems_count=30,
        duration=0.5,
        bathy=0.5,
        area=100.0,
        coverage=100.0,
        volume=50.0,
        rate=170.0,
        include=True,
    ):
        return {
            "GridID": cell_id,
            "LicenceNo": "L1",
            "RasterPixels": raster_pixels,
            "CoveragePct": coverage,
            "ObsLower_m": bathy,
            "ValidArea_m2": area,
            "EMS_PointCount": ems_count,
            "EMS_Min": duration * 60,
            "EMS_Hours": duration,
            "NetVolume_m3": volume,
            "ObsRate_m3ph": (
                volume / duration if duration else float("inf")
            ),
            "BulkDensity": 1.7,
            "ObsRate_tph": rate,
            "ObsLowerFac": bathy / duration if duration else float("inf"),
            "IncludeFlag": include,
            "CalibrationClass": "A_ACCEPTED" if include else "B_REVIEW",
            "QA_Reason": "" if include else "test rejection",
        }

    def _diagnostics(self, rows):
        source = pd.DataFrame(rows).set_index("GridID", drop=False)
        return validator._build_validation_diagnostic_table(
            source, source, self.settings
        )

    def test_diagnostic_categories_and_multiple_rejection_reasons(self):
        rows = [
            self._diagnostic_row("valid"),
            self._diagnostic_row(
                "zero_bathy", bathy=0.0, volume=0.0, rate=0.0, include=False
            ),
            self._diagnostic_row(
                "negative_bathy", bathy=-1.0, volume=-100.0, rate=-340.0,
                include=False,
            ),
            self._diagnostic_row(
                "zero_duration", ems_count=0, duration=0.0, volume=100.0,
                rate=float("inf"), include=False,
            ),
            self._diagnostic_row(
                "missing_ems", ems_count=0, duration=0.0, volume=0.0,
                rate=float("nan"), include=False,
            ),
            self._diagnostic_row(
                "invalid_bathy", bathy=float("nan"), volume=float("nan"),
                rate=float("nan"), include=False,
            ),
            self._diagnostic_row(
                "no_raster", raster_pixels=0, bathy=float("nan"),
                area=0.0, coverage=0.0, volume=float("nan"),
                rate=float("nan"), include=False,
            ),
            self._diagnostic_row(
                "finite_volume_invalid_rate", ems_count=0, duration=0.0,
                volume=100.0, rate=float("inf"), include=False,
            ),
            self._diagnostic_row(
                "multiple", ems_count=0, duration=0.0, bathy=-1.0,
                coverage=10.0, volume=-100.0, rate=float("-inf"),
                include=False,
            ),
            self._diagnostic_row(
                "below_threshold", bathy=0.1, volume=10.0, rate=34.0,
                include=False,
            ),
        ]
        table = self._diagnostics(rows).set_index("GridID")

        self.assertTrue(table.loc["valid", "valid_positive_rate"])
        self.assertEqual(
            table.loc["zero_bathy", "primary_rejection_reason"],
            "ZERO_BATHY_CHANGE",
        )
        self.assertIn(
            "NEGATIVE_BATHY_CHANGE",
            table.loc["negative_bathy", "rejection_reasons"],
        )
        self.assertIn(
            "ZERO_DREDGING_DURATION",
            table.loc["zero_duration", "rejection_reasons"],
        )
        self.assertIn("NO_EMS_RECORDS", table.loc["missing_ems", "rejection_reasons"])
        self.assertIn(
            "INVALID_BATHYMETRY",
            table.loc["invalid_bathy", "rejection_reasons"],
        )
        self.assertEqual(
            table.loc["no_raster", "primary_rejection_reason"],
            "NO_VALID_BATHYMETRY",
        )
        self.assertIn(
            "INVALID_RATE",
            table.loc["finite_volume_invalid_rate", "rejection_reasons"],
        )
        self.assertTrue(
            {
                "NO_EMS_RECORDS",
                "ZERO_DREDGING_DURATION",
                "COVERAGE_BELOW_MINIMUM",
                "NEGATIVE_BATHY_CHANGE",
                "NEGATIVE_VOLUME",
                "INVALID_RATE",
            }.issubset(set(table.loc["multiple", "rejection_reasons"].split(";")))
        )
        self.assertIn(
            "BELOW_MINIMUM_DETECTABLE_CHANGE",
            table.loc["below_threshold", "rejection_reasons"],
        )

    def test_sign_convention_diagnostics_report_transforms_and_warning(self):
        table = self._diagnostics([self._diagnostic_row("cell")])
        messages = []
        validator._log_validation_diagnostics(
            table,
            table,
            table,
            30,
            {"valid_grid": 30},
            pd.Series([30]),
            self.settings,
            "RASTER",
            "POSITIVE_DEPTH",
            "negative_is_lowering",
            messages.append,
        )
        report = "\n".join(messages)
        self.assertIn("positive ObsLower_m is expected to mean seabed lowering", report)
        self.assertIn("SIGN-CONVENTION WARNING", report)
        self.assertIn("NetVolume_m3 = ObsLower_m x ValidArea_m2", report)
        self.assertIn("duration units are hours", report)

    def test_hand_calculable_cell_preserves_formula_and_writes_additional_csv(self):
        raster_path = self.root / "difference.tif"
        output = self.root / "output"
        output.mkdir()
        with patch.dict(
            "os.environ",
            {"PROJ_LIB": str(Path(rasterio.__file__).parent / "proj_data")},
        ):
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=1,
                height=1,
                count=1,
                dtype="float32",
                crs="EPSG:27700",
                transform=from_origin(0, 250, 250, 250),
                nodata=-9999.0,
            ) as dataset:
                dataset.write(np.array([[0.5]], dtype=np.float32), 1)
        grid = gpd.GeoDataFrame(
            {"GridID": ["cell-1"]},
            geometry=[box(0, 0, 250, 250)],
            crs="EPSG:27700",
        )
        licences = gpd.GeoDataFrame(
            {"LicenceNo": ["licence-1"]},
            geometry=[box(0, 0, 250, 250)],
            crs="EPSG:27700",
        )
        ems = gpd.GeoDataFrame(
            {"_datetime": [pd.Timestamp("2024-01-01")] * 30},
            geometry=[Point(125, 125)] * 30,
            crs="EPSG:27700",
        )
        settings = validator.Settings(
            difference_raster=str(raster_path),
            calibration_grid=str(self.root / "grid.gpkg"),
            licence_layer=str(self.root / "licences.gpkg"),
            ems_source=str(self.root / "ems.csv"),
            output_folder=str(output),
            survey_start="01/01/2024",
            survey_end="01/01/2024",
            minimum_ems_hours=0.5,
            minimum_coverage_percent=80.0,
            minimum_detectable_change_m=0.2,
            maximum_rate_tph=200_000,
            sample_interval_minutes=1.0,
            bulk_density=1.7,
        )
        messages = []
        with patch.dict(
            "os.environ",
            {"PROJ_LIB": str(Path(rasterio.__file__).parent / "proj_data")},
        ):
            summary = validator._run_validation_single(
                settings,
                messages.append,
                grid_override=grid,
                licences_override=licences,
                ems_override=ems,
                difference_raster_override=str(raster_path),
            )
        table = pd.read_csv(output / "validation_250m_diagnostics.csv")
        row = table.iloc[0]

        self.assertAlmostEqual(row["ObsLower_m"], 0.5)
        self.assertAlmostEqual(row["ValidArea_m2"], 62_500)
        self.assertAlmostEqual(row["NetVolume_m3"], 31_250)
        self.assertAlmostEqual(row["EMS_Hours"], 0.5)
        self.assertAlmostEqual(row["ObsRate_m3ph"], 62_500)
        self.assertAlmostEqual(row["ObsRate_tph"], 106_250)
        self.assertTrue(bool(row["valid_positive_rate"]))
        self.assertAlmostEqual(summary.iloc[0]["ValidatedRate_m3ph"], 62_500)
        self.assertIn("250 m VALIDATION DIAGNOSTIC SUMMARY", "\n".join(messages))
        self.assertIn("Diagnostic table written to:", "\n".join(messages))
        self.assertIn(
            "User-supplied difference raster",
            "\n".join(messages),
        )


class BathymetricRasterValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.proj_data = str(Path(rasterio.__file__).parent / "proj_data")

    def _write_float_tiff(self, path, values, *, nodata=None, crs="EPSG:27700"):
        height, width = values.shape
        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=1,
                dtype="float32",
                crs=crs,
                transform=from_origin(0, height, 1, 1),
                nodata=nodata,
            ) as dataset:
                dataset.write(np.asarray(values, dtype=np.float32), 1)

    def test_float_tiff_counts_finite_nodata_zero_and_sign_values(self):
        path = self.root / "difference.tif"
        self._write_float_tiff(
            path,
            np.array([[-1.0, 0.0, 2.0], [-9999.0, np.nan, np.inf]]),
            nodata=-9999.0,
        )
        messages = []

        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            info = validator.validate_bathymetric_difference_raster(
                str(path),
                source_sign="negative_is_lowering",
                log=messages.append,
            )

        self.assertEqual(info["format"], "GeoTIFF")
        self.assertEqual(info["dtype"], "float32")
        self.assertEqual(info["valid"], 3)
        self.assertEqual(info["masked"], 1)
        self.assertEqual(info["nan"], 1)
        self.assertEqual(info["infinite"], 1)
        self.assertEqual(info["negative"], 1)
        self.assertEqual(info["zero"], 1)
        self.assertEqual(info["positive"], 1)
        self.assertEqual(info["median_sample"], 0.0)
        self.assertAlmostEqual(info["mean"], 1.0 / 3.0)
        self.assertTrue(info["positive_values_are_lowering"] is False)
        self.assertIn("Status: VALID", "\n".join(messages))

    def test_zonal_statistics_warn_when_analysis_has_no_configured_lowering_sign(self):
        path = self.root / "all-positive.tif"
        self._write_float_tiff(path, np.ones((2, 2), dtype=np.float32))
        grid = gpd.GeoDataFrame(
            {"GridID": ["cell"]},
            geometry=[box(0, 0, 2, 2)],
            crs="EPSG:27700",
        )
        messages = []

        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            stats = validator.zonal_change(
                grid,
                str(path),
                "negative_is_lowering",
                messages.append,
            )

        self.assertLess(stats.iloc[0]["ObsLower_m"], 0)
        self.assertTrue(
            any("No bathymetric-difference pixels matching" in line for line in messages)
        )

    def test_rejects_rgba_display_raster_instead_of_using_band_one(self):
        path = self.root / "rendered.tif"
        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=4,
                dtype="uint8",
                crs="EPSG:27700",
                transform=from_origin(0, 2, 1, 1),
                nodata=0,
            ) as dataset:
                dataset.write(np.ones((4, 2, 2), dtype=np.uint8))

            with self.assertRaisesRegex(
                ValueError,
                "UNSUITABLE BATHYMETRIC-DIFFERENCE RASTER",
            ) as raised:
                validator.validate_bathymetric_difference_raster(str(path))

        self.assertIn("4 bands", str(raised.exception))
        self.assertIn("No EMS data has been queried", str(raised.exception))

    def test_validates_esri_float_grid_and_companion_file_size(self):
        path = self.root / "bathy.flt"
        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            with rasterio.open(
                path,
                "w",
                driver="EHdr",
                width=2,
                height=2,
                count=1,
                dtype="float32",
                crs="EPSG:27700",
                transform=from_origin(0, 2, 1, 1),
                nodata=-9999.0,
            ) as dataset:
                dataset.write(
                    np.array([[1.0, 0.0], [-1.0, -9999.0]], dtype=np.float32),
                    1,
                )
            info = validator.validate_bathymetric_difference_raster(str(path))

        self.assertEqual(info["format"], "ESRI Binary Float Grid")
        self.assertEqual(info["valid"], 3)
        self.assertTrue(path.with_suffix(".hdr").is_file())
        self.assertTrue(path.with_suffix(".prj").is_file())

    def test_accepts_standard_lsbf_first_float_grid_byte_order_label(self):
        path = self.root / "lsb-first.flt"
        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            with rasterio.open(
                path,
                "w",
                driver="EHdr",
                width=2,
                height=2,
                count=1,
                dtype="float32",
                crs="EPSG:27700",
                transform=from_origin(0, 2, 1, 1),
            ) as dataset:
                dataset.write(np.ones((2, 2), dtype=np.float32), 1)

            header = path.with_suffix(".hdr")
            contents = header.read_text(encoding="ascii")
            self.assertIn("BYTEORDER", contents)
            header.write_text(
                contents.replace("BYTEORDER      I", "BYTEORDER      LSBFirst"),
                encoding="ascii",
            )
            info = validator.validate_bathymetric_difference_raster(str(path))

        self.assertEqual(info["format"], "ESRI Binary Float Grid")
        self.assertEqual(info["valid"], 4)

    def test_missing_float_grid_header_and_projection_fail_actionably(self):
        path = self.root / "missing.flt"
        path.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Missing ESRI Float Grid header"):
            validator.validate_bathymetric_difference_raster(str(path))

        path.with_suffix(".hdr").write_text(
            "ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 1\n"
            "NODATA_value -9999\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            ValueError,
            "no usable coordinate reference system",
        ):
            validator.validate_bathymetric_difference_raster(str(path))

    def test_invalid_raster_stops_before_ems_and_output_creation(self):
        path = self.root / "rgba.tif"
        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=4,
                dtype="uint8",
                crs="EPSG:27700",
                transform=from_origin(0, 2, 1, 1),
            ) as dataset:
                dataset.write(np.ones((4, 2, 2), dtype=np.uint8))

        output = self.root / "not-created"
        settings = validator.Settings(
            difference_raster=str(path),
            calibration_grid=str(self.root / "grid.gpkg"),
            licence_layer=str(self.root / "licences.gpkg"),
            ems_source=str(self.root / "ems"),
            output_folder=str(output),
            survey_start="01/01/2024",
            survey_end="01/01/2024",
        )
        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            with patch.object(validator, "load_ems") as load_ems:
                with self.assertRaisesRegex(
                    ValueError,
                    "UNSUITABLE BATHYMETRIC-DIFFERENCE RASTER",
                ):
                    validator.run_validation(settings)

        load_ems.assert_not_called()
        self.assertFalse(output.exists())


class ExperimentalResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.proj_data = str(Path(rasterio.__file__).parent / "proj_data")

    def _parent_grid(self):
        return gpd.GeoDataFrame(
            {"GridID": ["parent-1"]},
            geometry=[box(0, 0, 250, 250)],
            crs="EPSG:27700",
        )

    def _write_raster(self, path, values, nodata=-9999.0):
        height, width = values.shape
        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=1,
                dtype="float32",
                crs="EPSG:27700",
                transform=from_origin(0, height, 1, 1),
                nodata=nodata,
            ) as dataset:
                dataset.write(np.asarray(values, dtype=np.float32), 1)

    def test_parent_subdivides_into_25_aligned_stable_children(self):
        parent = self._parent_grid()
        layout = inspect_master_grid(parent, "GridID")
        first = create_aligned_child_grid(parent, layout, child_size_m=50)
        second = create_aligned_child_grid(parent, layout, child_size_m=50)

        self.assertEqual(len(first), 25)
        self.assertEqual(first.geometry.area.sum(), 62_500)
        self.assertTrue(first.geometry.area.eq(2_500).all())
        bounds = first.geometry.bounds
        self.assertTrue((bounds["maxx"] - bounds["minx"]).eq(50.0).all())
        self.assertTrue((bounds["maxy"] - bounds["miny"]).eq(50.0).all())
        self.assertEqual(bounds["minx"].min(), 0.0)
        self.assertEqual(bounds["maxx"].max(), 250.0)
        self.assertEqual(bounds["miny"].min(), 0.0)
        self.assertEqual(bounds["maxy"].max(), 250.0)
        self.assertEqual(
            sum(bounds["maxx"].eq(x).sum() for x in (50.0, 100.0, 150.0, 200.0, 250.0)),
            25,
        )
        self.assertEqual(first["Parent250_ID"].tolist(), ["parent-1"] * 25)
        self.assertEqual(first["Child50_ID"].tolist(), second["Child50_ID"].tolist())
        self.assertEqual(first["Child50_ID"].nunique(), 25)
        self.assertEqual(
            first["GridID"].tolist(),
            [f"parent-1_r{row}_c{column}" for row in range(1, 6) for column in range(1, 6)],
        )

    def test_raster_filtered_parents_keep_authoritative_50m_alignment(self):
        parents = gpd.GeoDataFrame(
            {
                "GridID": [
                    f"parent-r{row}-c{column}"
                    for row in range(2)
                    for column in range(2)
                ]
            },
            geometry=[
                box(column * 250, row * 250, (column + 1) * 250, (row + 1) * 250)
                for row in range(2)
                for column in range(2)
            ],
            crs="EPSG:27700",
        )
        layout = inspect_master_grid(parents, "GridID")
        western_extent = parents.loc[parents["GridID"].str.endswith("c0")].copy()
        eastern_extent = parents.loc[parents["GridID"].str.endswith("c1")].copy()

        western_children = create_aligned_child_grid(
            western_extent, layout, child_size_m=50
        )
        eastern_children = create_aligned_child_grid(
            eastern_extent, layout, child_size_m=50
        )
        repeated_eastern_children = create_aligned_child_grid(
            eastern_extent, layout, child_size_m=50
        )

        self.assertEqual(len(western_children), 50)
        self.assertEqual(len(eastern_children), 50)
        self.assertTrue(
            eastern_children.groupby("Parent250_ID").size().eq(25).all()
        )
        self.assertTrue(
            np.allclose(
                eastern_children.groupby("Parent250_ID").geometry.apply(
                    lambda geometries: geometries.area.sum()
                ),
                62_500.0,
            )
        )
        eastern_parent = eastern_children.loc[
            eastern_children["Parent250_ID"] == "parent-r0-c1"
        ].sort_values(["ChildRow", "ChildColumn"])
        self.assertEqual(
            eastern_parent["Parent250_ID"].unique().tolist(), ["parent-r0-c1"]
        )
        self.assertEqual(
            sorted({round(geometry.bounds[0], 6) for geometry in eastern_parent.geometry}),
            [250.0, 300.0, 350.0, 400.0, 450.0],
        )
        self.assertEqual(
            sorted({round(geometry.bounds[1], 6) for geometry in eastern_parent.geometry}),
            [0.0, 50.0, 100.0, 150.0, 200.0],
        )
        self.assertEqual(
            eastern_children["Child50_ID"].tolist(),
            repeated_eastern_children["Child50_ID"].tolist(),
        )
        self.assertTrue(
            all(
                left.equals(right)
                for left, right in zip(
                    eastern_children.geometry, repeated_eastern_children.geometry
                )
            )
        )

    def test_clipped_source_features_recover_authoritative_full_parent_cells(self):
        clip = box(100, 100, 400, 400)
        source_parents = gpd.GeoDataFrame(
            {
                "GridID": [
                    "parent-sw",
                    "parent-se",
                    "parent-nw",
                    "parent-ne",
                ]
            },
            geometry=[
                box(0, 0, 250, 250).intersection(clip),
                box(250, 0, 500, 250).intersection(clip),
                box(0, 250, 250, 500).intersection(clip),
                box(250, 250, 500, 500).intersection(clip),
            ],
            crs="EPSG:27700",
        )

        layout = inspect_master_grid(source_parents, "GridID")
        children = create_aligned_child_grid(
            source_parents.iloc[[0]], layout, child_size_m=50
        )

        self.assertEqual(len(children), 25)
        self.assertEqual(children["Parent250_ID"].unique().tolist(), ["parent-sw"])
        self.assertTrue(children.geometry.area.eq(2_500).all())
        self.assertAlmostEqual(children.geometry.area.sum(), 62_500)
        self.assertEqual(children.geometry.total_bounds.tolist(), [0.0, 0.0, 250.0, 250.0])

    def test_adjacent_parent_children_share_exact_boundaries(self):
        parents = gpd.GeoDataFrame(
            {"GridID": ["parent-west", "parent-east"]},
            geometry=[box(0, 0, 250, 250), box(250, 0, 500, 250)],
            crs="EPSG:27700",
        )
        layout = inspect_master_grid(parents, "GridID")
        children = create_aligned_child_grid(parents, layout, child_size_m=50)
        west = children.loc[children["Parent250_ID"] == "parent-west"]
        east = children.loc[children["Parent250_ID"] == "parent-east"]

        self.assertTrue(children.groupby("Parent250_ID").size().eq(25).all())
        self.assertTrue(children.geometry.area.eq(2_500.0).all())
        self.assertTrue(west.geometry.bounds["maxx"].eq(250.0).any())
        self.assertTrue(east.geometry.bounds["minx"].eq(250.0).any())
        self.assertEqual(
            sorted(west.geometry.bounds["maxy"].unique()),
            sorted(east.geometry.bounds["maxy"].unique()),
        )
        self.assertEqual(west["Parent250_ID"].nunique(), 1)
        self.assertEqual(east["Parent250_ID"].nunique(), 1)

    def test_boundary_ems_is_assigned_once_to_stable_child(self):
        parent = self._parent_grid()
        children = create_aligned_child_grid(
            parent,
            inspect_master_grid(parent, "GridID"),
            child_size_m=50,
        )
        ems = gpd.GeoDataFrame(
            {"_datetime": [pd.Timestamp("2024-01-01")]},
            geometry=[Point(50, 225)],
            crs="EPSG:27700",
        )

        counts = validator._ems_grid_counts(
            ems, children.rename(columns={"GridID": "GridID"}), True
        )

        self.assertEqual(int(counts.sum()), 1)
        self.assertEqual(int(counts.loc["parent-1_r5_c1"]), 1)

    def test_partial_raster_coverage_uses_only_valid_pixels(self):
        raster_path = self.root / "partial.tif"
        values = np.full((50, 50), 0.3, dtype=np.float32)
        values[:, 25:] = -9999.0
        self._write_raster(raster_path, values)
        cell = gpd.GeoDataFrame(
            {"GridID": ["child-1"]},
            geometry=[box(0, 0, 50, 50)],
            crs="EPSG:27700",
        )

        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            stats = validator.zonal_change(
                cell,
                str(raster_path),
                "positive_is_lowering",
                lambda _message: None,
            )

        self.assertEqual(stats.iloc[0]["RasterPixels"], 1_250)
        self.assertAlmostEqual(stats.iloc[0]["ValidArea_m2"], 1_250)
        self.assertAlmostEqual(stats.iloc[0]["CoveragePct"], 50.0)
        self.assertAlmostEqual(stats.iloc[0]["ObsLower_m"], 0.3)

    def test_50m_hand_calculation_and_existing_volume_time_formulas(self):
        raster_path = self.root / "difference-50m.tif"
        self._write_raster(
            raster_path,
            np.full((50, 50), 0.3, dtype=np.float32),
        )
        output = self.root / "output"
        output.mkdir()
        grid = gpd.GeoDataFrame(
            {
                "GridID": ["parent-1_r1_c1"],
                "Parent250_ID": ["parent-1"],
                "Child50_ID": ["parent-1_r1_c1"],
                "ChildRow": [0],
                "ChildColumn": [0],
            },
            geometry=[box(0, 0, 50, 50)],
            crs="EPSG:27700",
        )
        licences = gpd.GeoDataFrame(
            {"LicenceNo": ["licence-1"]},
            geometry=[box(0, 0, 50, 50)],
            crs="EPSG:27700",
        )
        ems = gpd.GeoDataFrame(
            {"_datetime": [pd.Timestamp("2024-01-01")] * 60},
            geometry=[Point(25, 25)] * 60,
            crs="EPSG:27700",
        )
        settings = validator.Settings(
            difference_raster=str(raster_path),
            calibration_grid=str(self.root / "grid.gpkg"),
            licence_layer=str(self.root / "licences.gpkg"),
            ems_source=str(self.root / "ems.csv"),
            output_folder=str(output),
            survey_start="01/01/2024",
            survey_end="01/01/2024",
            minimum_ems_hours=0.5,
            minimum_coverage_percent=80.0,
            minimum_detectable_change_m=0.2,
            maximum_rate_tph=1_500.0,
            sample_interval_minutes=1.0,
            bulk_density=1.7,
            resolution_mode=validator.RESOLUTION_50,
        )

        with patch.dict("os.environ", {"PROJ_LIB": self.proj_data}):
            summary = validator._run_validation_single(
                settings,
                grid_override=grid,
                licences_override=licences,
                ems_override=ems,
                difference_raster_override=str(raster_path),
                resolution_label=validator.RESOLUTION_50,
                output_suffix="_50m_experimental",
            )
        diagnostic = pd.read_csv(output / "validation_50m_diagnostics.csv").iloc[0]
        self.assertAlmostEqual(diagnostic["ValidArea_m2"], 2_500)
        self.assertAlmostEqual(diagnostic["NetVolume_m3"], 750, places=3)
        self.assertAlmostEqual(diagnostic["EMS_Hours"], 1.0)
        self.assertAlmostEqual(diagnostic["ObsRate_tph"], 1_275, places=3)
        self.assertEqual(diagnostic["Parent250_ID"], "parent-1")
        self.assertEqual(diagnostic["Child50_ID"], "parent-1_r1_c1")
        self.assertEqual(diagnostic["ChangeClass"], "DEEPENING")
        self.assertTrue(bool(diagnostic["valid_positive_rate"]))
        self.assertAlmostEqual(
            summary.iloc[0]["ValidatedRate_m3ph"], 750, places=3
        )
        self.assertTrue(
            (output / "lowering_factor_validation_grid_50m_experimental.gpkg").exists()
        )

    def test_parent_aggregation_uses_summed_volume_over_summed_hours(self):
        diagnostics = pd.DataFrame(
            {
                "Parent250_ID": ["parent-1", "parent-1"],
                "raster_valid": [True, True],
                "EMS_PointCount": [60, 30],
                "EMS_Hours": [1.0, 0.5],
                "ValidArea_m2": [2_500.0, 2_500.0],
                "NetVolume_m3": [750.0, 1_500.0],
            }
        )

        result = validator._aggregate_50m_parent_cells(diagnostics, 1.7).iloc[0]

        self.assertEqual(result["ValidRasterChildren"], 2)
        self.assertEqual(result["EMSReceivingChildren"], 2)
        self.assertAlmostEqual(result["ValidChildArea_m2"], 5_000)
        self.assertAlmostEqual(result["RateContributingEMS_Hours"], 1.5)
        self.assertAlmostEqual(result["RateContributingVolume_m3"], 2_250)
        self.assertAlmostEqual(result["ExperimentalCombinedRate_m3ph"], 1_500)
        self.assertAlmostEqual(result["ExperimentalCombinedRate_tph"], 2_550)

    def test_both_resolution_comparison_writes_metrics_and_parent_aggregation(self):
        settings = validator.Settings(
            difference_raster="surface.flt",
            calibration_grid="grid.gpkg",
            licence_layer="licences.gpkg",
            ems_source="ems",
            output_folder=str(self.root),
            survey_start="01/01/2024",
            survey_end="01/01/2024",
            bulk_density=1.7,
        )
        common = {
            "raster_valid": True,
            "EMS_PointCount": 60,
            "EMS_Hours": 1.0,
            "ObsLower_m": 0.3,
            "NetVolume_m3": 750.0,
            "ObsRate_tph": 1_275.0,
            "valid_positive_rate": True,
            "CoveragePct": 100.0,
            "rejection_reasons": "",
        }
        diagnostics_250m = pd.DataFrame(
            [{**common, "ValidArea_m2": 62_500.0}]
        )
        diagnostics_50m = pd.DataFrame(
            [
                {
                    **common,
                    "Parent250_ID": "parent-1",
                    "ValidArea_m2": 2_500.0,
                }
            ]
        )
        messages = []

        validator._write_resolution_comparison(
            self.root,
            settings,
            diagnostics_250m,
            diagnostics_50m,
            messages.append,
        )

        comparison = pd.read_csv(
            self.root / "validation_50m_vs_250m_summary.csv"
        ).set_index("Metric")
        aggregation = pd.read_csv(
            self.root / "validation_50m_parent_250m_aggregation.csv"
        )
        self.assertEqual(comparison.loc["grid_cells", "250m"], 1)
        self.assertEqual(comparison.loc["grid_cells", "50m"], 1)
        self.assertEqual(len(aggregation), 1)
        self.assertEqual(aggregation.loc[0, "Parent250_ID"], "parent-1")
        self.assertTrue(any("VALIDATION RESOLUTION COMPARISON" in line for line in messages))

    def test_50m_only_run_does_not_call_source_model_writeback(self):
        parent = self._parent_grid()
        licences = gpd.GeoDataFrame(
            {"LicenceNo": ["licence-1"]},
            geometry=[box(0, 0, 250, 250)],
            crs="EPSG:27700",
        )
        ems = gpd.GeoDataFrame(
            {"_datetime": []},
            geometry=gpd.GeoSeries([], crs="EPSG:27700"),
            crs="EPSG:27700",
        )
        settings = validator.Settings(
            difference_raster="surface.flt",
            calibration_grid="grid.gpkg",
            licence_layer="licences.gpkg",
            ems_source="ems",
            output_folder=str(self.root / "output"),
            survey_start="01/01/2024",
            survey_end="01/01/2024",
            resolution_mode=validator.RESOLUTION_50,
            analysis_extent="TRACK_FOOTPRINT",
        )
        summary = pd.DataFrame(
            {"LicenceNo": ["licence-1"], "ValidatedRate_m3ph": [750.0]}
        )

        with (
            patch.object(validator, "prepare_difference_raster", return_value="surface.flt"),
            patch.object(validator, "validate_bathymetric_difference_raster", return_value={}),
            patch.object(
                validator,
                "read_vector",
                side_effect=[parent, licences],
            ),
            patch.object(validator, "load_ems", return_value=ems),
            patch.object(validator, "_run_validation_single", return_value=summary),
            patch.object(validator, "update_model_lowering_factors") as update_model,
        ):
            validator.run_validation(settings)

        update_model.assert_not_called()

    def test_full_overlap_inspects_full_master_before_loading_filtered_parents(self):
        clip = box(100, 100, 400, 400)
        complete_grid = gpd.GeoDataFrame(
            {
                "GridID": [
                    "parent-sw",
                    "parent-se",
                    "parent-nw",
                    "parent-ne",
                ]
            },
            geometry=[
                box(0, 0, 250, 250).intersection(clip),
                box(250, 0, 500, 250).intersection(clip),
                box(0, 250, 250, 500).intersection(clip),
                box(250, 250, 500, 500).intersection(clip),
            ],
            crs="EPSG:27700",
        )
        working_grid = complete_grid.iloc[[1]].copy()
        licences = gpd.GeoDataFrame(
            {"LicenceNo": ["licence-1"]},
            geometry=[box(0, 0, 500, 250)],
            crs="EPSG:27700",
        )
        ems = gpd.GeoDataFrame(
            {"_datetime": []},
            geometry=gpd.GeoSeries([], crs="EPSG:27700"),
            crs="EPSG:27700",
        )
        settings = validator.Settings(
            difference_raster="surface.flt",
            calibration_grid="grid.gpkg",
            licence_layer="licences.gpkg",
            ems_source="ems",
            output_folder=str(self.root / "output"),
            survey_start="01/01/2024",
            survey_end="01/01/2024",
            resolution_mode=validator.RESOLUTION_50,
            analysis_extent="FULL_OVERLAP",
        )
        summaries = pd.DataFrame(
            {"LicenceNo": ["licence-1"], "ValidatedRate_m3ph": [750.0]}
        )
        reads = [complete_grid, working_grid, licences]
        read_calls = []
        run_calls = []

        def read_vector(*args, **kwargs):
            read_calls.append((args, kwargs))
            return reads.pop(0)

        def run_single(*args, **kwargs):
            run_calls.append(kwargs)
            return summaries.copy()

        messages = []
        with (
            patch.object(
                validator, "prepare_difference_raster", return_value="surface.flt"
            ),
            patch.object(
                validator, "validate_bathymetric_difference_raster", return_value={}
            ),
            patch.object(
                validator,
                "raster_bounds_in_vector_crs",
                return_value=(250.0, 0.0, 500.0, 250.0),
            ),
            patch.object(validator, "read_vector", side_effect=read_vector),
            patch.object(validator, "load_ems", return_value=ems),
            patch.object(validator, "_run_validation_single", side_effect=run_single),
        ):
            validator.run_validation(settings, messages.append)

        self.assertIsNone(read_calls[0][1].get("bbox"))
        self.assertEqual(read_calls[1][1]["bbox"], (250.0, 0.0, 500.0, 250.0))
        children = run_calls[0]["grid_override"]
        self.assertEqual(len(children), 25)
        self.assertEqual(children["Parent250_ID"].unique().tolist(), ["parent-se"])
        self.assertEqual(children.geometry.total_bounds.tolist(), [250.0, 0.0, 500.0, 250.0])
        self.assertLess(
            next(
                index
                for index, message in enumerate(messages)
                if message == "Master-grid alignment validated."
            ),
            next(
                index
                for index, message in enumerate(messages)
                if message == "Restricting working grid to survey raster coverage..."
            ),
        )
        self.assertTrue(any("Master grid origin: X=0.000, Y=0.000" in message for message in messages))
        self.assertTrue(
            any("Survey/raster candidate parent cells: 1" in message for message in messages)
        )
        self.assertTrue(
            any("Licence-assigned parent cells: 1" in message for message in messages)
        )

    def test_250m_full_overlap_keeps_existing_filtered_grid_path(self):
        parent = self._parent_grid()
        licences = gpd.GeoDataFrame(
            {"LicenceNo": ["licence-1"]},
            geometry=[box(0, 0, 250, 250)],
            crs="EPSG:27700",
        )
        ems = gpd.GeoDataFrame(
            {"_datetime": []},
            geometry=gpd.GeoSeries([], crs="EPSG:27700"),
            crs="EPSG:27700",
        )
        settings = validator.Settings(
            difference_raster="surface.flt",
            calibration_grid="grid.gpkg",
            licence_layer="licences.gpkg",
            ems_source="ems",
            output_folder=str(self.root / "output-250"),
            survey_start="01/01/2024",
            survey_end="01/01/2024",
            resolution_mode=validator.RESOLUTION_250,
            analysis_extent="FULL_OVERLAP",
        )
        summary = pd.DataFrame(
            {"LicenceNo": ["licence-1"], "ValidatedRate_m3ph": [750.0]}
        )
        with (
            patch.object(
                validator, "prepare_difference_raster", return_value="surface.flt"
            ),
            patch.object(
                validator, "validate_bathymetric_difference_raster", return_value={}
            ),
            patch.object(
                validator,
                "raster_bounds_in_vector_crs",
                return_value=(0.0, 0.0, 250.0, 250.0),
            ),
            patch.object(validator, "read_vector", side_effect=[parent, licences]) as read,
            patch.object(validator, "inspect_master_grid") as inspect,
            patch.object(validator, "load_ems", return_value=ems),
            patch.object(validator, "_run_validation_single", return_value=summary),
            patch.object(validator, "update_model_lowering_factors"),
        ):
            validator.run_validation(settings)

        inspect.assert_not_called()
        self.assertEqual(read.call_args_list[0].kwargs["bbox"], (0.0, 0.0, 250.0, 250.0))


class EMSDatabasePerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source.csv"
        self.source.write_text("test source", encoding="utf-8")
        self.database = EMSDatabase(self.source, self.root / "index.sqlite3")
        self.profile = "test-profile"
        self.records = [
            EMSRecord(
                f"2024-01-01 00:00:0{index}.000000",
                float(index),
                float(index + 1),
                dredge,
                vessel,
                "EPSG:4326",
            )
            for index, (dredge, vessel) in enumerate(
                [("3", "v1"), ("4", "v1"), ("inactive", "v2")], start=1
            )
        ]

    def _parser(self, _path):
        return iter(self.records)

    def _source_file_counts(self):
        connection = self.database._connect()
        try:
            main_count = connection.execute(
                "SELECT COUNT(*) FROM ems_records"
            ).fetchone()[0]
            bounds_count = connection.execute(
                "SELECT COUNT(*) FROM ems_record_bounds"
            ).fetchone()[0]
            return main_count, bounds_count
        finally:
            connection.close()

    def test_startup_profile_times_initialization_and_index_checks(self):
        profile_messages = []
        database_path = self.root / "startup-profile.sqlite3"
        EMSDatabase(self.source, database_path, log=profile_messages.append)
        profile = profile_messages[0]

        for expected in (
            "EMS database startup profile",
            "Database size:",
            "Source-root normalisation:",
            "Database path/hash calculation:",
            "Directory creation/check:",
            "WAL journal_mode PRAGMA:",
            "Schema check: source_files:",
            "Schema create/check: ems_records:",
            "Schema create/check: ems_record_bounds:",
            "Source-file CRS migration/check:",
            "No ems_records migration or scan",
            "ANALYZE: not performed",
            "WAL checkpoint: not performed",
            "TOTAL EMSDatabase startup:",
            "PRAGMA journal_mode",
            "PRAGMA synchronous",
            "PRAGMA temp_store",
            "PRAGMA cache_size",
            "PRAGMA mmap_size",
            "PRAGMA page_size",
            "PRAGMA wal_autocheckpoint",
        ):
            self.assertIn(expected, profile)

        for index_name in (
            "idx_ems_datetime",
            "idx_ems_source_datetime",
            "idx_ems_source_vessel",
            "idx_ems_dredge_datetime",
        ):
            self.assertIn(
                f"{index_name}: existed before startup: NO", profile
            )
            self.assertIn(
                f"{index_name} CREATE INDEX IF NOT EXISTS time:", profile
            )

        existing_profile_messages = []
        EMSDatabase(self.source, database_path, log=existing_profile_messages.append)
        for index_name in (
            "idx_ems_datetime",
            "idx_ems_source_datetime",
            "idx_ems_source_vessel",
            "idx_ems_dredge_datetime",
        ):
            self.assertIn(
                f"{index_name}: existed before startup: YES",
                existing_profile_messages[0],
            )

    def test_configured_database_is_used_and_missing_path_does_not_create_fallback(self):
        config_path = self.root / "shared" / "config.json"
        configured_database = self.root / "portable" / "EMS_Master.sqlite3"
        configured_database.parent.mkdir()
        EMSDatabase(self.source, configured_database)

        with patch.object(ems_database, "shared_config_path", return_value=config_path):
            ems_database.set_configured_database_path(configured_database)
            database = EMSDatabase(self.source)
            self.assertEqual(database.path, configured_database.resolve())

            missing_path = self.root / "external" / "EMS_Master.sqlite3"
            ems_database.set_configured_database_path(missing_path)
            with self.assertRaises(ems_database.EMSDatabaseUnavailableError):
                EMSDatabase(self.source)
            self.assertFalse(ems_database.database_path_for_source(self.source).exists())

    def test_database_validation_rejects_unrelated_sqlite_file(self):
        unrelated = self.root / "unrelated.sqlite3"
        connection = sqlite3.connect(unrelated)
        connection.execute("CREATE TABLE unrelated (id INTEGER)")
        connection.commit()
        connection.close()
        original = unrelated.read_bytes()

        with self.assertRaisesRegex(ValueError, "Not an EMS database"):
            ems_database.validate_database(unrelated)
        with self.assertRaisesRegex(ValueError, "Not an EMS database"):
            EMSDatabase(self.source, unrelated)
        self.assertEqual(unrelated.read_bytes(), original)

    def test_verified_database_copy_retains_source_and_metadata(self):
        source_database = self.root / "source-index.sqlite3"
        database = EMSDatabase(self.source, source_database)
        database.sync_files([self.source], self.profile, self._parser)
        destination = self.root / "copied-index.sqlite3"

        copied = ems_database.copy_database_verified(source_database, destination)

        self.assertTrue(source_database.is_file())
        self.assertTrue(destination.is_file())
        self.assertEqual(copied.source_file_count, 1)
        self.assertEqual(copied.source_roots, (str(self.database.source_root),))
        self.assertEqual(copied.profiles, (self.profile,))

    def test_import_counts_rtree_and_file_timing(self):
        messages = []
        result = self.database.sync_files(
            [self.source], self.profile, self._parser, log=messages.append
        )
        self.assertEqual((result.imported, result.failed, result.reused), (1, 0, 0))
        self.assertEqual(self.database.record_count(self.profile), 3)
        self.assertEqual(self._source_file_counts(), (3, 3))
        self.assertTrue(
            any("EMS indexing performance - source.csv" in message for message in messages)
        )
        self.assertTrue(
            any(
                any(
                    line.startswith("Rows:")
                    and line.rsplit(" ", 1)[-1] == "3"
                    for line in message.splitlines()
                )
                for message in messages
            )
        )
        self.assertTrue(any("Overall rate:" in message for message in messages))
        self.assertTrue(any("File read:" in message for message in messages))
        self.assertTrue(any("Vector conversion:" in message for message in messages))
        self.assertTrue(
            any("SQLite ems_records insert:" in message for message in messages)
        )
        self.assertTrue(any("R-tree creation:" in message for message in messages))
        for timing_label in (
            "SQLite file cleanup:",
            "SQLite source metadata:",
            "SQLite connection setup:",
            "SQLite commit:",
            "SQLite connection close:",
        ):
            self.assertTrue(
                any(timing_label in message for message in messages),
                f"Missing timing category: {timing_label}",
            )
        self.assertTrue(any("EMS sync time:" in message for message in messages))

    def test_row_progress_is_logged_with_filename(self):
        messages = []
        with patch.object(ems_database, "SYNC_PROGRESS_INTERVAL", 2):
            self.database.sync_files(
                [self.source], self.profile, self._parser, log=messages.append
            )
        self.assertTrue(
            any("source.csv" in message and "2 rows processed" in message
                for message in messages)
        )

    def test_cache_uses_safe_wal_pragmas(self):
        connection = self.database._connect()
        try:
            journal_mode = connection.execute(
                "PRAGMA journal_mode"
            ).fetchone()[0]
            synchronous = connection.execute(
                "PRAGMA synchronous"
            ).fetchone()[0]
            temp_store = connection.execute(
                "PRAGMA temp_store"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(journal_mode.lower(), "wal")
        self.assertEqual(synchronous, 1)
        self.assertEqual(temp_store, 2)

    def test_import_page_cache_is_bounded_and_connection_local(self):
        import_connection = self.database._connect(cache_size_kib=64 * 1024)
        default_connection = self.database._connect()
        try:
            import_cache = import_connection.execute(
                "PRAGMA cache_size"
            ).fetchone()[0]
            default_cache = default_connection.execute(
                "PRAGMA cache_size"
            ).fetchone()[0]
        finally:
            import_connection.close()
            default_connection.close()
        self.assertEqual(import_cache, -(64 * 1024))
        self.assertNotEqual(default_cache, import_cache)

    def test_unchanged_ready_file_is_reused(self):
        self.database.sync_files([self.source], self.profile, self._parser)
        result = self.database.sync_files([self.source], self.profile, self._parser)
        self.assertEqual((result.imported, result.reused), (0, 1))
        self.assertEqual(self.database.record_count(self.profile), 3)

    def test_changed_file_is_refreshed(self):
        self.database.sync_files([self.source], self.profile, self._parser)
        self.source.write_text("changed source size", encoding="utf-8")
        result = self.database.sync_files([self.source], self.profile, self._parser)
        self.assertEqual((result.imported, result.reused), (1, 0))
        self.assertEqual(self.database.record_count(self.profile), 3)

    def test_failed_file_is_retried_and_failed_attempt_leaves_no_records(self):
        def broken_parser(_path):
            yield self.records[0]
            raise ValueError("malformed row")

        result = self.database.sync_files(
            [self.source], self.profile, broken_parser
        )
        self.assertEqual(result.failed, 1)
        self.assertEqual(self.database.record_count(self.profile), 0)
        self.assertEqual(self._source_file_counts(), (0, 0))

        result = self.database.sync_files([self.source], self.profile, self._parser)
        self.assertEqual(result.imported, 1)
        self.assertEqual(self.database.record_count(self.profile), 3)

    def test_cancel_after_insert_batch_rolls_back_file(self):
        cancel_event = Event()
        def cancel_parser(_path):
            cancel_event.set()
            yield from self.records

        with patch.object(ems_database, "INSERT_BATCH_SIZE", 2):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.database.sync_files(
                    [self.source],
                    self.profile,
                    cancel_parser,
                    cancel_event=cancel_event,
                )
        self.assertEqual(self.database.record_count(self.profile), 0)
        self.assertEqual(self._source_file_counts(), (0, 0))

    def test_failed_file_can_be_explicitly_skipped(self):
        def broken_parser(_path):
            raise ValueError("malformed row")
            yield

        self.database.sync_files([self.source], self.profile, broken_parser)
        result = self.database.sync_files(
            [self.source],
            self.profile,
            self._parser,
            retry_failed=False,
        )
        self.assertEqual((result.failed, result.imported), (1, 0))

    def test_date_dredge_filters_match_expected_records(self):
        self.database.sync_files([self.source], self.profile, self._parser)
        start = "2024-01-01 00:00:00.000000"
        end = "2024-01-01 00:00:02.000000"
        equal = list(
            self.database.iter_query_records(
                self.profile,
                start=start,
                end=end,
                dredge_value=3,
                dredge_mode="equals",
            )
        )
        greater = list(
            self.database.iter_query_records(
                self.profile,
                start=start,
                end=end,
                dredge_value=3,
                dredge_mode="greater_or_equal",
            )
        )
        dated = list(
            self.database.iter_query_records(
                self.profile,
                start="2024-01-01 00:00:02.000000",
                end="2024-01-01 00:00:02.000000",
            )
        )
        self.assertEqual([(r["row_number"], r["dredge"]) for r in equal], [(1, "3")])
        self.assertEqual([record["row_number"] for record in dated], [2])
        self.assertEqual(
            [(r["row_number"], r["dredge"]) for r in greater],
            [(1, "3"), (2, "4")],
        )

    def test_count_diagnostic_matches_stream_filters_without_ordering(self):
        self.database.sync_files([self.source], self.profile, self._parser)
        options = {
            "start": "2024-01-01 00:00:01.000000",
            "end": "2024-01-01 00:00:02.000000",
            "dredge_value": 3,
            "dredge_text_value": "3",
            "coordinates_only": True,
        }
        messages = []
        count, elapsed = self.database.count_query_records(
            self.profile, **options, log=messages.append
        )
        streamed = list(self.database.iter_query_records(self.profile, **options))

        self.assertEqual(count, len(streamed))
        self.assertEqual(count, 1)
        self.assertGreaterEqual(elapsed, 0)
        report = "\n".join(messages)
        self.assertIn("SELECT COUNT(*)", report)
        self.assertNotIn("ORDER BY", report)
        self.assertIn("1 matching rows", report)

    def test_simplified_vessel_order_preserves_time_bin_samples(self):
        second_source = self.root / "second.csv"
        second_source.write_text("second source", encoding="utf-8")
        records_by_file = {
            self.source.name: [
                EMSRecord("2024-01-01 00:00:01.000000", 1, 1, "3", "alpha", "EPSG:4326"),
                EMSRecord("2024-01-01 00:00:20.000000", 2, 2, "3", "alpha", "EPSG:4326"),
                EMSRecord("2024-01-01 00:00:05.000000", 3, 3, "3", None, "EPSG:4326"),
                EMSRecord("2024-01-01 00:00:10.000000", 4, 4, "3", "", "EPSG:4326"),
                EMSRecord("2024-01-01 00:00:02.000000", 5, 5, "3", "beta", "EPSG:4326"),
            ],
            second_source.name: [
                EMSRecord("2024-01-01 00:00:01.000000", 6, 6, "3", "alpha", "EPSG:4326"),
                EMSRecord("2024-01-01 00:00:30.000000", 7, 7, "3", None, "EPSG:4326"),
                EMSRecord("2024-01-01 00:00:40.000000", 8, 8, "3", "", "EPSG:4326"),
                EMSRecord("2024-01-01 00:01:01.000000", 9, 9, "3", "beta", "EPSG:4326"),
            ],
        }

        self.database.sync_files(
            [self.source, second_source],
            self.profile,
            lambda path: iter(records_by_file[Path(path).name]),
        )

        def sampled_keys(simplify):
            state = [object(), None]
            selected = []
            rows = self.database.iter_query_records(
                self.profile,
                order_by_vessel_time=True,
                simplify_vessel_order=simplify,
            )
            for record in rows:
                timestamp = pd.Timestamp(datetime.fromisoformat(record["timestamp"]))
                if validator._sampled_record_is_in_bounds(
                    record, timestamp, None, 60, state
                ):
                    selected.append((record["source_path"], record["row_number"]))
            return selected

        normal_order = list(
            self.database.iter_query_records(
                self.profile, order_by_vessel_time=True
            )
        )
        simple_order = list(
            self.database.iter_query_records(
                self.profile,
                order_by_vessel_time=True,
                simplify_vessel_order=True,
            )
        )
        self.assertIsNone(normal_order[0]["vessel"])
        self.assertIsNone(simple_order[0]["vessel"])
        self.assertEqual(
            {row["vessel"] for row in normal_order},
            {"", "alpha", "beta", None},
        )
        self.assertEqual(
            [
                (row["vessel"], row["timestamp"], row["source_path"], row["row_number"])
                for row in normal_order
            ],
            [
                (row["vessel"], row["timestamp"], row["source_path"], row["row_number"])
                for row in simple_order
            ],
        )
        self.assertEqual(
            set(sampled_keys(False)),
            set(sampled_keys(True)),
        )
        self.assertEqual(len(sampled_keys(False)), len(sampled_keys(True)))
        selected_keys = set(sampled_keys(False))
        selected_names = {(Path(path).name, row_number) for path, row_number in selected_keys}
        self.assertIn((second_source.name, 1), selected_names)
        self.assertNotIn((self.source.name, 1), selected_names)

    def test_candidate_ordering_index_plans_on_temporary_database(self):
        benchmark_records = [
            EMSRecord(
                f"2024-01-01 00:{index // 60:02d}:{index % 60:02d}.000000",
                float(index),
                float(index),
                "3",
                ("", None, "alpha", "beta")[index % 4],
                "EPSG:4326",
            )
            for index in range(2_000)
        ]
        self.database.sync_files(
            [self.source], self.profile, lambda _path: iter(benchmark_records)
        )
        options = {
            "start": "2024-01-01 00:00:00.000000",
            "end": "2024-01-01 00:33:19.000000",
            "dredge_value": 3,
            "dredge_text_value": "3",
            "coordinates_only": True,
            "order_by_vessel_time": True,
        }
        connection = self.database._connect()
        connection.create_function(
            "casefold",
            1,
            lambda value: "" if value is None else str(value).strip().casefold(),
        )
        try:
            query, values, _ = self.database._build_query_records_sql(
                self.profile, **options
            )
            simple_query, simple_values, _ = self.database._build_query_records_sql(
                self.profile, **options, simplify_vessel_order=True
            )
            ready_paths = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT source_path FROM source_files
                    WHERE source_root = ? AND profile = ? AND status = 'ready'
                    """,
                    (str(self.database.source_root), self.profile),
                )
            ]
            no_join_query = simple_query.replace(
                """FROM ems_records AS r
            JOIN source_files AS f
              ON f.source_root = r.source_root
             AND f.source_path = r.source_path
             AND f.profile = r.profile""",
                "FROM ems_records AS r",
            )
            no_join_query = no_join_query.replace(
                "f.status = 'ready' AND ", ""
            )
            no_join_query, no_join_order = no_join_query.rsplit("\nORDER BY", 1)
            path_filter = (
                " AND r.source_path IN ("
                + ", ".join("?" for _ in ready_paths)
                + ")"
            )
            no_join_query += path_filter + "\nORDER BY" + no_join_order
            no_join_values = [*simple_values, *ready_paths]
            existing_plans = {
                "existing order": [
                    str(row[3])
                    for row in connection.execute(
                        "EXPLAIN QUERY PLAN " + query, values
                    )
                ],
                "simplified order": [
                    str(row[3])
                    for row in connection.execute(
                        "EXPLAIN QUERY PLAN " + simple_query, simple_values
                    )
                ],
                "ready-path filter candidate": [
                    str(row[3])
                    for row in connection.execute(
                        "EXPLAIN QUERY PLAN " + no_join_query, no_join_values
                    )
                ],
            }
            storage_before = sum(
                path.stat().st_size
                for path in self.root.glob("index.sqlite3*")
                if path.exists()
            )
            index_started = time.perf_counter()
            connection.execute(
                """
                CREATE INDEX idx_test_order_vessel_time
                ON ems_records
                    (source_root, profile, vessel, timestamp, source_path, row_number)
                """
            )
            connection.execute(
                """
                CREATE INDEX idx_test_order_dredge_vessel_time
                ON ems_records
                    (source_root, profile, dredge_numeric, vessel, timestamp,
                     source_path, row_number)
                """
            )
            connection.execute("ANALYZE")
            connection.commit()
            index_elapsed = time.perf_counter() - index_started
            storage_after = sum(
                path.stat().st_size
                for path in self.root.glob("index.sqlite3*")
                if path.exists()
            )
            plans = {}
            for label, sql, parameters in (
                ("existing order with candidates", query, values),
                ("simplified order with candidates", simple_query, simple_values),
                (
                    "ready paths with forced vessel candidate",
                    no_join_query.replace(
                        "FROM ems_records AS r",
                        "FROM ems_records AS r INDEXED BY idx_test_order_vessel_time",
                    ),
                    no_join_values,
                ),
            ):
                plans[label] = [
                    str(row[3])
                    for row in connection.execute(
                        "EXPLAIN QUERY PLAN " + sql, parameters
                    )
                ]
        finally:
            connection.close()

        filters = {key: value for key, value in options.items() if key != "order_by_vessel_time"}
        count, count_elapsed = self.database.count_query_records(
            self.profile, **filters
        )
        timings = {}
        for label, simplify in (
            ("existing order", False),
            ("simplified order", True),
        ):
            started = time.perf_counter()
            iterator = self.database.iter_query_records(
                self.profile,
                **options,
                simplify_vessel_order=simplify,
            )
            first = next(iterator, None)
            first_elapsed = time.perf_counter() - started
            remaining = sum(1 for _ in iterator)
            timings[label] = (first_elapsed, remaining + (first is not None))

        print(f"Temporary candidate index build: {index_elapsed:.4f} sec.")
        print(
            "Temporary candidate index storage increase: "
            f"{storage_after - storage_before:,} bytes."
        )
        print(f"Temporary benchmark matching COUNT: {count:,} in {count_elapsed:.4f} sec.")
        for label, plan in (*existing_plans.items(), *plans.items()):
            print(f"Temporary benchmark {label} plan: {' | '.join(plan)}")
        for label, (first_elapsed, row_count) in timings.items():
            print(
                f"Temporary benchmark {label}: first row {first_elapsed:.4f} sec; "
                f"{row_count:,} rows total."
            )
        self.assertEqual(count, 2_000)
        self.assertEqual(timings["existing order"][1], count)
        self.assertEqual(timings["simplified order"][1], count)

    def test_query_progress_handler_cancels_and_connection_remains_usable(self):
        many_records = [
            EMSRecord(
                f"2024-01-01 00:{index // 60:02d}:{index % 60:02d}.000000",
                float(index),
                float(index),
                "3",
                "vessel",
                "EPSG:4326",
            )
            for index in range(2_000)
        ]
        self.database.sync_files(
            [self.source], self.profile, lambda _path: iter(many_records)
        )
        cancel_event = Event()
        cancel_event.set()
        records = self.database.iter_query_records(
            self.profile,
            order_by_vessel_time=True,
            cancel_event=cancel_event,
        )
        with self.assertRaisesRegex(RuntimeError, "query cancelled"):
            next(records)
        self.assertEqual(self.database.record_count(self.profile), 2_000)

    def test_source_crs_query_logs_sql_and_explain_plan(self):
        self.database.sync_files([self.source], self.profile, self._parser)
        messages = []

        values = self.database.source_crs_values(
            self.profile,
            log=messages.append,
            diagnostic_label="source_crs_values call #1",
        )

        self.assertEqual(values, ["EPSG:4326"])
        report = "\n".join(messages)
        self.assertIn("SELECT DISTINCT crs FROM source_files", report)
        self.assertNotIn("FROM ems_records", report)
        self.assertIn("EXPLAIN QUERY PLAN completed", report)
        self.assertIn("query plan:", report)
        self.assertIn("temporary B-tree for DISTINCT=", report)
        self.assertNotIn("EMS record:", report)

    def test_configured_crs_resolution_does_not_query_ems_records(self):
        messages = []
        with patch.object(
            self.database,
            "source_crs_values",
            side_effect=AssertionError("metadata query should not be needed"),
        ):
            values = self.database.resolve_crs_values(
                self.profile,
                configured_crs="EPSG:4326",
                log=messages.append,
            )

        self.assertEqual(values, ["EPSG:4326"])
        self.assertIn("EMS CRS source: profile configuration", "\n".join(messages))

    def test_existing_source_files_rows_can_be_backfilled_without_reimport(self):
        self.database.sync_files(
            [self.source], self.profile, self._parser, profile_crs="EPSG:4326"
        )
        with self.database._connection() as connection:
            connection.execute("UPDATE source_files SET crs = NULL")

        result = self.database.sync_files(
            [self.source], self.profile, self._parser, profile_crs="EPSG:4326"
        )

        self.assertEqual((result.imported, result.reused), (0, 1))
        self.assertEqual(self.database.record_count(self.profile), 3)
        self.assertEqual(self.database.source_crs_values(self.profile), ["EPSG:4326"])

    def test_legacy_source_files_without_crs_column_are_migrated_without_record_scan(self):
        self.database.sync_files([self.source], self.profile, self._parser)
        with self.database._connection() as connection:
            connection.execute("ALTER TABLE source_files DROP COLUMN crs")

        migrated = EMSDatabase(self.source, self.database.path)
        result = migrated.sync_files(
            [self.source], self.profile, self._parser, profile_crs="EPSG:4326"
        )

        self.assertEqual((result.imported, result.reused), (0, 1))
        self.assertEqual(migrated.record_count(self.profile), 3)
        self.assertEqual(
            migrated.source_crs_values(self.profile), ["EPSG:4326"]
        )

    def test_validator_resolves_crs_once_and_reuses_it_for_bounds(self):
        source = self.root / "validator-crs.csv"
        source.write_text(
            "DATE,TIME,LONGITUDE,LATITUDE,DREDGE,VESSEL\n"
            "04 Jan 2024,00:00:00,1,2,3,vessel-1\n",
            encoding="utf-8",
        )
        database = EMSDatabase(source, self.root / "validator-crs.sqlite3")
        settings = SimpleNamespace(
            ems_source=str(source),
            ems_x_field="LONGITUDE",
            ems_y_field="LATITUDE",
            ems_datetime_field="DATETIME",
            ems_time_field="TIME",
            ems_dredge_field="DREDGE",
            ems_vessel_field="VESSEL",
            ems_dredge_value="3",
            ems_dredge_mode="equals",
            survey_start="04/01/2024",
            survey_end="04/01/2024",
            sample_interval_minutes=1,
        )
        grid = SimpleNamespace(
            crs="EPSG:27700",
            total_bounds=(0.0, 0.0, 10.0, 10.0),
        )

        with (
            patch.object(validator, "EMSDatabase", return_value=database),
            patch.object(
                database,
                "resolve_crs_values",
                wraps=database.resolve_crs_values,
            ) as crs_resolver,
            patch.object(
                database,
                "source_crs_values",
                wraps=database.source_crs_values,
            ) as source_crs_lookup,
            patch.object(
                validator.Transformer,
                "from_crs",
                return_value=SimpleNamespace(
                    transform_bounds=lambda *args, **kwargs: (1, 2, 3, 4)
                ),
            ) as from_crs,
        ):
            messages = []
            result = validator.load_ems(
                settings,
                messages.append,
                grid,
                spatial_first_experiment=True,
            )

        crs_resolver.assert_called_once()
        source_crs_lookup.assert_not_called()
        from_crs.assert_called_once_with(
            "EPSG:27700", "EPSG:4326", always_xy=True
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.crs.to_string(), "EPSG:4326")
        self.assertEqual(result.geometry.iloc[0].x, 1.0)
        self.assertEqual(result.geometry.iloc[0].y, 2.0)
        self.assertIn("EMS CRS source: profile configuration", "\n".join(messages))
        self.assertIn("Starting indexed EMS query...", "\n".join(messages))
        self.assertIn("Waiting for first matching EMS record...", "\n".join(messages))
        self.assertIn("Spatial-first output comparison:", "\n".join(messages))
        self.assertIn("R-tree virtual-table search=YES", "\n".join(messages))

    def test_spatial_first_prefilter_can_change_time_bin_sample(self):
        records = [
            EMSRecord("2024-01-01 12:00:00.000000", 10, 10, "3", "vessel", "EPSG:4326"),
            EMSRecord("2024-01-01 12:00:10.000000", 2, 2, "3", "vessel", "EPSG:4326"),
            EMSRecord("2024-01-01 12:00:20.000000", 3, 3, "3", "vessel", "EPSG:4326"),
            EMSRecord("2024-01-01 12:01:05.000000", 4, 4, "3", "vessel", "EPSG:4326"),
        ]
        self.database.sync_files(
            [self.source], self.profile, lambda _path: iter(records)
        )
        bounds = {"EPSG:4326": (0, 0, 5, 5)}
        options = {
            "start": None,
            "end": None,
            "dredge_value": None,
            "dredge_text_value": None,
            "dredge_mode": "equals",
            "coordinates_only": True,
            "exact_bounds_by_crs": bounds,
            "query_bounds_by_crs": None,
            "seconds": 60,
            "log": lambda _message: None,
        }
        current = validator._query_sampled_ems(
            self.database,
            self.profile,
            **options,
            label="Current pathway",
        )
        spatial = validator._query_sampled_ems(
            self.database,
            self.profile,
            **{**options, "query_bounds_by_crs": bounds},
            label="Spatial-first pathway",
        )

        current_rows = {
            (Path(row["source_path"]).name, row["row_number"])
            for row in current["retained"]
        }
        spatial_rows = {
            (Path(row["source_path"]).name, row["row_number"])
            for row in spatial["retained"]
        }
        self.assertNotIn((self.source.name, 2), current_rows)
        self.assertIn((self.source.name, 2), spatial_rows)
        self.assertIn((self.source.name, 4), current_rows)
        self.assertIn((self.source.name, 4), spatial_rows)

    def test_spatial_first_query_forces_rtree_search_and_preserves_ready_filter(self):
        ready_source = self.root / "ready.csv"
        ready_source.write_text("ready source", encoding="utf-8")
        self.database.sync_files(
            [self.source, ready_source], self.profile, self._parser
        )
        with self.database._connection() as connection:
            connection.execute(
                """
                UPDATE source_files SET status = 'failed'
                WHERE source_path = ?
                """,
                (str(self.source.resolve()),),
            )
        messages = []
        records = list(
            self.database.iter_query_records(
                self.profile,
                bounds_by_crs={"EPSG:4326": (0, 0, 2.5, 3.5)},
                coordinates_only=True,
                order_by_vessel_time=True,
                spatial_first=True,
                diagnostic_label="spatial-first test",
                log=messages.append,
            )
        )

        self.assertEqual([row["row_number"] for row in records], [1, 2])
        self.assertEqual(
            {Path(row["source_path"]).name for row in records},
            {ready_source.name},
        )
        report = "\n".join(messages)
        self.assertIn("FROM ems_record_bounds AS b", report)
        self.assertIn("CROSS JOIN ems_records AS r ON b.id = r.id", report)
        self.assertIn("VIRTUAL TABLE INDEX", report)
        self.assertIn("R-tree virtual-table search=YES", report)
        self.assertIn("f.status = 'ready'", report)
        self.assertIn("temporary B-tree for ORDER BY=YES", report)

    def test_iter_query_logs_filtered_sql_plan_without_logging_records(self):
        self.database.sync_files([self.source], self.profile, self._parser)
        messages = []

        records = list(
            self.database.iter_query_records(
                self.profile,
                start="2024-01-01 00:00:00.000000",
                end="2024-01-01 00:00:02.000000",
                dredge_value=3,
                dredge_text_value="3",
                coordinates_only=True,
                order_by_vessel_time=True,
                log=messages.append,
            )
        )

        self.assertEqual([row["row_number"] for row in records], [1])
        report = "\n".join(messages)
        self.assertIn("r.timestamp >= ?", report)
        self.assertIn("r.timestamp <= ?", report)
        self.assertIn("r.dredge_numeric = ?", report)
        self.assertIn("r.x IS NOT NULL", report)
        self.assertIn("ORDER BY r.vessel IS NOT NULL", report)
        self.assertIn("EXPLAIN QUERY PLAN completed", report)
        self.assertIn("temporary B-tree for ORDER BY=", report)
        self.assertIn("SQL execution returned cursor in", report)
        self.assertNotIn("EMS record:", report)

    def test_bounds_filter_uses_only_corresponding_rtree_records(self):
        self.database.sync_files([self.source], self.profile, self._parser)
        results = list(
            self.database.iter_query_records(
                self.profile,
                bounds_by_crs={"EPSG:4326": (0, 0, 2.5, 3.5)},
            )
        )
        self.assertEqual([record["row_number"] for record in results], [1, 2])

    def test_spatial_filter_does_not_change_vessel_time_sampling(self):
        bounds = {"EPSG:4326": (0, 0, 5, 5)}
        state = [object(), None]
        outside_first = {
            "vessel": "v1",
            "crs": "EPSG:4326",
            "x": 10.0,
            "y": 10.0,
        }
        inside_same_bin = {
            "vessel": "v1",
            "crs": "EPSG:4326",
            "x": 2.0,
            "y": 2.0,
        }
        inside_next_bin = {
            "vessel": "v1",
            "crs": "EPSG:4326",
            "x": 2.0,
            "y": 2.0,
        }
        self.assertFalse(
            validator._sampled_record_is_in_bounds(
                outside_first,
                pd.Timestamp(datetime.fromisoformat("2024-01-01 00:00:01")),
                bounds,
                60,
                state,
            )
        )
        self.assertFalse(
            validator._sampled_record_is_in_bounds(
                inside_same_bin,
                pd.Timestamp(datetime.fromisoformat("2024-01-01 00:00:30")),
                bounds,
                60,
                state,
            )
        )
        self.assertTrue(
            validator._sampled_record_is_in_bounds(
                inside_next_bin,
                pd.Timestamp(datetime.fromisoformat("2024-01-01 00:01:01")),
                bounds,
                60,
                state,
            )
        )

    def test_bulk_import_matches_emsrecord_import(self):
        source = self.root / "bulk.csv"
        source.write_text(
            "GPSLON,GPSLAT,DATETIME,DREDGE,VESSEL\n"
            "1,2,01/02/2024 12:00:00,3,V1\n"
            "3,4,2024-02-01T12:00:01,3.5,\n"
            "5,6,2024-02-01T12:00:02+01:00,,V3\n"
            "7,8,2024-02-01T12:00:03+02:00,3,V4\n"
            "9,10,, ,V5\n"
            "invalid,6,2024-02-01 12:00:02,,V3\n",
            encoding="utf-8",
        )
        settings = SimpleNamespace(
            ems_x_field="GPSLON",
            ems_y_field="GPSLAT",
            ems_datetime_field="DATETIME",
            ems_time_field="TIME",
            ems_dredge_field="DREDGE",
            ems_vessel_field="VESSEL",
        )
        legacy_profile = "legacy-profile"
        bulk_profile = "bulk-profile"
        bulk_log = []
        with (
            patch.object(validator, "EMS_READ_CHUNK_SIZE", 2),
            patch.object(validator, "INSERT_BATCH_SIZE", 2),
        ):
            self.database.sync_files(
                [source],
                legacy_profile,
                lambda path: validator._iter_validator_delimited_records(
                    path, settings
                ),
            )
            self.database.sync_files(
                [source],
                bulk_profile,
                lambda path: (),
                bulk_parser=lambda path, root, full_path, profile: (
                    validator._iter_validator_delimited_batches(
                        path, settings, root, full_path, profile
                    )
                ),
                log=bulk_log.append,
            )
        performance_report = next(
            message
            for message in bulk_log
            if "EMS indexing performance - bulk.csv" in message
        )
        for timing_label in (
            "X conversion:",
            "Y conversion:",
            "Datetime source assembly:",
            "Datetime parsing:",
            "Datetime formatting:",
            "Dredge text preparation:",
            "Dredge numeric conversion:",
            "Vessel preparation:",
            "Validity/mask processing:",
            "NumPy conversion:",
            "Other vector work:",
        ):
            self.assertIn(timing_label, performance_report)

        connection = self.database._connect()
        try:
            fields = (
                "row_number, timestamp, x, y, dredge, dredge_numeric, vessel, crs"
            )
            legacy_rows = connection.execute(
                f"SELECT {fields} FROM ems_records WHERE profile = ? ORDER BY row_number",
                (legacy_profile,),
            ).fetchall()
            bulk_rows = connection.execute(
                f"SELECT {fields} FROM ems_records WHERE profile = ? ORDER BY row_number",
                (bulk_profile,),
            ).fetchall()
        finally:
            connection.close()
        self.assertEqual(
            [tuple(row) for row in bulk_rows],
            [tuple(row) for row in legacy_rows],
        )
        self.assertEqual(len(bulk_rows), 5)
        self.assertEqual(self.database.record_count(legacy_profile), 5)
        self.assertEqual(self.database.record_count(bulk_profile), 5)

    def test_bulk_timestamp_format_preserves_uniform_timezone_offsets(self):
        source = self.root / "timezone.csv"
        source.write_text(
            "GPSLON,GPSLAT,DATETIME,DREDGE\n"
            "1,2,2024-02-01T12:00:00+01:00,3\n"
            "3,4,2024-02-01T12:00:01+01:00,3\n",
            encoding="utf-8",
        )
        settings = SimpleNamespace(
            ems_x_field="GPSLON",
            ems_y_field="GPSLAT",
            ems_datetime_field="DATETIME",
            ems_time_field="TIME",
            ems_dredge_field="DREDGE",
            ems_vessel_field="VESSEL",
        )
        rows = [
            row
            for batch in validator._iter_validator_delimited_batches(
                source, settings, "root", str(source), "profile"
            )
            for row in batch.rows
        ]
        self.assertEqual(
            [row[4] for row in rows],
            [
                "2024-02-01 12:00:00.000000+01:00",
                "2024-02-01 12:00:01.000000+01:00",
            ],
        )

    def test_split_ems_date_time_format_matches_flexible_parser(self):
        source = self.root / "split_datetime.csv"
        source.write_text(
            "GPSLON,GPSLAT,DATE,TIME,DREDGE\n"
            "1,2,04 Jan 2026,00:00:00,3\n"
            "3,4,04 Jan 2026,00:00:10,3\n"
            "5,6,31 Dec 2026,23:59:59,3\n"
            "7,8,01 Jan 2027,00:00:00,3\n"
            "9,10,not-a-date,25:61:99,3\n"
            "11,12,31 Feb 2027,00:00:00,3\n"
            "13,14,21 Sep 1677,00:12:43,3\n"
            "15,16,11 Apr 2262,23:47:17,3\n",
            encoding="utf-8",
        )
        settings = SimpleNamespace(
            ems_x_field="GPSLON",
            ems_y_field="GPSLAT",
            ems_datetime_field="DATETIME",
            ems_time_field="TIME",
            ems_dredge_field="DREDGE",
            ems_vessel_field="VESSEL",
        )
        with patch.object(
            validator.pd,
            "to_datetime",
            side_effect=AssertionError("split EMS path called pandas datetime parser"),
        ):
            bulk_rows = [
                row
                for batch in validator._iter_validator_delimited_batches(
                    source, settings, "root", str(source), "bulk-profile"
                )
                for row in batch.rows
            ]
        frame = pd.read_csv(source)
        legacy_values = pd.to_datetime(
            validator._ems_datetime_values(frame, settings),
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        expected_timestamps = [
            None
            if pd.isna(timestamp)
            else timestamp.isoformat(sep=" ", timespec="microseconds")
            for timestamp in legacy_values
        ]
        self.assertEqual([row[4] for row in bulk_rows], expected_timestamps)
        self.assertEqual(
            [row[4] for row in bulk_rows],
            [
                "2026-01-04 00:00:00.000000",
                "2026-01-04 00:00:10.000000",
                "2026-12-31 23:59:59.000000",
                "2027-01-01 00:00:00.000000",
                None,
                None,
                None,
                None,
            ],
        )


if __name__ == "__main__":
    unittest.main()
