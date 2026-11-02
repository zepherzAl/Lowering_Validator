import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ems_database import EMSDatabase, EMSRecord, fallback_vessel_name


class FallbackVesselNameTests(unittest.TestCase):
    def test_date_only_file_uses_vessel_folder(self):
        self.assertEqual(
            fallback_vessel_name(
                r"D:\Tracks_2020_mid2026\2022\1\232003506 - (Welsh Piper)\19-01-2022.txt"
            ),
            "232003506 - (Welsh Piper)",
        )
        self.assertEqual(
            fallback_vessel_name("/data/2022/1/232215000 - (Sand Falcon)/19-01-2022.txt"),
            "232215000 - (Sand Falcon)",
        )

    def test_file_named_after_vessel_keeps_file_name(self):
        self.assertEqual(
            fallback_vessel_name(r"D:\Tracks_2020_mid2026\01-01-2024 CGI.txt"),
            "01-01-2024 CGI",
        )


class RepairFallbackVesselTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name) / "tracks"
        self.files = []
        for folder in ("232003506 - (Welsh Piper)", "232215000 - (Sand Falcon)"):
            path = self.root / "2022" / "1" / folder / "19-01-2022.txt"
            path.parent.mkdir(parents=True)
            path.write_text("placeholder", encoding="utf-8")
            self.files.append(path)
        named = self.root / "01-01-2024 CGI.txt"
        named.write_text("placeholder", encoding="utf-8")
        self.files.append(named)
        self.database = EMSDatabase(self.root, Path(self.temp_dir.name) / "index.sqlite3")
        self.profile = "validator:test"

    def _old_style_parser(self, path):
        # Simulates the earlier import, which used the file name as the vessel.
        return iter(
            EMSRecord(
                f"2022-05-01 00:0{minute}:00.000000", 1.0, 51.0, "3", path.stem, "EPSG:4326"
            )
            for minute in range(3)
        )

    def _vessels(self):
        connection = sqlite3.connect(self.database.path)
        try:
            return sorted(
                {row[0] for row in connection.execute("SELECT vessel FROM ems_records")}
            )
        finally:
            connection.close()

    def test_repair_renames_merged_vessels_once(self):
        self.database.sync_files(self.files, self.profile, self._old_style_parser)
        self.assertEqual(self._vessels(), ["01-01-2024 CGI", "19-01-2022"])

        messages = []
        updated = self.database.repair_fallback_vessels(self.profile, log=messages.append)

        self.assertEqual(updated, 6)
        self.assertEqual(
            self._vessels(),
            ["01-01-2024 CGI", "232003506 - (Welsh Piper)", "232215000 - (Sand Falcon)"],
        )
        self.assertTrue(any("one-time update" in message for message in messages))
        self.assertEqual(self.database.repair_fallback_vessels(self.profile), 0)


if __name__ == "__main__":
    unittest.main()
