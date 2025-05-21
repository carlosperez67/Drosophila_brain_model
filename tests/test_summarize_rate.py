import shutil
import tempfile
import unittest
from pathlib import Path
from scripts.summarize_rate import filter_parquets


class TestSummarizeRate(unittest.TestCase):
    """
    Unit-tests for summarize_rates._filter_parquets.
    Uses an in-memory list of Path objects that mimic the directory layout
    produced by run_frequency_sweeps.py.
    """

    def setUp(self):
        # Create a temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create controls and experiments subdirectories
        controls_dir = self.temp_dir / "controls"
        experiments_dir = self.temp_dir / "experiments"
        controls_dir.mkdir()
        experiments_dir.mkdir()

        # Create empty .parquet files
        self._touch(controls_dir / "sugar_only_45Hz.parquet")
        self._touch(controls_dir / "sugar_only_5Hz.parquet")
        self._touch(controls_dir / "test.parquet")

        self._touch(experiments_dir / "sugar45Hz_plus_ORN_D_20Hz.parquet")
        self._touch(experiments_dir / "sugar45Hz_plus_ORN_D_30Hz.parquet")
        self._touch(experiments_dir / "sugar50Hz_plus_ORN_D_20Hz.parquet")
        self._touch(experiments_dir / "sugar45Hz_plus_ORN_D_40Hz.parquet")
        self._touch(experiments_dir / "sugar45Hz_plus_ORN_AB_20Hz.parquet")

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)

    def _touch(self, path: Path):
        path.touch()


    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------
    def test_selects_control_and_correct_experiments(self):
        controls, experiments = filter_parquets(
            self.temp_dir,
            sugar_hz=45,
            orn_hz={20, 30},
            type_sel={"ORN_D"},
        )
        expected_controls = {
            self.temp_dir / "controls/sugar_only_45Hz.parquet"
        }
        expected_experiments = {
            self.temp_dir / "experiments/sugar45Hz_plus_ORN_D_20Hz.parquet",
            self.temp_dir / "experiments/sugar45Hz_plus_ORN_D_30Hz.parquet"
        }

        self.assertEqual(set(controls), expected_controls)
        self.assertEqual(set(experiments), expected_experiments)

    def test_respects_sugar_selection(self):
        controls, experiments = filter_parquets(
            self.temp_dir,
            sugar_hz=50,
            orn_hz={20},
            type_sel={"ORN_D"},
        )
        expected_experiments = {
            self.temp_dir / "experiments/sugar50Hz_plus_ORN_D_20Hz.parquet"
        }
        self.assertEqual(set(experiments), expected_experiments)

    def test_respects_orn_frequency_selection(self):
        controls, experiments = filter_parquets(
            self.temp_dir,
            sugar_hz=45,
            orn_hz={40},
            type_sel={"ORN_D"},
        )
        assert experiments == [self.temp_dir / "experiments/sugar45Hz_plus_ORN_D_40Hz.parquet"]
        assert controls == [self.temp_dir / "controls/sugar_only_45Hz.parquet"]

    def test_type_sel_none_means_all_cell_types(self):
        controls, experiments = filter_parquets(
            self.temp_dir,
            sugar_hz=45,
            orn_hz={20},
            type_sel=None,  # allow all types
        )
        expected_controls = {
            self.temp_dir / "controls/sugar_only_45Hz.parquet",
        }
        expected_experiments = {
            self.temp_dir / "experiments/sugar45Hz_plus_ORN_D_20Hz.parquet",
            self.temp_dir / "experiments/sugar45Hz_plus_ORN_AB_20Hz.parquet",
        }

        assert set(controls) == expected_controls
        assert set(experiments) == expected_experiments