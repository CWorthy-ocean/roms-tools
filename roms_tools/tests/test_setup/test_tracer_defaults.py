from pathlib import Path

import pytest

from roms_tools.setup.utils import _load_tracer_defaults, get_tracer_defaults


@pytest.fixture(autouse=True)
def clear_tracer_defaults_cache():
    _load_tracer_defaults.cache_clear()
    yield
    _load_tracer_defaults.cache_clear()


class TestTracerDefaults:
    def test_reads_recommended_values_from_bundled_csv(self, monkeypatch):
        bundled = (
            Path(__file__).resolve().parents[2]
            / "datasets"
            / "data"
            / "river_tracer_defaults.csv"
        )
        monkeypatch.setattr(
            "roms_tools.setup.utils.download_river_tracer_defaults",
            lambda: str(bundled),
        )

        defaults = get_tracer_defaults()

        assert defaults["DIC"] == pytest.approx(1640.071)
        assert defaults["ALK"] == pytest.approx(1173.693)
        assert defaults["DOC"] == pytest.approx(460.476)
        assert defaults["DON"] == pytest.approx(31.61635)
        assert defaults["DOP"] == pytest.approx(1.0242011)
        assert defaults["NH4"] == 0.0
        assert defaults["spC"] == 0.0
        assert len(defaults) == 32

    def test_ignores_alternate_value_column(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "river_tracer_defaults.csv"
        csv_path.write_text(
            "tracer,units,recommended_value,recommended_citation,alternate_value\n"
            "PO4,mmol/m3,4.2453,rec,99.9\n"
            "NO3,mmol/m3,57.3565,rec,88.8\n"
        )
        monkeypatch.setattr(
            "roms_tools.setup.utils.download_river_tracer_defaults",
            lambda: str(csv_path),
        )

        defaults = get_tracer_defaults()

        assert defaults["PO4"] == pytest.approx(4.2453)
        assert defaults["NO3"] == pytest.approx(57.3565)
