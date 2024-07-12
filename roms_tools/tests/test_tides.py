import pytest
from roms_tools.setup.tides import TPXO
import os


class TestTPXO:
    def test_load_data_file_not_found(self):
        # Test loading data from a non-existing file
        with pytest.raises(FileNotFoundError):
            TPXO.load_data("non_existing_file.nc")

    def test_load_data_checksum_mismatch(self):
        # Create a temporary file for testing
        filename = "test_tidal_data.nc"
        with open(filename, "wb") as file:
            # Write some data to the file
            file.write(b"test data")
        # Test loading data with incorrect checksum
        with open(filename, "wb") as file:
            with pytest.raises(ValueError):
                TPXO.load_data(filename)
        # Remove temporary file
        os.remove(filename)
