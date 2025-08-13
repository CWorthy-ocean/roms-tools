from pathlib import Path

import numpy as np
import pytest

from roms_tools.utils import _generate_focused_coordinate_range, _path_list_from_input


@pytest.mark.parametrize(
    "min_val, max_val, center, sc, N",
    [
        (-20.0, 5.5, -3.1, 1.0, 100),
        (100.0, 200.0, 150.0, 30.0, 100),
        (0.0, 2000.0, 150.0, 0.0, 100),
        (0.0, 2000.0, 150.0, 30.0, 100),
    ],
)
def test_coordinate_range_monotonicity(min_val, max_val, center, sc, N):
    centers, faces = _generate_focused_coordinate_range(
        min_val=min_val, max_val=max_val, center=center, sc=sc, N=N
    )
    assert np.all(np.diff(faces) > 0), "faces is not strictly increasing"
    assert np.all(np.diff(centers) > 0), "centers is not strictly increasing"


class TestPathListFromInput:
    """A collection of tests for the _path_list_from_input function."""

    # Test cases that don't require I/O
    def test_list_of_strings(self):
        """Test with a list of file paths as strings."""
        files_list = ["path/to/file1.txt", "path/to/file2.txt"]
        result = _path_list_from_input(files_list)
        assert len(result) == 2
        assert result[0] == Path("path/to/file1.txt")
        assert result[1] == Path("path/to/file2.txt")

    def test_list_of_path_objects(self):
        """Test with a list of pathlib.Path objects."""
        files_list = [Path("file_a.txt"), Path("file_b.txt")]
        result = _path_list_from_input(files_list)
        assert len(result) == 2
        assert result[0] == Path("file_a.txt")
        assert result[1] == Path("file_b.txt")

    def test_single_path_object(self):
        """Test with a single pathlib.Path object."""
        file_path = Path("a_single_file.csv")
        result = _path_list_from_input(file_path)
        assert len(result) == 1
        assert result[0] == file_path

    def test_invalid_input_type_raises(self):
        """Test that an invalid input type raises a TypeError."""
        with pytest.raises(TypeError, match="'files' should be str, Path, or List"):
            _path_list_from_input(123)

    # Test cases that require I/O and `tmp_path`
    def test_single_file_as_str(self, tmp_path):
        """Test with a single file given as a string, requiring a file to exist."""
        p = tmp_path / "test_file.txt"
        p.touch()
        result = _path_list_from_input(str(p))
        assert len(result) == 1
        assert result[0] == p

    def test_wildcard_pattern(self, tmp_path, monkeypatch):
        """Test with a wildcard pattern, requiring files to exist, using monkeypatch."""
        # Setup
        d = tmp_path / "data"
        d.mkdir()
        (d / "file1.csv").touch()
        (d / "file2.csv").touch()
        (d / "other_file.txt").touch()

        # Action: Temporarily change the current working directory
        monkeypatch.chdir(tmp_path)

        result = _path_list_from_input("data/*.csv")

        # Assertion
        assert len(result) == 2
        assert result[0].name == "file1.csv"
        assert result[1].name == "file2.csv"

    def test_non_matching_pattern_raises(self, tmp_path):
        """Test that a non-matching pattern raises a FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No files matched"):
            _path_list_from_input(str(tmp_path / "non_existent_file_*.txt"))
