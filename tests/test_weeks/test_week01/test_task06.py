import unittest

import numpy as np
import pytest

from weeks.week01 import task06


class TestPrintOneDArray(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_array_properties(self):
        # Arrange
        expected_array = np.array([1, 2, 3, 4, 5])
        shape = expected_array.shape
        dtype = expected_array.dtype
        ndim = expected_array.ndim
        size = expected_array.size
        expected_output = (
            f"1D array (vector): shape={shape}, dtype={dtype}, "
            f"ndim={ndim}, size={size}\n"
            f"{expected_array}"
        )

        # Act
        task06.print_one_d_array()

        # Assert
        captured = self.capsys.readouterr()
        self.assertEqual(captured.out.strip(), expected_output)


class TestPrintTwoDArray(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_array_properties(self):
        # Arrange
        expected_array = np.array([[1, 2, 3], [4, 5, 6]])
        shape = expected_array.shape
        dtype = expected_array.dtype
        ndim = expected_array.ndim
        size = expected_array.size
        expected_output = (
            f"2D array (matrix): shape={shape}, dtype={dtype}, "
            f"ndim={ndim}, size={size}\n"
            f"{expected_array}"
        )

        # Act
        task06.print_two_d_array()

        # Assert
        captured = self.capsys.readouterr()
        self.assertEqual(captured.out.strip(), expected_output)


class TestPrintThreeDArray(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_array_properties(self):
        # Arrange
        expected_array = np.arange(24).reshape(2, 3, 4)
        shape = expected_array.shape
        dtype = expected_array.dtype
        ndim = expected_array.ndim
        size = expected_array.size
        expected_output = (
            f"3D array (volume): shape={shape}, dtype={dtype}, "
            f"ndim={ndim}, size={size}\n"
            f"{expected_array}"
        )

        # Act
        task06.print_three_d_array()

        # Assert
        captured = self.capsys.readouterr()
        self.assertEqual(captured.out.strip(), expected_output)


class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_output_from_all_array_functions(self):
        # Arrange
        task06.print_one_d_array()
        expected_one_d_output = self.capsys.readouterr().out

        task06.print_two_d_array()
        expected_two_d_output = self.capsys.readouterr().out

        task06.print_three_d_array()
        expected_three_d_output = self.capsys.readouterr().out

        expected_output = (
            expected_one_d_output
            + expected_two_d_output
            + expected_three_d_output
        )

        # Act
        task06.main()
        actual_output = self.capsys.readouterr().out

        # Assert
        self.assertEqual(actual_output, expected_output)