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
