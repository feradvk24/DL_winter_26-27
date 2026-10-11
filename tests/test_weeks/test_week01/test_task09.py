import unittest

import numpy as np
import pytest

from weeks.week01 import task09


class TestScalarAddition(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys
        
    def test_when_called_then_returns_matrix_with_added_scalar(self):
        # Arrange
        matrix = np.array([[1, 2], [3, 4]])
        scalar = 5
        expected_result = np.array([[6, 7], [8, 9]])

        # Act
        result = task09.add_scalar_to_matrix(matrix, scalar)

        # Assert
        np.testing.assert_array_equal(result, expected_result)


class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_main_called_then_prints_matrix_plus_scalar(self):
        # Act
        task09.main()

        # Assert
        captured = self.capsys.readouterr()
        self.assertRegex(captured.out, r"Matrix \+ scalar \(\d+\):")