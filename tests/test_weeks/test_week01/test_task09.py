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


class TestAddRowVectorToMatrix(unittest.TestCase):
    def test_when_called_then_returns_matrix_with_added_row_vector(self):
        # Arrange
        matrix = np.array([[1, 2], [3, 4]])
        row_vector = np.array([10, 20])
        expected_result = np.array([[11, 22], [13, 24]])

        # Act
        result = task09.add_row_vector_to_matrix(matrix, row_vector)

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
        output = self.capsys.readouterr().out
        self.assertRegex(output, r"Matrix \+ scalar \(\d+\):")
        self.assertRegex(
            output,
            r"Matrix \(\d+, \d+\) \+ row vector \(\d+,\):",
        )