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


class TestAddVectorToMatrix(unittest.TestCase):
    def test_when_called_then_returns_matrix_with_added_vector(self):
        # Arrange
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([10, 20])
        expected_result = np.array([[11, 22], [13, 24]])

        # Act
        result = task09.add_vector_to_matrix(matrix, vector)

        # Assert
        np.testing.assert_array_equal(result, expected_result)

    def test_when_shapes_are_incompatible_then_raises_value_error(self):
        # Arrange
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([10, 20, 30])  # Incompatible shape

        # Act & Assert
        with self.assertRaises(ValueError):
            task09.add_vector_to_matrix(matrix, vector)


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
        self.assertRegex(
            output,
            r"Matrix \(\d+, \d+\) \+ column vector \(\d+, 1\):",
        ) 
        self.assertRegex(
            output,
            r"Matrix \(\d+, \d+\) \+ incompatible "
            r"vector \(\d+,\):",
        )
        self.assertIn("operands could not be broadcast together with shapes:", output)