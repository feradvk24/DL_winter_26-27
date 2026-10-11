import unittest

import numpy as np
import pytest

from weeks.week01 import task07


class TestPrintRowAndColumn(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_second_row_and_third_column(self):
        # Arrange
        matrix = np.array(
            [
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
            ]
        )
        expected_row = np.array([50, 60, 70, 80])
        expected_column = np.array([30, 70, 110])
        expected_output = (
            f"Second row: {expected_row}\n"
            f"Third column: {expected_column}"
        )

        # Act
        task07.print_row_and_column(matrix)

        # Assert
        captured = self.capsys.readouterr()
        self.assertEqual(captured.out.strip(), expected_output)


class TestPrintMatrixSlice(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_matrix_slice(self):
        # Arrange
        matrix = np.array(
            [
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
            ]
        )
        expected_matrix_slice = np.array([[20, 30], [60, 70]])
        expected_output = f"Submatrix:\n{expected_matrix_slice}"

        # Act
        task07.print_matrix_slice(matrix)

        # Assert
        captured = self.capsys.readouterr()
        self.assertEqual(captured.out.strip(), expected_output)