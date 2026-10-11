import unittest

import numpy as np
import pytest

from weeks.week01 import task10

class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_matrix_products_and_transposition(self):
        # Arrange
        expected_a = np.array(
            [[1, 2],[3, 4]],
        )
        expected_b = np.array(
            [[5, 6],[7, 8]]
        )
        expected_elementwise = np.array(
            [[5, 12],[21, 32],]
        )
        expected_matrix_product = np.array(
            [[19, 22],[43, 50]]
        )
        expected_output = (
            f"A:\n{expected_a}\n"
            f"B:\n{expected_b}\n"
            f"Elementwise:\n{expected_elementwise}\n"
            f"Matrix multiplication:\n{expected_matrix_product}"
        )

        # Act
        task10.main()

        # Assert
        actual_output = self.capsys.readouterr().out
        self.assertEqual(actual_output.strip(), expected_output)