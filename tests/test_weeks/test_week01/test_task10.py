import unittest

import numpy as np
import pytest

from weeks.week01 import task10

class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_matrix_products_and_transposition(self):
        # Act
        task10.main()

        # Assert
        output = self.capsys.readouterr().out
        self.assertIn("A:\n[[", output)
        self.assertIn("B:\n[[", output)