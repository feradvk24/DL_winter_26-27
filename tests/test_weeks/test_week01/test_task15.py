import unittest
from unittest.mock import patch

import numpy as np
import pytest

from weeks.week01 import task15


class TestPrintNpArrayInfo(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_print_correct_statement(self):
        # Arrange
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

        # Act
        task15.print_np_array_info(arr)

        # Assert
        output = self.capsys.readouterr().out
        self.assertIn("NumPy array:", output)
        self.assertIn("[[1. 2. 3.]", output)
        self.assertIn("[4. 5. 6.]]", output)
        self.assertIn("dtype=float64, shape=(2, 3)", output)


class TestMain(unittest.TestCase):
    @patch("weeks.week01.task15.print_np_array_info")
    def test_when_called_then_prints_array_details(self, mock_print_np_array):
        task15.main()
        mock_print_np_array.assert_called_once()
