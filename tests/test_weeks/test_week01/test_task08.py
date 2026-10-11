import unittest

import numpy as np
import pytest

from weeks.week01 import task08


class TestArraySumWithForLoop(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_returns_array_sum(self):
        # Arrange
        arr = np.array([0.5, 1, 1.5, 2, 3])
        expected_sum = 8.0

        # Act
        array_sum = task08.for_loop_sum(arr)

        # Assert
        self.assertAlmostEqual(array_sum, expected_sum)


class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_array_length(self):
        # Act
        task08.main()

        # Assert
        output = self.capsys.readouterr().out
        self.assertIn("Array length: 1,000,000", output)
        self.assertIn("Loop sum:", output)