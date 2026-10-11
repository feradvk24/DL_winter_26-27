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


class TestComputeTwoSums(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_returns_loop_and_vectorized_sums(self):
        # Arrange
        arr = np.array([0.5, 1, 1.5, 2, 3])
        expected_loop_sum = 8.0
        expected_vectorized_sum = 8.0

        # Act
        loop_sum, vectorized_sum, loop_time, vectorized_time = task08.compute_two_sums(arr)

        # Assert
        self.assertAlmostEqual(loop_sum, expected_loop_sum)
        self.assertAlmostEqual(vectorized_sum, expected_vectorized_sum)

    def test_when_called_then_loop_and_vectorized_sums_are_equal(self):
        # Arrange
        arr = np.random.rand(1_000_000)

        # Act
        loop_sum, vectorized_sum, loop_time, vectorized_time = task08.compute_two_sums(arr)

        # Assert
        self.assertAlmostEqual(loop_sum, vectorized_sum)


class TestArrayStats(unittest.TestCase):
    def test_when_called_then_returns_expected_array_stats(self):
        # Arrange
        arr = np.array([0.0, 2.0])
        expected_mean = 1.0
        expected_std = 1.0
        expected_min = 0.0
        expected_max = 2.0

        # Act
        actual_mean, actual_std, actual_min, actual_max = (
            task08.get_array_stats(arr)
        )

        # Assert
        self.assertAlmostEqual(actual_mean, expected_mean)
        self.assertAlmostEqual(actual_std, expected_std)
        self.assertAlmostEqual(actual_min, expected_min)
        self.assertAlmostEqual(actual_max, expected_max)


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
        self.assertRegex(
            output,
            r"Loop sum: -?\d+\.\d+ \(\d+\.\d+s\)",
        )
        self.assertRegex(
            output,
            r"Vectorized sum: -?\d+\.\d+ \(\d+\.\d+s\)",
        )
        self.assertRegex(
            output,
            r"Speedup: \d+\.\d+x",
        )
        self.assertRegex(
        output,
        r"Vectorized mean: -?\d+\.\d+",
        )
        self.assertRegex(
            output,
            r"Vectorized std: \d+\.\d+",
        )
        self.assertRegex(
            output,
            r"Vectorized min/max: -?\d+\.\d+ / -?\d+\.\d+",
        )
