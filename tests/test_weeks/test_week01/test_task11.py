import unittest
from unittest.mock import patch

import pytest
import numpy as np

from weeks.week01 import task11


class TestCreateDatasetWithNoise(unittest.TestCase):
    def test_when_called_then_returns_correct_shapes_and_values(self):
        # Arrange
        num_points = 50
        range_x = (0, 10)
        noise_range = (-1, 1)

        # Act
        x, y = task11.create_dataset_with_noise(
            num_points, range_x, noise_range
        )

        # Assert
        self.assertEqual(x.shape, (num_points,))
        self.assertEqual(y.shape, (num_points,))
        self.assertTrue(np.all(x >= range_x[0]) and np.all(x <= range_x[1]))
        self.assertTrue(
            np.all(y >= 2.0 * x + 1.0 + noise_range[0])
            and np.all(y <= 2.0 * x + 1.0 + noise_range[1])
        )


class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_dataset_summary_and_shows_plot(self):
        # Arrange
        task11.main()

        output = self.capsys.readouterr().out
        self.assertIn(
            "Generated 50 points for y = 2.0*x + 1.0 + noise",
            output,
        )
        self.assertRegex(
            output,
            r"X range: \((-?\d+\.\d+), (-?\d+\.\d+)\)",
        )
        self.assertRegex(
            output,
            r"Y range: \((-?\d+\.\d+), (-?\d+\.\d+)\)",
        )