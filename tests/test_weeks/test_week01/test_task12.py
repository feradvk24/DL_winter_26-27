import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from weeks.week01 import task12


class TestCreatePlotData(unittest.TestCase):
    def test_when_called_then_returns_correct_length(self):
        # Arrange
        samples = 1000

        # Act
        x, sin_values, cos_values, hist_data = task12.create_plot_data(samples)

        # Assert
        self.assertEqual(len(x), samples)
        self.assertEqual(len(sin_values), samples)
        self.assertEqual(len(cos_values), samples)
        self.assertEqual(len(hist_data), samples)

    def test_when_called_then_returns_correct_sin_and_cos_ranges(self):
        # Arrange
        samples = 1000

        # Act
        _, sin_values, cos_values, _ = task12.create_plot_data(samples)

        # Assert
        self.assertAlmostEqual(np.min(sin_values), -1.0, places=4)
        self.assertAlmostEqual(np.max(sin_values), 1.0, places=4)
        self.assertAlmostEqual(np.min(cos_values), -1.0, places=4)
        self.assertAlmostEqual(np.max(cos_values), 1.0, places=4)


class TestCreatePlot(unittest.TestCase):
    def test_when_called_then_creates_two_ax_plot_with_correct_attributes(self):
        # Arrange
        samples = 1000
        x, sin_values, cos_values, hist_data = task12.create_plot_data(samples)

        # Act
        figure, axes = task12.create_plot(x, sin_values, cos_values, hist_data)
        self.assertEqual(len(axes), 2)

class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    @patch("weeks.week01.task12.plt.show")
    @patch("weeks.week01.task12.create_plot")
    def test_when_called_then_prints_sin_range(self, mock_create_plot, mock_show):
        # Arrange
        expected_sin_range = [-1.0, 1.0]
        expected_sample_count = 1000

        # Act
        task12.main()

        # Assert
        output = self.capsys.readouterr().out
        self.assertEqual(
            output,
            f"sin(x) range: [{expected_sin_range[0]:.4f}, {expected_sin_range[1]:.4f}]\n"
            f"cos(x) range: [{expected_sin_range[0]:.4f}, {expected_sin_range[1]:.4f}]\n"
            f"Histogram sample count: {expected_sample_count}\n",
        )

        mock_create_plot.assert_called_once()
        mock_show.assert_called_once_with()
