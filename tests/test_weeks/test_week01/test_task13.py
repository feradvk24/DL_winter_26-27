import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from weeks.week01 import task13


class TestGenerateSamples(unittest.TestCase):
    def test_when_uniform_then_uses_provided_bounds_and_count(self):
        expected = np.array([2.0, 3.0, 4.0])
        with patch.object(task13.np.random, "uniform", return_value=expected) as sampler:
            samples = task13.generate_samples(3, (2.0, 5.0), "uniform")

        sampler.assert_called_once_with(2.0, 5.0, 3)
        self.assertIs(samples, expected)

    def test_when_normal_then_uses_provided_mean_std_and_count(self):
        expected = np.array([-1.0, 2.0, 5.0])
        with patch.object(task13.np.random, "normal", return_value=expected) as sampler:
            samples = task13.generate_samples(3, (2.0, 3.0), "normal")

        sampler.assert_called_once_with(2.0, 3.0, 3)
        self.assertIs(samples, expected)

    def test_when_exponential_then_uses_provided_offset_scale_and_count(self):
        with patch.object(
            task13.np.random, "exponential", return_value=np.array([0.0, 1.0, 8.0])
        ) as sampler:
            samples = task13.generate_samples(3, (2.0, 3.0), "exponential")

        sampler.assert_called_once_with(3.0, 3)
        np.testing.assert_array_equal(samples, [2.0, 3.0, 10.0])


class TestCreateHistograms(unittest.TestCase):
    def test_when_called_then_plots_each_distribution_on_its_own_axis(self):
        distributions = (
            np.array([0.0, 0.5, 1.0]),
            np.array([-1.0, 1.0]),
            np.array([0.0, 1.0, 2.0, 3.0]),
        )

        figure, axes = task13.create_histograms(*distributions)
        self.addCleanup(plt.close, figure)

        self.assertEqual(len(figure.axes), 3)
        for ax, title in zip(
            axes, ("Uniform", "Normal", "Exponential")
        ):
            with self.subTest(distribution=title):
                self.assertEqual(ax.get_title(), title)
                self.assertEqual(ax.get_xlabel(), "value")
                self.assertEqual(ax.get_ylabel(), "count")


class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    @patch("weeks.week01.task13.plt.show")
    @patch("weeks.week01.task13.create_histograms")
    def test_when_called_then_print_statistics_for_distributions(
        self, mock_create_histograms, mock_show
    ):
        # Arrange
        statistic_pattern = (
            r"mean=-?\d+\.\d{4}, "
            r"std=\d+\.\d{4}, "
            r"min=-?\d+\.\d{4}, "
            r"max=-?\d+\.\d{4}"
        )

        # Act
        task13.main()

        # Assert
        output = self.capsys.readouterr().out
        self.assertRegex(output, rf"Uniform: {statistic_pattern}")
        self.assertRegex(output, rf"Normal: {statistic_pattern}")
        self.assertRegex(output, rf"Exponential: {statistic_pattern}")
        mock_create_histograms.assert_called_once()
        mock_show.assert_called_once_with()
