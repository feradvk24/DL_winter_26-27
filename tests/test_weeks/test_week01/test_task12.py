import unittest
from unittest.mock import patch

import numpy as np
import pytest

from weeks.week01 import task12


class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_sin_range(self):
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

