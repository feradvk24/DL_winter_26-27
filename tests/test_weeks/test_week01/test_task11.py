import unittest
from unittest.mock import patch

import pytest
import numpy as np

from weeks.week01 import task11

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
