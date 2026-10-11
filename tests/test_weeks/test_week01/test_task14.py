import unittest

import pytest

from weeks.week01 import task14


class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_print_min_max_and_standardized_values(self):
        # Act
        task14.main()

        # Assert
        output = self.capsys.readouterr().out
        self.assertRegex(output, r"Original:\s*\[[^\]]+\]")
        self.assertRegex(output, r"Min-max:\s*\[[^\]]+\]")
        self.assertRegex(output, r"Standardized:\s*\[[^\]]+\]")
        self.assertRegex(output, r"Min-max range:\s*\[[^\]]+\]")
        self.assertRegex(output, r"Standardized mean/std: .+ / .+")
