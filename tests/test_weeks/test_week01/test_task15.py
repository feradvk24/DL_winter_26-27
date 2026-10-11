import unittest
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

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


class TestPrintTensorInfo(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_print_correct_statement(self):
        # Arrange
        tensor = torch.tensor(
            [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu"
        )

        # Act
        task15.print_tensor_info(tensor)

        # Assert
        output = self.capsys.readouterr().out
        self.assertIn("PyTorch tensor:", output)
        self.assertIn("tensor([[1., 2., 3.],", output)
        self.assertIn("[4., 5., 6.]])", output)
        self.assertIn("dtype=torch.float32", output)
        self.assertIn("shape=torch.Size([2, 3])", output)
        self.assertIn("device=cpu", output)


class TestMoveTensorToDevice(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    @patch("weeks.week01.task15.torch.backends.mps.is_available", return_value=True)
    def test_when_mps_available_then_moves_tensor_to_mps(self, mock_available):
        # Arrange
        tensor = Mock(spec=torch.Tensor)
        tensor.to.return_value = Mock(device="mps:0", dtype=torch.float32)

        # Act
        moved_tensor = task15.move_tensor_to_device(tensor)

        # Assert
        tensor.to.assert_called_once_with("mps")
        self.assertIs(moved_tensor, tensor.to.return_value)
        output = self.capsys.readouterr().out
        self.assertIn("Selected device: mps", output)
        self.assertIn("Tensor moved to device: mps:0, dtype=torch.float32", output)

    @patch("weeks.week01.task15.torch.backends.mps.is_available", return_value=False)
    def test_when_mps_unavailable_then_moves_tensor_to_cpu(self, mock_available):
        # Arrange
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cpu")

        # Act
        moved_tensor = task15.move_tensor_to_device(tensor)

        # Assert
        self.assertEqual(moved_tensor.device, torch.device("cpu"))
        self.assertEqual(moved_tensor.dtype, torch.float32)
        output = self.capsys.readouterr().out
        self.assertIn("Selected device: cpu", output)
        self.assertIn("Tensor moved to device: cpu, dtype=torch.float32", output)


class TestMain(unittest.TestCase):
    @patch("weeks.week01.task15.move_tensor_to_device")
    @patch("weeks.week01.task15.print_tensor_info")
    @patch("weeks.week01.task15.print_np_array_info")
    def test_when_called_then_prints_array_and_tensor_details(
        self, mock_print_np_array, mock_print_tensor, mock_move
    ):
        # Act
        task15.main()

        # Assert
        mock_print_np_array.assert_called_once()
        mock_print_tensor.assert_called_once()
        mock_move.assert_called_once()
