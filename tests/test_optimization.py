import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Fully mock all dependencies to avoid ModuleNotFoundError
mock_torch = MagicMock()
mock_torch.device.return_value = "cuda"
mock_torch.randn.return_value = MagicMock()
mock_nn = MagicMock()
mock_optim = MagicMock()
mock_plt = MagicMock()
mock_tqdm = MagicMock()
mock_torchvision = MagicMock()

# Define a mock for loss that handles string formatting and arithmetic correctly
class MockLoss:
    def item(self):
        return 0.5
    def mean(self):
        return self
    def backward(self):
        pass
    def __format__(self, format_spec):
        return format(self.item(), format_spec)
    def __sub__(self, other):
        return self
    def __rsub__(self, other):
        return self
    def __add__(self, other):
        return self
    def __radd__(self, other):
        return self
    def __truediv__(self, other):
        return self
    def __neg__(self):
        return self

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_nn
sys.modules["torch.optim"] = mock_optim
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = mock_plt
sys.modules["matplotlib.ticker"] = MagicMock()
sys.modules["tqdm"] = mock_tqdm
sys.modules["torchvision"] = mock_torchvision
sys.modules["torchvision.utils"] = mock_torchvision.utils

# Mock internal modules
sys.modules["app.core.v1.generator"] = MagicMock()
sys.modules["app.core.v1.discriminator"] = MagicMock()

# Now import the trainer after mocking
from app.utils.trainer import SAGANTrainer

class TestSAGANTrainerOptimization(unittest.TestCase):
    def setUp(self):
        self.generator = MagicMock()
        self.discriminator = MagicMock()
        self.dataloader = [ (MagicMock(), MagicMock()) ]
        self.config = {
            "train": {
                "sample_step": 1,
                "checkpoint_step": 1,
                "beta1": 0.5,
                "beta2": 0.999,
                "lr_g": 0.0001,
                "lr_d": 0.0004,
                "epochs": 1,
                "batch_size": 1
            },
            "model": {
                "latent_dim": 100
            },
            "path": {
                "sample_dir": "samples",
                "checkpoint_dir": "checkpoints"
            }
        }

    def test_randn_uses_device_param(self):
        # Reset mock to clear calls during initialization
        mock_torch.randn.reset_mock()

        # Instantiate trainer
        trainer = SAGANTrainer(self.generator, self.discriminator, self.dataloader, self.config)

        # Check if randn was called with device during __init__ (fixed_noise)
        self.assertTrue(any(call.kwargs.get('device') == trainer.device for call in mock_torch.randn.call_args_list))

        # Reset again for train call
        mock_torch.randn.reset_mock()

        # Mocking necessary parts for train() to run once
        trainer.g = MagicMock()
        trainer.d = MagicMock()
        trainer.g_opt = MagicMock()
        trainer.d_opt = MagicMock()

        # Dataloader should return something iterable that yields (img, label)
        img_mock = MagicMock()
        img_mock.size.return_value = (1, 3, 128, 128)
        img_mock.to.return_value = img_mock
        trainer.dataloader = [(img_mock, MagicMock())]

        # Mocking progress bar
        mock_pb = MagicMock()

        # Use our custom MockLoss
        mock_loss = MockLoss()

        # Mocking model outputs and nn.ReLU outputs
        trainer.g.return_value = MagicMock()
        trainer.d.return_value = mock_loss
        mock_nn.ReLU.return_value = lambda x: mock_loss

        # Run one step of training
        with patch('os.path.exists', return_value=True), \
             patch('os.mkdir'), \
             patch('os.path.abspath', side_effect=lambda x: x), \
             patch('os.path.join', side_effect=lambda *args: "/".join(args)), \
             patch('app.utils.trainer.diff_augment', side_effect=lambda x, **kwargs: x), \
             patch('app.utils.trainer.tqdm', return_value=mock_pb):

            # Since tqdm is mocked to return mock_pb, it needs to be iterable
            mock_pb.__iter__.return_value = enumerate(trainer.dataloader)

            # Replace d_loss calculation to use MockLoss
            with patch('app.utils.trainer.nn.ReLU', return_value=lambda x: mock_loss), \
                 patch('app.utils.trainer.SAGANTrainer.save_loss_plot'), \
                 patch('app.utils.trainer.SAGANTrainer.save_checkpoint'), \
                 patch('torch.save'):
                trainer.train(epochs=1)

        # Verify randn was called with device parameter
        randn_calls = mock_torch.randn.call_args_list

        # Expected calls for 1 epoch, 1 batch, epoch=0:
        # - Discriminator training (line 106): 1 call
        # - Generator training (line 140): g_run_cnt=3 calls
        # - Visualization (line 170 and 174): 2 calls
        # Total: 6 calls

        self.assertEqual(len(randn_calls), 6)

        for i, call in enumerate(randn_calls):
            self.assertIn('device', call.kwargs, f"randn call #{i} without device: {call}")
            self.assertEqual(call.kwargs.get('device'), trainer.device, f"randn call #{i} with wrong device: {call}")

if __name__ == "__main__":
    unittest.main()
