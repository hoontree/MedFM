import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Init dummy config
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.getcwd())

from trainers.ca_sam_trainer import CASAMTrainer


class TestCASAMContinual(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "model": {
                    "mode": "continual",
                    "name": "ca_sam",
                    "sam_type": "vit_b",
                    "alignment": {"hidden_dim": 16, "num_blocks": 1},
                    "vae": {"latent_dim": 8, "threshold_percentile": 95},
                },
                "data": {
                    "name": "dynamic",
                    "train": ["Task1", "Task2"],
                    "num_classes": 1,
                },
                "training": {
                    "num_epochs": 1,
                    "batch_size": 2,
                    "num_workers": 0,
                    "base_lr": 1e-4,
                },
                "hardware": {"n_gpu": 0},
                "output": {"dir": "logs_test"},
            }
        )

    @patch("trainers.ca_sam_trainer.SegDatasetProcessor")
    @patch("trainers.ca_sam_trainer.CASAM")
    @patch("trainers.ca_sam_trainer.sam_model_registry")
    def test_train_continual_flow(
        self, mock_sam_registry, mock_casam_cls, mock_data_proc
    ):
        # Mock SAM
        mock_sam = MagicMock()
        mock_sam.image_encoder = MagicMock()
        mock_sam.mask_decoder = MagicMock()
        mock_sam.prompt_encoder = MagicMock()
        build_sam = MagicMock(return_value=(mock_sam, 256))
        mock_sam_registry.__getitem__.return_value = build_sam

        # Mock CASAM instance
        mock_casam_instance = MagicMock()
        mock_casam_cls.return_value = mock_casam_instance
        mock_casam_instance.add_new_task.side_effect = [0, 1, 2]  # Subsequent task IDs
        mock_casam_instance.alignment_layers = [MagicMock(), MagicMock()]  # 2 layers
        mock_casam_instance.get_num_trainable_params.side_effect = lambda x: 1000

        # Mock Dataloaders
        mock_loader = MagicMock()
        mock_loader.dataset = [1, 2, 3]  # len 3
        mock_loader.__len__.return_value = 2
        mock_loader.__iter__.return_value = iter(
            [
                (torch.randn(2, 3, 64, 64), torch.randn(2, 64, 64)),  # batch 1
                (torch.randn(2, 3, 64, 64), torch.randn(2, 64, 64)),  # batch 2
            ]
        )

        # Setup mock return values for SegDatasetProcessor
        # build_continual_data_loaders returns (train, val, test)
        mock_data_proc.build_continual_data_loaders.return_value = (
            mock_loader,
            mock_loader,
            mock_loader,
        )
        mock_data_proc.load_dataset_from_config.return_value = (
            MagicMock()
        )  # return dataset

        # Initialize Trainer
        trainer = CASAMTrainer(self.cfg)
        trainer.logger = MagicMock()
        trainer._create_model()  # Manually create model since setup() is skipped

        # Mock other trainer methods to avoid full execution
        trainer._save_checkpoint = MagicMock()
        trainer._save_model = MagicMock()
        trainer.train_epoch = MagicMock(return_value={"loss": 0.1, "iou": 0.8})
        trainer.validate = MagicMock(
            return_value={"loss": 0.1, "iou": 0.8, "Dice": 0.8}
        )
        trainer.train_vae_for_current_task = MagicMock()
        trainer.evaluate_all_tasks = MagicMock()

        # Run train_continual
        trainer.train_continual()

        # Verifications

        # 1. Check if it iterated over 2 tasks
        tasks = self.cfg.data.train
        expect_calls = [((self.cfg, t),) for t in tasks]
        # Check _setup_task_data called build_continual_data_loaders correct number of times
        self.assertEqual(mock_data_proc.build_continual_data_loaders.call_count, 2)

        # 2. Check add_new_task called (for 2nd task)
        # First task (idx 0) is created in __init__
        # Second task (idx 1) is created in loop
        # So add_new_task should be called at least 1 time in loop + 1 time in init = 2 times
        # trainer.current_task_id starts at 0 (from init)
        # train_continual loop:
        #   Task 0: no add_new_task
        #   Task 1: add_new_task() called
        self.assertEqual(trainer.ca_sam_model.add_new_task.call_count, 2)

        # 3. Check VAE training called twice
        self.assertEqual(trainer.train_vae_for_current_task.call_count, 2)

        # 4. Check Evaluate All Tasks called twice
        self.assertEqual(trainer.evaluate_all_tasks.call_count, 2)
        trainer.evaluate_all_tasks.assert_any_call(["Task1"])
        trainer.evaluate_all_tasks.assert_any_call(["Task1", "Task2"])

        print("Continual Learning Flow Test Passed!")


if __name__ == "__main__":
    unittest.main()
