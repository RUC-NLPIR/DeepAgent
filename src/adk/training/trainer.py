"""
Trainer for Bidirectional Reasoning Network

Implements multi-task training with:
- Forward task (emotion classification)
- Reverse task (input reconstruction)
- Contrastive learning
- Alexithymia-aware augmentation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from ..bidirectional_reasoning import BidirectionalReasoningNetwork, ReasoningConfig
from ..utils.logger import get_logger


class BidirectionalTrainer:
    """Trainer for bidirectional reasoning network"""

    def __init__(
        self,
        model: BidirectionalReasoningNetwork,
        config: ReasoningConfig,
        save_dir: str = "checkpoints"
    ):
        """
        Initialize trainer

        Args:
            model: BidirectionalReasoningNetwork instance
            config: ReasoningConfig
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("system")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000
        )

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of average metrics
        """
        self.model.train()

        epoch_metrics = {
            'total_loss': 0.0,
            'forward_loss': 0.0,
            'reverse_loss': 0.0,
            'contrastive_loss': 0.0,
            'alignment': 0.0,
            'uniformity': 0.0
        }

        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.config.device)
            target_forward_ids = batch['target_forward_ids'].to(self.config.device)
            target_reverse_ids = batch.get('target_reverse_ids', input_ids).to(self.config.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                target_forward_ids=target_forward_ids,
                target_reverse_ids=target_reverse_ids,
                training=True
            )

            # Backward pass
            loss = outputs['total_loss']
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            for key in epoch_metrics.keys():
                if key in outputs:
                    epoch_metrics[key] += outputs[key].item() if torch.is_tensor(outputs[key]) else outputs[key]

            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{outputs['total_loss'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        self.train_metrics.append(epoch_metrics)

        return epoch_metrics

    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        val_metrics = {
            'total_loss': 0.0,
            'forward_loss': 0.0,
            'reverse_loss': 0.0,
            'contrastive_loss': 0.0,
            'alignment': 0.0,
            'uniformity': 0.0,
            'verification_rate': 0.0
        }

        num_batches = 0
        num_verified = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch}"):
                input_ids = batch['input_ids'].to(self.config.device)
                target_forward_ids = batch['target_forward_ids'].to(self.config.device)
                target_reverse_ids = batch.get('target_reverse_ids', input_ids).to(self.config.device)

                outputs = self.model(
                    input_ids=input_ids,
                    target_forward_ids=target_forward_ids,
                    target_reverse_ids=target_reverse_ids,
                    training=False
                )

                # Update metrics
                for key in ['total_loss', 'forward_loss', 'reverse_loss', 'contrastive_loss', 'alignment', 'uniformity']:
                    if key in outputs:
                        val_metrics[key] += outputs[key].item() if torch.is_tensor(outputs[key]) else outputs[key]

                # Check verification (low reconstruction error)
                reconstruction_error = outputs.get('reverse_loss', 0.0)
                if torch.is_tensor(reconstruction_error):
                    reconstruction_error = reconstruction_error.item()

                if reconstruction_error < 1.0:  # Threshold
                    num_verified += input_ids.size(0)

                total_samples += input_ids.size(0)
                num_batches += 1

        # Average metrics
        for key in val_metrics:
            if key != 'verification_rate':
                val_metrics[key] /= num_batches

        val_metrics['verification_rate'] = num_verified / max(1, total_samples)

        self.val_metrics.append(val_metrics)

        return val_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        best: bool = False
    ):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            metrics: Current metrics
            best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }

        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

        # Save training history
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics
            }, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return checkpoint['epoch'], checkpoint['metrics']

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_every: int = 1
    ):
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"Starting epoch {epoch}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.logger.info(f"Train metrics: {train_metrics}")

            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader, epoch)
                self.logger.info(f"Val metrics: {val_metrics}")

                # Save best model
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    self.save_checkpoint(epoch, val_metrics, best=True)
            else:
                val_metrics = {}

            # Save regular checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, train_metrics, best=False)

        self.logger.info("Training complete!")
