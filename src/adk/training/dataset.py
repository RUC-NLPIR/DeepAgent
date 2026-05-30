"""
Datasets for training bidirectional reasoning network

Includes alexithymia-augmented dataset for bias mitigation
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import numpy as np


class EmotionDataset(Dataset):
    """
    Base emotion dataset for bidirectional training

    Expected format:
    - input_ids: Tokenized audio/text features
    - target_forward_ids: Emotion labels
    - target_reverse_ids: Reconstruction targets (usually same as input)
    """

    def __init__(
        self,
        data: List[Dict],
        max_seq_length: int = 512
    ):
        """
        Initialize dataset

        Args:
            data: List of dictionaries with 'input_ids', 'forward_labels', etc.
            max_seq_length: Maximum sequence length
        """
        self.data = data
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item

        Returns:
            Dict with input_ids, target_forward_ids, target_reverse_ids
        """
        item = self.data[idx]

        # Get input
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)

        # Pad/truncate to max length
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
        else:
            padding = torch.zeros(self.max_seq_length - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])

        # Forward target (emotion label)
        target_forward = torch.tensor(item.get('forward_label', [0]), dtype=torch.long)
        if len(target_forward) < self.max_seq_length:
            padding = torch.zeros(self.max_seq_length - len(target_forward), dtype=torch.long)
            target_forward = torch.cat([target_forward, padding])

        # Reverse target (reconstruction - usually input)
        target_reverse = input_ids.clone()

        return {
            'input_ids': input_ids,
            'target_forward_ids': target_forward,
            'target_reverse_ids': target_reverse
        }


class AlexithymiaAugmentedDataset(EmotionDataset):
    """
    Alexithymia-augmented dataset for bias mitigation

    Applies augmentations to simulate alexithymic patterns:
    - Flatten affect-related features
    - Add noise to prosody
    - Mask emotional prosody while preserving semantic content
    """

    def __init__(
        self,
        data: List[Dict],
        max_seq_length: int = 512,
        augmentation_prob: float = 0.3,
        affect_feature_ratio: float = 0.33
    ):
        """
        Initialize alexithymia-augmented dataset

        Args:
            data: Base data
            max_seq_length: Max sequence length
            augmentation_prob: Probability of applying alexithymia augmentation
            affect_feature_ratio: Ratio of features considered affect-related
        """
        super().__init__(data, max_seq_length)
        self.augmentation_prob = augmentation_prob
        self.affect_feature_ratio = affect_feature_ratio

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with alexithymia augmentation"""
        item = super().__getitem__(idx)

        # Apply alexithymia augmentation with probability
        if np.random.rand() < self.augmentation_prob:
            item = self._apply_alexithymia_augmentation(item)
            item['alexithymia_augmented'] = torch.tensor(1.0)
        else:
            item['alexithymia_augmented'] = torch.tensor(0.0)

        return item

    def _apply_alexithymia_augmentation(
        self,
        item: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply alexithymia augmentation

        Simulates flat affect:
        - Reduce variance in affect-related features
        - Add noise to prosody
        - Preserve semantic content
        """
        input_ids = item['input_ids'].clone()

        # Identify affect-related portion (last 1/3 of feature space)
        affect_start = int(len(input_ids) * (1 - self.affect_feature_ratio))

        # Strategy 1: Flatten affect (reduce to mean)
        affect_features = input_ids[affect_start:]
        mean_affect = affect_features[affect_features > 0].float().mean()
        if not torch.isnan(mean_affect):
            # Replace with mean (flat affect)
            input_ids[affect_start:] = mean_affect.long()

        # Strategy 2: Add prosody noise
        noise = torch.randn_like(input_ids.float()) * 0.1
        input_ids = (input_ids.float() + noise).long().clamp(0, 50000)

        # Strategy 3: Random feature masking (incomplete data)
        mask = torch.rand_like(input_ids.float()) > 0.1
        input_ids = input_ids * mask.long()

        item['input_ids'] = input_ids

        # Keep forward target unchanged - this is key!
        # We're training the model to recognize emotion even with flat affect

        return item


def create_synthetic_alexithymia_dataset(
    num_samples: int = 1000,
    seq_length: int = 128,
    num_emotions: int = 7
) -> AlexithymiaAugmentedDataset:
    """
    Create synthetic dataset for testing alexithymia bias mitigation

    Args:
        num_samples: Number of synthetic samples
        seq_length: Sequence length
        num_emotions: Number of emotion classes

    Returns:
        AlexithymiaAugmentedDataset
    """
    data = []

    for i in range(num_samples):
        # Random input features
        input_ids = np.random.randint(1, 1000, size=seq_length).tolist()

        # Random emotion label
        emotion_label = [np.random.randint(0, num_emotions)] + [0] * (seq_length - 1)

        data.append({
            'input_ids': input_ids,
            'forward_label': emotion_label
        })

    return AlexithymiaAugmentedDataset(data, max_seq_length=seq_length)
