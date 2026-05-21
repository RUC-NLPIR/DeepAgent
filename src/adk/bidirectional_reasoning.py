"""
Bidirectional Reasoning Network for Neuroadaptive Accessibility
Implements contrastive learning + obfuscation for robust emotion understanding

Based on:
- arXiv:2509.05553 (Bidirectional Transformers)
- Contrastive Learning for Sequential Recommendation (CIKM 2022)

Key Innovation: Addresses emotion AI bias for neurodivergent users (alexithymia)
by using bidirectional verification instead of unidirectional classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ReasoningConfig:
    """Configuration for bidirectional reasoning network"""
    # Layer 1: Input Encoding
    vocab_size: int = 50000
    embedding_dim: int = 768
    max_seq_length: int = 512

    # Layer 2: Transformer Encoder
    num_encoder_layers: int = 6
    num_attention_heads: int = 12
    hidden_dim: int = 768
    feedforward_dim: int = 3072
    dropout: float = 0.1

    # Layer 3: Bidirectional Decoders
    num_decoder_layers: int = 6
    use_cross_attention: bool = True

    # Layer 4: Contrastive Learning
    temperature: float = 0.07
    contrastive_weight: float = 0.3

    # Layer 5: Obfuscation
    obfuscation_prob: float = 0.15
    obfuscation_weight: float = 0.2

    # Training
    forward_task_weight: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MultiScaleEmbedding(nn.Module):
    """
    Layer 1: Input Encoding with multi-scale embeddings
    Handles hierarchical input (words, phrases, sentences)
    """
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.embedding_dim)

        # Multi-scale: word-level, phrase-level (3-gram), sentence-level
        self.scale_projections = nn.ModuleDict({
            'word': nn.Linear(config.embedding_dim, config.embedding_dim),
            'phrase': nn.Conv1d(config.embedding_dim, config.embedding_dim, kernel_size=3, padding=1),
            'sentence': nn.Linear(config.embedding_dim, config.embedding_dim)
        })

        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            Dict with multi-scale embeddings
        """
        batch_size, seq_len = input_ids.shape

        # Base embeddings
        token_emb = self.token_embedding(input_ids)  # [batch, seq, dim]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_emb = self.position_embedding(position_ids)

        base_emb = token_emb + position_emb

        # Multi-scale projections
        word_scale = self.scale_projections['word'](base_emb)

        phrase_scale = self.scale_projections['phrase'](
            base_emb.transpose(1, 2)  # [batch, dim, seq]
        ).transpose(1, 2)  # back to [batch, seq, dim]

        sentence_scale = self.scale_projections['sentence'](
            base_emb.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        )

        # Combine scales
        multi_scale = word_scale + phrase_scale + sentence_scale
        multi_scale = self.layer_norm(multi_scale)
        multi_scale = self.dropout(multi_scale)

        return {
            'embeddings': multi_scale,
            'word_scale': word_scale,
            'phrase_scale': phrase_scale,
            'sentence_scale': sentence_scale
        }


class BidirectionalReasoningModule(nn.Module):
    """
    Layer 3: Forward + Reverse Decoders with Cross-Attention

    Forward: Input → Reasoning/Emotion
    Reverse: Emotion → Input Reconstruction
    Cross-Attention: Ensures bidirectional consistency
    """
    def __init__(self, config: ReasoningConfig):
        super().__init__()

        # Forward decoder (input → output)
        self.forward_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.num_decoder_layers
        )

        # Reverse decoder (output → input reconstruction)
        self.reverse_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.num_decoder_layers
        )

        # Cross-attention between forward and reverse
        if config.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True
            )

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_forward: Optional[torch.Tensor] = None,
        target_reverse: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_output: [batch, seq, dim] from transformer encoder
            target_forward: [batch, target_seq, dim] for forward task
            target_reverse: [batch, seq, dim] for reverse reconstruction

        Returns:
            Dict with forward_output, reverse_output, cross_attended
        """
        # Forward reasoning: input → emotion/reasoning
        if target_forward is not None:
            forward_output = self.forward_decoder(
                tgt=target_forward,
                memory=encoder_output
            )
        else:
            # Autoregressive generation (for inference)
            forward_output = self._autoregressive_decode(
                self.forward_decoder, encoder_output
            )

        # Reverse reasoning: emotion → input reconstruction
        if target_reverse is not None:
            reverse_output = self.reverse_decoder(
                tgt=target_reverse,
                memory=forward_output
            )
        else:
            reverse_output = self._autoregressive_decode(
                self.reverse_decoder, forward_output
            )

        # Cross-attention for bidirectional alignment
        if hasattr(self, 'cross_attention'):
            cross_attended, _ = self.cross_attention(
                query=forward_output,
                key=reverse_output,
                value=reverse_output
            )
        else:
            cross_attended = forward_output

        return {
            'forward_output': forward_output,
            'reverse_output': reverse_output,
            'cross_attended': cross_attended
        }

    def _autoregressive_decode(self, decoder, memory, max_len=50):
        """Simple autoregressive decoding for inference"""
        batch_size = memory.size(0)
        device = memory.device

        # Start with <SOS> token (assume index 1)
        output = torch.ones(batch_size, 1, memory.size(-1), device=device)

        for _ in range(max_len):
            decoded = decoder(tgt=output, memory=memory)
            output = torch.cat([output, decoded[:, -1:, :]], dim=1)

        return output


class ContrastiveLearningModule(nn.Module):
    """
    Layer 4: Contrastive Learning for Bidirectional Consistency

    Ensures semantic alignment between:
    - Forward output (emotion from audio)
    - Reverse output (reconstructed audio features)
    """
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.temperature = config.temperature

        # Projection heads for contrastive learning
        self.forward_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 128)
        )

        self.reverse_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 128)
        )

    def forward(
        self,
        forward_features: torch.Tensor,
        reverse_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            forward_features: [batch, dim] from forward decoder
            reverse_features: [batch, dim] from reverse decoder
            labels: [batch] optional labels for supervised contrastive

        Returns:
            contrastive_loss, metrics_dict
        """
        # Project to contrastive space
        z_forward = F.normalize(self.forward_projection(forward_features), dim=1)
        z_reverse = F.normalize(self.reverse_projection(reverse_features), dim=1)

        batch_size = z_forward.size(0)

        # Compute similarity matrix
        similarity = torch.matmul(z_forward, z_reverse.T) / self.temperature

        # Positive pairs: forward[i] ↔ reverse[i] (same sample)
        positive_mask = torch.eye(batch_size, device=z_forward.device)

        # Negative pairs: all other combinations
        negative_mask = 1 - positive_mask

        # InfoNCE loss
        exp_sim = torch.exp(similarity)
        positive_sim = (exp_sim * positive_mask).sum(dim=1)
        negative_sim = (exp_sim * negative_mask).sum(dim=1)

        contrastive_loss = -torch.log(positive_sim / (positive_sim + negative_sim + 1e-8))
        contrastive_loss = contrastive_loss.mean()

        # Metrics
        with torch.no_grad():
            # Alignment: how close are positive pairs?
            alignment = (z_forward * z_reverse).sum(dim=1).mean()

            # Uniformity: how spread out are representations?
            uniformity = torch.pdist(z_forward).pow(2).mul(-2).exp().mean().log()

        metrics = {
            'contrastive_loss': contrastive_loss.item(),
            'alignment': alignment.item(),
            'uniformity': uniformity.item()
        }

        return contrastive_loss, metrics


class ObfuscationAugmentation(nn.Module):
    """
    Layer 5: Obfuscation-based Regularization

    Critical for accessibility: trains on ambiguous/alexithymic patterns
    - Flat affect with varying emotions (alexithymia)
    - Masked prosody features
    - Synthetic "hard negatives"
    """
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.obfuscation_prob = config.obfuscation_prob

        # Obfuscation strategies
        self.dropout = nn.Dropout(config.obfuscation_prob)
        self.noise_std = 0.1

    def forward(
        self,
        embeddings: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Apply obfuscation augmentations during training

        Args:
            embeddings: [batch, seq, dim]
            training: whether to apply obfuscation

        Returns:
            obfuscated_embeddings
        """
        if not training:
            return embeddings

        batch_size, seq_len, dim = embeddings.shape

        # Strategy 1: Feature dropout (simulates incomplete data)
        obfuscated = self.dropout(embeddings)

        # Strategy 2: Gaussian noise (simulates sensor noise)
        noise = torch.randn_like(embeddings) * self.noise_std
        obfuscated = obfuscated + noise

        # Strategy 3: Token masking (random)
        mask = torch.rand(batch_size, seq_len, 1, device=embeddings.device)
        mask = (mask > self.obfuscation_prob).float()
        obfuscated = obfuscated * mask

        # Strategy 4: Alexithymia simulation (flatten affect dimensions)
        # Assume last 1/3 of dims are affect-related
        affect_start = (2 * dim) // 3
        alexithymia_mask = torch.rand(batch_size, 1, 1, device=embeddings.device) < 0.3
        obfuscated[:, :, affect_start:] = torch.where(
            alexithymia_mask,
            torch.zeros_like(obfuscated[:, :, affect_start:]),
            obfuscated[:, :, affect_start:]
        )

        return obfuscated


class BidirectionalReasoningNetwork(nn.Module):
    """
    Complete Bidirectional Reasoning Architecture

    Integrates all layers for neuroadaptive accessibility:
    1. Multi-scale input encoding
    2. Transformer encoder
    3. Bidirectional decoders
    4. Contrastive learning
    5. Obfuscation augmentation
    6. Multi-task output
    """
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config

        # Layer 1: Input Encoding
        self.embedding = MultiScaleEmbedding(config)

        # Layer 2: Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.num_encoder_layers
        )

        # Layer 3: Bidirectional Reasoning
        self.bidirectional_module = BidirectionalReasoningModule(config)

        # Layer 4: Contrastive Learning
        self.contrastive_module = ContrastiveLearningModule(config)

        # Layer 5: Obfuscation
        self.obfuscation = ObfuscationAugmentation(config)

        # Layer 6: Output heads
        self.forward_output_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.reverse_output_head = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_forward_ids: Optional[torch.Tensor] = None,
        target_reverse_ids: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with multi-task training

        Args:
            input_ids: [batch, seq] input tokens
            target_forward_ids: [batch, target_seq] forward task labels
            target_reverse_ids: [batch, seq] reverse reconstruction targets
            training: whether in training mode

        Returns:
            Dict with outputs and losses
        """
        # Layer 1: Multi-scale embedding
        embedding_dict = self.embedding(input_ids)
        embeddings = embedding_dict['embeddings']

        # Layer 5: Apply obfuscation during training
        embeddings = self.obfuscation(embeddings, training=training)

        # Layer 2: Transformer encoding
        encoder_output = self.encoder(embeddings)

        # Prepare targets
        if target_forward_ids is not None:
            target_forward_emb = self.embedding(target_forward_ids)['embeddings']
        else:
            target_forward_emb = None

        if target_reverse_ids is not None:
            target_reverse_emb = self.embedding(target_reverse_ids)['embeddings']
        else:
            target_reverse_emb = encoder_output  # Reconstruct input

        # Layer 3: Bidirectional reasoning
        reasoning_outputs = self.bidirectional_module(
            encoder_output,
            target_forward=target_forward_emb,
            target_reverse=target_reverse_emb
        )

        # Layer 6: Output projections
        forward_logits = self.forward_output_head(reasoning_outputs['forward_output'])
        reverse_logits = self.reverse_output_head(reasoning_outputs['reverse_output'])

        outputs = {
            'forward_logits': forward_logits,
            'reverse_logits': reverse_logits,
            'forward_features': reasoning_outputs['forward_output'].mean(dim=1),
            'reverse_features': reasoning_outputs['reverse_output'].mean(dim=1)
        }

        # Compute losses if training
        if training and target_forward_ids is not None:
            losses = self._compute_losses(
                outputs,
                target_forward_ids,
                target_reverse_ids if target_reverse_ids is not None else input_ids
            )
            outputs.update(losses)

        return outputs

    def _compute_losses(
        self,
        outputs: Dict,
        target_forward: torch.Tensor,
        target_reverse: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Training Objective: Multi-task balanced loss

        L_total = α*L_forward + β*L_contrastive + γ*L_obfuscation
        """
        # Forward task loss (cross-entropy)
        forward_loss = F.cross_entropy(
            outputs['forward_logits'].reshape(-1, self.config.vocab_size),
            target_forward.reshape(-1),
            ignore_index=0  # Padding
        )

        # Reverse task loss (reconstruction)
        reverse_loss = F.cross_entropy(
            outputs['reverse_logits'].reshape(-1, self.config.vocab_size),
            target_reverse.reshape(-1),
            ignore_index=0
        )

        # Layer 4: Contrastive loss
        contrastive_loss, contrastive_metrics = self.contrastive_module(
            outputs['forward_features'],
            outputs['reverse_features']
        )

        # Balanced multi-task loss
        total_loss = (
            self.config.forward_task_weight * forward_loss +
            self.config.contrastive_weight * contrastive_loss +
            self.config.obfuscation_weight * reverse_loss
        )

        return {
            'total_loss': total_loss,
            'forward_loss': forward_loss,
            'reverse_loss': reverse_loss,
            'contrastive_loss': contrastive_loss,
            **contrastive_metrics
        }


# ============================================================================
# Integration with Neuroadaptive Wrapper
# ============================================================================

class BidirectionalEmotionClassifier:
    """
    Wrapper for using bidirectional reasoning in accessibility context

    Replaces simple Valence API with bidirectional consistency checking:
    - Forward: Audio → Emotion
    - Reverse: Emotion → Audio features (verify consistency)
    - Contrastive: Ensure alexithymic patterns don't create false negatives
    """
    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()
        self.model = BidirectionalReasoningNetwork(self.config)
        self.model.to(self.config.device)
        self.model.eval()

        # Emotion labels
        self.emotion_labels = [
            "neutral", "happy", "sad", "angry", "fearful",
            "disgusted", "surprised", "calm", "anxious"
        ]

    def classify_with_verification(
        self,
        audio_features: torch.Tensor
    ) -> Dict[str, any]:
        """
        Classify emotion with bidirectional verification

        Args:
            audio_features: [batch, seq, dim] audio feature tensor

        Returns:
            Dict with emotion, confidence, verification_score
        """
        with torch.no_grad():
            # Ensure proper shape and device
            if audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(0)

            audio_features = audio_features.to(self.config.device)

            # Convert to token IDs (simple bucketing for demo)
            input_ids = self._features_to_tokens(audio_features)

            outputs = self.model(
                input_ids,
                training=False
            )

            # Forward prediction
            forward_probs = F.softmax(outputs['forward_logits'], dim=-1)
            emotion_id = forward_probs[:, 0, :len(self.emotion_labels)].argmax(dim=-1)
            confidence = forward_probs[:, 0, :len(self.emotion_labels)].max(dim=-1).values

            # Reverse verification: can we reconstruct input from prediction?
            reverse_probs = F.softmax(outputs['reverse_logits'], dim=-1)
            reconstruction_loss = F.mse_loss(
                reverse_probs,
                F.softmax(outputs['forward_logits'], dim=-1)
            )

            # Verification score: low reconstruction loss = high confidence
            verification_score = torch.exp(-reconstruction_loss)

            return {
                'emotion': self.emotion_labels[emotion_id.item()] if emotion_id.item() < len(self.emotion_labels) else "neutral",
                'emotion_id': emotion_id.item(),
                'confidence': confidence.item(),
                'verification_score': verification_score.item(),
                'is_verified': verification_score.item() > 0.7,
                'all_probabilities': forward_probs[0, 0, :len(self.emotion_labels)].cpu().numpy().tolist()
            }

    def _features_to_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """
        Convert audio features to token IDs
        Simple bucketing approach for demo
        """
        # Normalize features
        features_norm = (features - features.mean()) / (features.std() + 1e-8)

        # Bucket into vocab range
        tokens = ((features_norm + 3) / 6 * 1000).long().clamp(0, self.config.vocab_size - 1)

        return tokens[:, :, 0]  # Take first feature dimension
