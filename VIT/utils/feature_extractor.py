"""
Feature Extractor for the Agentic Reasoning Module.

Extracts 9 forensic features from the ViT's internal representations:
1. vit_fake_probability  — softmax probability of fake
2. cls_token_norm        — L2 norm of CLS token
3. attention_entropy     — average entropy of attention distributions
4. patch_variance        — variance across patch embeddings
5. rgb_attention_weight  — attention focused on RGB patches
6. fft_attention_weight  — attention focused on FFT patches
7. noise_attention_weight — attention focused on Noise patches
8. modality_agreement    — agreement between modality attentions
9. prediction_margin     — gap between top-2 softmax predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import config


class ForensicFeatureExtractor:
    """
    Extracts forensic features from a ViT9Channel model's internals.

    Usage:
        extractor = ForensicFeatureExtractor(model)
        features = extractor.extract(image_tensor)
    """

    def __init__(self, model):
        """
        Args:
            model: ViT9Channel model instance
        """
        self.model = model
        self.num_patches_per_side = config.IMAGE_SIZE // config.PATCH_SIZE  # 14
        self.total_patches = self.num_patches_per_side ** 2  # 196

        # Each modality covers 1/3 of the input channels.
        # Since all patches span all 9 channels, we use attention patterns
        # to estimate per-modality importance via gradient-based attribution
        # or positional analysis. For simplicity, we analyze the patch embedding
        # weights' activation patterns for each modality group.

    @torch.no_grad()
    def extract(self, image_tensor, device=None):
        """
        Extract all 9 forensic features from a single image.

        Args:
            image_tensor: [1, 9, 224, 224] tensor (preprocessed)
            device: computation device

        Returns:
            dict with all 9 forensic features (scalar values)
        """
        device = device or config.DEVICE
        self.model.eval()

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)

        logits, internal_features = self.model(image_tensor, return_features=True)

        cls_token = internal_features["cls_token"]          # [1, 384]
        patch_embeddings = internal_features["patch_embeddings"]  # [1, 196, 384]
        attention_weights = internal_features["attention_weights"]  # list of [1, 6, N, N]

        probs = F.softmax(logits, dim=-1)  # [1, 2]

        vit_fake_prob = probs[0, 1].item()

        cls_norm = torch.norm(cls_token, p=2, dim=-1).item()

        attn_entropy = self._compute_attention_entropy(attention_weights)

        patch_var = self._compute_patch_variance(patch_embeddings)

        # === Features 5, 6, 7: modality attention weights ===
        rgb_attn, fft_attn, noise_attn = self._compute_modality_attention(
            image_tensor, attention_weights
        )

        # === Feature 8: modality_agreement ===
        modality_agreement = self._compute_modality_agreement(rgb_attn, fft_attn, noise_attn)

        # === Feature 9: prediction_margin ===
        pred_margin = self._compute_prediction_margin(probs)

        features = {
            "vit_fake_probability": vit_fake_prob,
            "cls_token_norm": cls_norm,
            "attention_entropy": attn_entropy,
            "patch_variance": patch_var,
            "rgb_attention_weight": rgb_attn,
            "fft_attention_weight": fft_attn,
            "noise_attention_weight": noise_attn,
            "modality_agreement": modality_agreement,
            "prediction_margin": pred_margin,
        }

        return features

    def _compute_attention_entropy(self, attention_weights):
        """
        Compute average entropy across all attention heads and layers.
        Low entropy = focused attention = confident model.
        High entropy = scattered attention = uncertain model.

        Args:
            attention_weights: list of [B, heads, N, N] tensors

        Returns:
            float: average attention entropy
        """
        entropies = []
        for attn in attention_weights:
            # attn: [1, heads, N, N]
            # We look at CLS token's attention to all patches
            cls_attn = attn[0, :, 0, 1:]  # [heads, num_patches] — CLS attending to patches

            # Entropy per head
            for head_idx in range(cls_attn.shape[0]):
                p = cls_attn[head_idx]
                p = p + 1e-10  # avoid log(0)
                entropy = -(p * torch.log2(p)).sum().item()
                entropies.append(entropy)

        return float(np.mean(entropies))

    def _compute_patch_variance(self, patch_embeddings):
        """
        Compute variance across patch embeddings.
        High variance = diverse spatial features.

        Args:
            patch_embeddings: [1, num_patches, hidden_dim]

        Returns:
            float: mean variance across embedding dimensions
        """
        # Variance across patches for each hidden dimension
        var = patch_embeddings[0].var(dim=0)  # [hidden_dim]
        return var.mean().item()

    def _compute_modality_attention(self, image_tensor, attention_weights):
        """
        Estimate per-modality attention weights.

        Since all patches span all 9 channels simultaneously, we use a
        gradient-free approach: compute the contribution of each modality
        by analyzing how the patch embedding layer responds to each
        3-channel group.

        We zero out each modality group and measure the change in attention
        patterns to estimate modality importance.

        Args:
            image_tensor: [1, 9, 224, 224]
            attention_weights: attention from full forward pass

        Returns:
            tuple: (rgb_weight, fft_weight, noise_weight) — each in [0, 1]
        """
        # Get baseline CLS attention from last layer
        baseline_attn = attention_weights[-1][0].mean(dim=0)[0, 1:]  # [num_patches]
        baseline_energy = baseline_attn.sum().item()

        modality_weights = []

        for ch_start in [0, 3, 6]:  # RGB, FFT, Noise
            # Create input with this modality zeroed out
            masked_input = image_tensor.clone()
            masked_input[:, ch_start:ch_start + 3, :, :] = 0

            # Forward pass with masked input
            _, masked_features = self.model(masked_input, return_features=True)
            masked_attn = masked_features["attention_weights"][-1][0].mean(dim=0)[0, 1:]

            # Importance = how much attention changed when modality was removed
            change = (baseline_attn - masked_attn).abs().sum().item()
            modality_weights.append(change)

        # Normalize to sum to 1
        total = sum(modality_weights) + 1e-10
        rgb_w = modality_weights[0] / total
        fft_w = modality_weights[1] / total
        noise_w = modality_weights[2] / total

        return rgb_w, fft_w, noise_w

    def _compute_modality_agreement(self, rgb_w, fft_w, noise_w):
        """
        Compute agreement between modality attention weights.
        If all modalities contribute roughly equally → high agreement.
        If one dominates → lower agreement.

        Uses inverse coefficient of variation (lower CV = more agreement).

        Args:
            rgb_w, fft_w, noise_w: modality attention weights

        Returns:
            float: agreement score in [0, 1], higher = more agreement
        """
        weights = np.array([rgb_w, fft_w, noise_w])
        mean_w = weights.mean()
        std_w = weights.std()

        if mean_w < 1e-10:
            return 0.0

        # Coefficient of variation (CV)
        cv = std_w / mean_w

        # Convert to agreement: low CV = high agreement
        # CV=0 → agreement=1, CV→∞ → agreement→0
        agreement = 1.0 / (1.0 + cv)

        return float(agreement)

    def _compute_prediction_margin(self, probs):
        """
        Compute the gap between the top-2 softmax probabilities.
        Large margin = decisive prediction, small margin = uncertain.

        Args:
            probs: [1, 2] softmax probabilities

        Returns:
            float: margin in [0, 1]
        """
        sorted_probs, _ = probs[0].sort(descending=True)
        margin = (sorted_probs[0] - sorted_probs[1]).item()
        return margin
