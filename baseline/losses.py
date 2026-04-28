"""Advanced loss functions for PCVR prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE loss with large-scale in-batch negatives.

    Args:
        temperature: Temperature parameter for scaling logits
        negative_mode: 'batch' for in-batch negatives
    """

    def __init__(self, temperature: float = 0.07, negative_mode: str = 'batch'):
        super().__init__()
        self.temperature = temperature
        self.negative_mode = negative_mode

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) binary labels (1=positive, 0=negative)

        Returns:
            Scalar loss
        """
        B = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix: (B, B)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask out diagonal (self-similarity)
        mask = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # For each positive sample, all other samples in batch are negatives
        pos_mask = labels.bool()

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute loss only for positive samples
        pos_sim = sim_matrix[pos_mask]  # (num_pos, B)

        # LogSumExp trick for numerical stability
        loss = -pos_sim[:, pos_mask].diag() + torch.logsumexp(pos_sim, dim=1)

        return loss.mean()


class HybridLoss(nn.Module):
    """Hybrid loss combining BCE/Focal with InfoNCE.

    Args:
        bce_weight: Weight for BCE/Focal loss
        infonce_weight: Weight for InfoNCE loss
        use_focal: Use Focal loss instead of BCE
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
        temperature: InfoNCE temperature
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        infonce_weight: float = 0.5,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.infonce_weight = infonce_weight
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.infonce = InfoNCELoss(temperature=temperature)

    def focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sigmoid focal loss."""
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        probs = torch.sigmoid(logits)
        pt = labels * probs + (1 - labels) * (1 - probs)
        alpha_t = labels * self.focal_alpha + (1 - labels) * (1 - self.focal_alpha)
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma
        return (focal_weight * bce_loss).mean()

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute hybrid loss.

        Args:
            logits: (B,) prediction logits
            embeddings: (B, D) output embeddings
            labels: (B,) binary labels

        Returns:
            Total loss and loss components dict
        """
        # Classification loss
        if self.use_focal:
            cls_loss = self.focal_loss(logits, labels)
        else:
            cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Contrastive loss
        contrast_loss = self.infonce(embeddings, labels)

        # Total loss
        total_loss = self.bce_weight * cls_loss + self.infonce_weight * contrast_loss

        loss_dict = {
            'cls_loss': cls_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict
