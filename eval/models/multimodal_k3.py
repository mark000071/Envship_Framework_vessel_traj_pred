"""Multi-modal K=3 prediction head for long-horizon Track B.

The 60-minute horizon is intrinsically multi-modal (a ferry could go to
several different ports; a fishing vessel could turn back).  Single-modal
ADE underestimates the model's predictive ability by averaging across
mutually-exclusive futures.

This module wraps a base model (TCN / LSTM+Env-SDF / LSTM+Social+Env-SDF)
with a K=3 mixture head that produces three candidate trajectories per
sample, each anchored on a learned offset from a k-means anchor centroid.

Training:
  - Pre-compute K=3 anchors via k-means on training-set 60-min endpoints
    (in normalized planar coords).
  - At each step, the base model produces a shared feature h_t.
  - Three lightweight heads predict (Δx, Δy) offsets from each anchor's
    extrapolation.
  - One classifier head predicts an unnormalized logit p_k per anchor.
  - Loss: best-of-K Huber on the trajectory whose anchor is closest to GT
    end, plus a CE on the classifier targeting argmin_k ‖endpoint_k - GT‖.

Inference: report minADE@3 and minFDE@3 alongside ADE@1 (best classified)
and FDE@1.

This is the ONLY architectural extension allowed beyond the v3 paper
(per R3 of CC_prompts_lab.md, frame in writing as ``preliminary
multi-modal diagnostic``).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


K_ANCHORS = 3


class MultimodalK3Head(nn.Module):
    """Replace the 2-d displacement head of a base model with a K=3 mixture."""

    def __init__(self, feat_dim: int, anchors: torch.Tensor, k: int = K_ANCHORS):
        super().__init__()
        self.k = k
        # Anchors: (K, 2) — relative endpoint coords in normalized space
        self.register_buffer("anchors", anchors.clone().float())
        # Per-anchor displacement head + classifier
        self.offset_heads = nn.ModuleList([
            nn.Linear(feat_dim, 2) for _ in range(k)
        ])
        self.classifier = nn.Linear(feat_dim, k)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feat: (B, feat_dim).  Returns:
          endpoints: (B, K, 2)   — K candidate endpoints (anchor + offset)
          logits:    (B, K)       — classifier logits
        """
        offsets = torch.stack([h(feat) for h in self.offset_heads], dim=1)  # (B,K,2)
        endpoints = self.anchors.unsqueeze(0) + offsets                       # (B,K,2)
        logits = self.classifier(feat)                                        # (B,K)
        return endpoints, logits


def compute_anchors_kmeans(future_endpoints: torch.Tensor, k: int = K_ANCHORS,
                            n_iters: int = 25) -> torch.Tensor:
    """Run k-means on (N, 2) future endpoints; return (K, 2) centroids.

    Used once at training-startup to seed the K anchors from the training
    distribution.  Detached from training graph (registered as a buffer).
    """
    N = future_endpoints.shape[0]
    perm = torch.randperm(N)[:k]
    centroids = future_endpoints[perm].clone()
    for _ in range(n_iters):
        # E-step: assign each point to nearest centroid
        dists = torch.cdist(future_endpoints, centroids)            # (N, K)
        assign = dists.argmin(dim=-1)                                # (N,)
        # M-step: update centroids
        new_centroids = torch.stack([
            future_endpoints[assign == k_i].mean(dim=0) if (assign == k_i).any()
            else centroids[k_i] for k_i in range(k)])
        if torch.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids
    return centroids


def best_of_k_loss(endpoints_pred: torch.Tensor,   # (B, K, 2)
                    logits: torch.Tensor,           # (B, K)
                    target_end: torch.Tensor,       # (B, 2)
                    huber_delta: float = 1.0,
                    cls_weight: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Best-of-K Huber loss + classifier CE on argmin anchor.

    Returns (total_loss, k_assignment).
    """
    # Distance from each anchor's prediction to GT endpoint
    diffs = endpoints_pred - target_end.unsqueeze(1)                # (B,K,2)
    dists = diffs.norm(dim=-1)                                      # (B,K)
    k_star = dists.argmin(dim=-1)                                   # (B,)
    # Per-sample huber on the closest anchor's predicted endpoint
    sel_pred = endpoints_pred.gather(1, k_star.view(-1,1,1).expand(-1,1,2)).squeeze(1)
    huber = F.huber_loss(sel_pred, target_end, delta=huber_delta, reduction="mean")
    ce = F.cross_entropy(logits, k_star)
    return huber + cls_weight * ce, k_star


def min_ade_k(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute minADE@K.
    predictions: (B, K, T, 2)
    target:      (B, T, 2)
    Returns minADE per sample (B,).
    """
    diffs = predictions - target.unsqueeze(1)            # (B, K, T, 2)
    ade_per_k = diffs.norm(dim=-1).mean(dim=-1)          # (B, K)
    return ade_per_k.min(dim=-1).values                  # (B,)


def min_fde_k(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute minFDE@K (final-step displacement).
    predictions: (B, K, T, 2)
    target:      (B, T, 2)
    """
    final_diffs = predictions[..., -1, :] - target[:, -1:, :]
    fde_per_k = final_diffs.norm(dim=-1).squeeze(-1)     # (B, K)
    return fde_per_k.min(dim=-1).values                  # (B,)
