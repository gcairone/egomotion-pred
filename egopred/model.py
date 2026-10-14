"""Temporal model over frozen per-frame features, with a multimodal trajectory head.

Variants (always reported side by side, with B0):
  A: kinematics only (ego history + current dynamics + driving command)
  B: vision only, plus current speed and driving command
  C: vision + kinematics
Vision tokens per history frame: the CLS vector and, unless vision="cls", the pooled patch
grid, with learned frame-time and grid-position embeddings. K learned queries attend to the
encoded tokens; each yields a trajectory (FUT x [x, y, yaw]) and a logit. Loss: winner-takes-
all regression on the hypothesis closest to the ground truth (ADE) + cross-entropy that
teaches the logits to pick that hypothesis.
With anchors=True each query k predicts an offset from a fixed anchor trajectory (k-means of the
training futures, `fit_anchors`), and the winner is the anchor closest to the ground truth: the
classification target no longer depends on the predictions, so it is stable, and modes that
lose cannot drift away.
With future=(dt, ...) (vision variants) an auxiliary head, used only in training, predicts from
the encoded tokens how the features of the frames at t0 + dt change from the current frame
(CLS, and grid cells for vision="grid"); future_cond also feeds it the ground-truth trajectory.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .trajectory import FUT, HIST

VARIANTS = {"A": (True, False), "B": ("speed", True), "C": (True, True)}  # (kinematics, vision)


@dataclass
class ModelConfig:
    variant: str = "C"
    vision: str = "grid"  # "grid": CLS + patch grid tokens; "cls": CLS only
    k: int = 6
    d: int = 256
    layers: int = 4
    decoder_layers: int = 2
    heads: int = 8
    dropout: float = 0.1
    xy_scale: float = 5.0  # m; regression loss on xy / xy_scale
    yaw_weight: float = 1.0
    cls_weight: float = 0.5
    anchors: bool = False  # offsets from k-means anchor trajectories, winner = nearest anchor
    future: tuple = ()  # s after t0 of the frames whose feature change the auxiliary head predicts
    future_weight: float = 0.1
    future_cond: bool = False  # condition the auxiliary head on the ground-truth trajectory

    def dict(self) -> dict:
        return asdict(self)


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class TrajectoryModel(nn.Module):
    def __init__(self, cfg: ModelConfig, kin_dim: int, cls_dim: int = 0, grid_dim: int = 0, n_grid: int = 0):
        super().__init__()
        self.cfg = cfg
        _, self.use_vision = VARIANTS[cfg.variant]
        self.use_grid = self.use_vision and cfg.vision == "grid"
        d = cfg.d
        self.kin = _mlp(kin_dim, d, d)
        if self.use_vision:
            self.cls_proj = nn.Linear(cls_dim, d)
            self.time = nn.Parameter(torch.randn(HIST, d) * 0.02)
        if self.use_grid:
            self.grid_proj = nn.Linear(grid_dim, d)
            self.grid_pos = nn.Parameter(torch.randn(n_grid, d) * 0.02)

        def layer(cls):
            return cls(d, cfg.heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(layer(nn.TransformerEncoderLayer), cfg.layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(cfg.k, d) * 0.02)
        self.decoder = nn.TransformerDecoder(layer(nn.TransformerDecoderLayer), cfg.decoder_layers)
        self.norm = nn.LayerNorm(d)
        self.traj = _mlp(d, d, FUT * 3)
        self.logit = nn.Linear(d, 1)
        if cfg.future and self.use_vision:
            n_fut = len(cfg.future)
            self.fut_queries = nn.Parameter(torch.randn(n_fut, 1 + (n_grid if self.use_grid else 0), d) * 0.02)
            self.fut_decoder = nn.TransformerDecoder(layer(nn.TransformerDecoderLayer), 1)
            self.fut_cls = nn.Linear(d, cls_dim)
            self.fut_grid = nn.Linear(d, grid_dim) if self.use_grid else None
            self.fut_cond = _mlp(FUT * 3, d, d) if cfg.future_cond else None
        # zeros until set_anchors(); saved in the state dict, so checkpoints carry them
        self.register_buffer("anchors", torch.zeros(cfg.k, FUT, 3))

    def set_anchors(self, anchors: np.ndarray) -> None:
        self.anchors.copy_(torch.as_tensor(anchors, dtype=self.anchors.dtype))

    def forward(self, kin: torch.Tensor, cls: torch.Tensor | None = None, grid: torch.Tensor | None = None,
                keep_frames: torch.Tensor | None = None, keep_cells: torch.Tensor | None = None,
                return_memory: bool = False):
        """kin (B, kin_dim); cls (B, HIST, cls_dim); grid (B, HIST, n_grid, grid_dim).
        keep_frames (B, HIST) / keep_cells (B, HIST, n_grid), bool, optional (training-time token
        dropout): dropped visual tokens are masked out of the attention.
        Returns trajectories (B, K, FUT, 3) in metres/radians and logits (B, K), plus the encoded
        tokens and their padding mask (or None) with return_memory."""
        b = kin.shape[0]
        tokens = [self.kin(kin)[:, None]]
        drop = [torch.zeros(b, 1, dtype=torch.bool, device=kin.device)]
        if self.use_vision:
            tokens.append(self.cls_proj(cls) + self.time)
            kf = keep_frames if keep_frames is not None else torch.ones(b, HIST, dtype=torch.bool, device=kin.device)
            drop.append(~kf)
        if self.use_grid:
            g = self.grid_proj(grid) + self.grid_pos[None, None] + self.time[None, :, None]
            tokens.append(g.flatten(1, 2))
            kc = kf[..., None].expand(-1, -1, grid.shape[2])
            if keep_cells is not None:
                kc = kc & keep_cells
            drop.append(~kc.flatten(1, 2))
        mask = torch.cat(drop, dim=1) if keep_frames is not None or keep_cells is not None else None
        memory = self.encoder(torch.cat(tokens, dim=1), src_key_padding_mask=mask)
        h = self.norm(self.decoder(self.queries.expand(b, -1, -1), memory, memory_key_padding_mask=mask))
        traj = self.traj(h).view(kin.shape[0], self.cfg.k, FUT, 3)
        scale = torch.tensor([self.cfg.xy_scale, self.cfg.xy_scale, 1.0], device=traj.device, dtype=traj.dtype)
        traj = traj * scale
        if self.cfg.anchors:
            traj = traj + self.anchors.to(traj.dtype)
        if return_memory:
            return traj, self.logit(h).squeeze(-1), memory, mask
        return traj, self.logit(h).squeeze(-1)

    def predict_future(self, memory: torch.Tensor, mask: torch.Tensor | None, gt: torch.Tensor | None = None):
        """Auxiliary head: predicted feature change of the future frames, CLS (B, n_fut, cls_dim)
        and grid (B, n_fut, n_grid, grid_dim) or None. gt (B, FUT, 3) is used with future_cond."""
        b = memory.shape[0]
        n_fut, n_tok, d = self.fut_queries.shape
        q = self.fut_queries.reshape(1, n_fut * n_tok, d).expand(b, -1, -1)
        if self.fut_cond is not None:
            scale = torch.tensor([self.cfg.xy_scale, self.cfg.xy_scale, 1.0], device=gt.device, dtype=gt.dtype)
            q = q + self.fut_cond((gt / scale).flatten(1)).to(q.dtype)[:, None]
        h = self.fut_decoder(q, memory, memory_key_padding_mask=mask).view(b, n_fut, n_tok, d)
        return self.fut_cls(h[:, :, 0]), self.fut_grid(h[:, :, 1:]) if self.fut_grid is not None else None


def fit_anchors(fut: np.ndarray, k: int, iters: int = 50, max_samples: int = 50000, seed: int = 0) -> np.ndarray:
    """k-means (k-means++ init, deterministic) of future trajectories (N, FUT, 3) on their xy,
    as anchors (k, FUT, 3): xy and yaw are the cluster means."""
    rng = np.random.default_rng(seed)
    if len(fut) > max_samples:
        fut = fut[rng.choice(len(fut), max_samples, replace=False)]
    x = fut[..., :2].reshape(len(fut), -1).astype(np.float64)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = np.min([((x - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    c = np.stack(centers)
    for _ in range(iters):
        lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
    return np.stack([fut[lab == j].mean(0) if (lab == j).any() else np.concatenate(
        [c[j].reshape(FUT, 2), np.zeros((FUT, 1))], 1) for j in range(k)]).astype(np.float32)


def wta_loss(traj: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, cfg: ModelConfig,
             anchors: torch.Tensor | None = None):
    """Winner-takes-all on the hypothesis with the lowest ADE, plus cross-entropy on its index.
    With `anchors` (K, FUT, 3) the winner is the anchor closest to the ground truth instead."""
    ade = (traj[..., :2] - target[:, None, :, :2]).norm(dim=-1).mean(-1)  # (B, K)
    if anchors is not None:
        best = (anchors[None, ..., :2] - target[:, None, :, :2]).norm(dim=-1).mean(-1).argmin(dim=1)
    else:
        best = ade.argmin(dim=1)
    sel = traj[torch.arange(len(best), device=traj.device), best]
    reg = F.smooth_l1_loss(sel[..., :2] / cfg.xy_scale, target[..., :2] / cfg.xy_scale)
    yaw = F.smooth_l1_loss(sel[..., 2], target[..., 2])
    ce = F.cross_entropy(logits.float(), best)
    loss = reg + cfg.yaw_weight * yaw + cfg.cls_weight * ce
    return loss, {"reg": reg.detach(), "yaw": yaw.detach(), "ce": ce.detach(),
                  "min_ade": ade.detach().min(dim=1).values.mean()}
