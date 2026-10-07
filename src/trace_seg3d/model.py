"""MedNeXt with feature-statistics context and Counterfactual Context Transport (CCT).

Why this design
---------------
In the old ``CausalMedNeXt`` the context latent only entered the decoder through a FiLM term
bounded by ``0.1 * tanh`` while every skip connection (which carries the acquisition style)
was left untouched, so ``do(z_c = c)`` barely changed the output and the stability loss taught
the decoder to ignore ``z_c`` entirely -> the CCT instability map was ~0 by construction.

Here the context is *defined* as the per-channel first/second moments of the encoder feature
maps at every scale (the part of a CNN representation known to carry acquisition / style
information), and the disease evidence is the moment-normalised content:

    f_l = sigma_l * f_hat_l + mu_l,        z_c := {(mu_l, sigma_l)}_l,   content := {f_hat_l}_l

CCT keeps the content fixed and replaces the moments with those of a source-domain support
case (AdaIN), at *all* skip levels and the bottleneck, then decodes. The decoder therefore
cannot ignore the intervention, and the variance across transported contexts is a genuine
measure of context sensitivity.

Proxy anchoring (M1) is kept but optional: a context head predicts acquisition proxies from
pooled moments, a disease head predicts region volumes / grade from the normalised bottleneck
content, and gradient-reversed heads discourage leakage between the two.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from causal_mednext.backbone import MedNeXtSegmenter
from causal_mednext.mechanisms.gradient_reversal import gradient_reverse

Stats = list[tuple[Tensor, Tensor]]  # per level (mu [B,C], sigma [B,C])


def feature_moments(feature: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    f = feature.float().flatten(2)
    mu = f.mean(dim=2)
    sigma = (f.var(dim=2, unbiased=False) + eps).sqrt()
    return mu, sigma


def _mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, out))


class TraceMedNeXt(nn.Module):
    def __init__(
        self,
        model_id: str = "S",
        kernel_size: int = 3,
        base_channels: int | None = None,
        num_classes: int = 3,
        transport_levels: Sequence[int] = (0, 1, 2, 3, 4),
        latent_dim: int = 128,
        context_sizes: Sequence[int] = (),
        disease_sizes: Sequence[int] = (),
        proxies: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = MedNeXtSegmenter(
            in_channels=4,
            num_classes=num_classes,
            model_id=model_id,
            kernel_size=kernel_size,
            deep_supervision=False,
            base_channels=base_channels,
        )
        self.channels = tuple(int(c) for c in self.backbone.feature_channels)
        self.transport_levels = tuple(sorted(int(l) for l in transport_levels))
        self.proxies = bool(proxies)
        self.config = {
            "model_id": model_id,
            "kernel_size": kernel_size,
            "base_channels": base_channels,
            "num_classes": num_classes,
            "transport_levels": list(self.transport_levels),
            "latent_dim": latent_dim,
            "context_sizes": list(context_sizes),
            "disease_sizes": list(disease_sizes),
            "proxies": self.proxies,
        }
        stat_dim = 2 * sum(self.channels[l] for l in self.transport_levels)
        bottleneck = self.channels[-1]
        if self.proxies:
            self.context_encoder = nn.Sequential(nn.LayerNorm(stat_dim), _mlp(stat_dim, latent_dim, latent_dim))
            self.disease_encoder = nn.Sequential(nn.LayerNorm(bottleneck), _mlp(bottleneck, latent_dim, latent_dim))
            self.context_heads = nn.ModuleList(_mlp(latent_dim, latent_dim, k) for k in context_sizes)
            self.disease_heads = nn.ModuleList(_mlp(latent_dim, latent_dim, k) for k in disease_sizes)
            self.volume_head = _mlp(latent_dim, latent_dim, 3)
            # adversaries (fed through gradient reversal)
            self.context_from_disease = nn.ModuleList(_mlp(latent_dim, latent_dim, k) for k in context_sizes)
            self.volume_from_context = _mlp(latent_dim, latent_dim, 3)

    # ------------------------------------------------------------------ core pieces
    def encode(self, x: Tensor) -> tuple[tuple[Tensor, ...], Stats]:
        features = tuple(self.backbone.encode_features(x))
        stats = [feature_moments(features[l]) for l in self.transport_levels]
        return features, stats

    def transport(self, features: tuple[Tensor, ...], stats: Stats, target: Stats, alpha: float | Tensor = 1.0) -> tuple[Tensor, ...]:
        """AdaIN: keep normalised content, swap moments with ``target`` (alpha=1 -> full swap)."""

        out = list(features)
        for (mu, sigma), (mu_t, sigma_t), level in zip(stats, target, self.transport_levels):
            f = features[level]
            dtype = f.dtype
            a = alpha if not isinstance(alpha, Tensor) else alpha.view(-1, 1)
            mu_new = a * mu_t + (1 - a) * mu
            sigma_new = a * sigma_t + (1 - a) * sigma
            shape = (f.shape[0], f.shape[1], 1, 1, 1)
            content = (f.float() - mu.view(shape)) / sigma.view(shape)
            out[level] = (content * sigma_new.view(shape) + mu_new.view(shape)).to(dtype)
        return tuple(out)

    def decode(self, features: tuple[Tensor, ...], shape: tuple[int, int, int]) -> Tensor:
        logits = self.backbone.decode_features(features, shape)
        assert isinstance(logits, Tensor)
        return logits

    def latents(self, features: tuple[Tensor, ...], stats: Stats) -> tuple[Tensor, Tensor]:
        flat = torch.cat([torch.cat([mu, torch.log(sigma)], dim=1) for mu, sigma in stats], dim=1)
        z_c = self.context_encoder(flat)
        xb = features[-1].float()
        mu_b, sigma_b = feature_moments(xb)
        content = ((xb.flatten(2) - mu_b.unsqueeze(-1)) / sigma_b.unsqueeze(-1)).abs().mean(dim=2)
        z_d = self.disease_encoder(content)
        return z_d, z_c

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x: Tensor,
        target_stats: Stats | None = None,
        alpha: float | Tensor = 1.0,
        adversary_strength: float = 1.0,
    ) -> dict[str, Tensor | Stats]:
        shape = tuple(x.shape[-3:])
        features, stats = self.encode(x)
        out: dict[str, Tensor | Stats] = {"logits": self.decode(features, shape), "stats": stats}  # type: ignore[arg-type]
        if target_stats is not None:
            out["transported_logits"] = self.decode(self.transport(features, stats, target_stats, alpha), shape)  # type: ignore[arg-type]
        if self.proxies:
            z_d, z_c = self.latents(features, stats)
            out["z_d"], out["z_c"] = z_d, z_c
            out["context_logits"] = [head(z_c) for head in self.context_heads]  # type: ignore[assignment]
            out["disease_logits"] = [head(z_d) for head in self.disease_heads]  # type: ignore[assignment]
            out["volume"] = self.volume_head(z_d)
            z_d_rev = gradient_reverse(z_d, adversary_strength)
            z_c_rev = gradient_reverse(z_c, adversary_strength)
            out["adv_context_logits"] = [head(z_d_rev) for head in self.context_from_disease]  # type: ignore[assignment]
            out["adv_volume"] = self.volume_from_context(z_c_rev)
        return out

    @torch.no_grad()
    def cct(self, x: Tensor, bank: "ContextBank", k: int, selection: str = "diverse") -> dict[str, Tensor]:
        """Counterfactual Context Transport at inference: factual probs, consensus, instability."""

        shape = tuple(x.shape[-3:])
        features, stats = self.encode(x)
        factual = torch.sigmoid(self.decode(features, shape).float())
        probs = []
        for target in bank.select(k, selection=selection, anchor=stats):
            target_b = [(mu.expand(x.shape[0], -1), sigma.expand(x.shape[0], -1)) for mu, sigma in target]
            probs.append(torch.sigmoid(self.decode(self.transport(features, stats, target_b), shape).float()))
        stack = torch.stack(probs, dim=0)
        return {
            "factual": factual,
            "consensus": stack.mean(dim=0),
            "instability": stack.std(dim=0, unbiased=False),
            "transported": stack,
        }


class ContextBank:
    """Moments of SOURCE-domain support cases. Stored per transport level as [N, C] tensors."""

    def __init__(self, mus: list[Tensor], sigmas: list[Tensor], case_ids: list[str] | None = None, dataset: str = "NA") -> None:
        self.mus = [m.float().cpu() for m in mus]
        self.sigmas = [s.float().cpu() for s in sigmas]
        self.case_ids = list(case_ids or [])
        self.dataset = dataset

    def __len__(self) -> int:
        return 0 if not self.mus else int(self.mus[0].shape[0])

    def _flat(self) -> Tensor:
        flat = torch.cat([torch.cat([m, torch.log(s)], 1) for m, s in zip(self.mus, self.sigmas)], 1)
        return (flat - flat.mean(0)) / flat.std(0).clamp_min(1e-6)

    def entry(self, index: int, device: torch.device | None = None) -> Stats:
        return [(m[index : index + 1].to(device), s[index : index + 1].to(device)) for m, s in zip(self.mus, self.sigmas)]

    def sample(self, batch: int, device: torch.device) -> Stats:
        idx = torch.randint(0, len(self), (batch,))
        return [(m[idx].to(device), s[idx].to(device)) for m, s in zip(self.mus, self.sigmas)]

    def select(self, k: int, selection: str = "diverse", anchor: Stats | None = None) -> list[Stats]:
        n = len(self)
        if n == 0:
            raise ValueError("empty context bank")
        k = min(int(k), n)
        device = anchor[0][0].device if anchor else None
        if selection == "diverse":
            flat = self._flat()
            chosen = [int(torch.linalg.vector_norm(flat, dim=1).argmin())]  # most typical case first
            dist = torch.cdist(flat[chosen], flat)[0]
            while len(chosen) < k:
                nxt = int(dist.argmax())
                chosen.append(nxt)
                dist = torch.minimum(dist, torch.cdist(flat[nxt : nxt + 1], flat)[0])
        elif selection == "random":
            chosen = torch.randperm(n)[:k].tolist()
        else:
            raise ValueError(f"unknown selection {selection!r}")
        return [self.entry(i, device) for i in chosen]

    def state(self) -> dict[str, object]:
        return {"mus": self.mus, "sigmas": self.sigmas, "case_ids": self.case_ids, "dataset": self.dataset}

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "ContextBank":
        return cls(state["mus"], state["sigmas"], state.get("case_ids"), str(state.get("dataset", "NA")))  # type: ignore[arg-type]


class StatsQueue:
    """FIFO of detached moments from recent training cases (used as transport targets while training)."""

    def __init__(self, size: int = 64) -> None:
        self.size = size
        self.mus: list[Tensor] | None = None
        self.sigmas: list[Tensor] | None = None

    def __len__(self) -> int:
        return 0 if self.mus is None else int(self.mus[0].shape[0])

    def push(self, stats: Stats) -> None:
        mus = [m.detach().float() for m, _ in stats]
        sigmas = [s.detach().float() for _, s in stats]
        if self.mus is None:
            self.mus, self.sigmas = mus, sigmas
        else:
            assert self.sigmas is not None
            self.mus = [torch.cat([a, b])[-self.size :] for a, b in zip(self.mus, mus)]
            self.sigmas = [torch.cat([a, b])[-self.size :] for a, b in zip(self.sigmas, sigmas)]

    def sample(self, batch: int, exclude_last: int = 0) -> Stats:
        assert self.mus is not None and self.sigmas is not None
        n = len(self) - exclude_last
        idx = torch.randint(0, max(n, 1), (batch,), device=self.mus[0].device)
        return [(m[idx], s[idx]) for m, s in zip(self.mus, self.sigmas)]


def build_model(config: dict) -> TraceMedNeXt:
    return TraceMedNeXt(**config)


@torch.no_grad()
def build_bank(model: TraceMedNeXt, loader, device: torch.device, dataset: str, amp: bool = False) -> ContextBank:
    model.eval()
    mus: list[list[Tensor]] = [[] for _ in model.transport_levels]
    sigmas: list[list[Tensor]] = [[] for _ in model.transport_levels]
    ids: list[str] = []
    for batch in loader:
        x = batch["image"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            _, stats = model.encode(x)
        for i, (mu, sigma) in enumerate(stats):
            mus[i].append(mu.float().cpu())
            sigmas[i].append(sigma.float().cpu())
        ids += list(batch["case_id"])
    return ContextBank([torch.cat(m) for m in mus], [torch.cat(s) for s in sigmas], ids, dataset)


__all__ = [
    "ContextBank",
    "StatsQueue",
    "TraceMedNeXt",
    "build_bank",
    "build_model",
    "feature_moments",
]
