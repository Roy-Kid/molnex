"""PiNet polarizability and charge-response prediction model.

Supports variants: ``localchi``, ``local``, ``etainv``, ``eem``, ``acks2``.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems", JCTC 2025.
    https://doi.org/10.1021/acs.jctc.4c01570
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config
from molix.data.types import GraphBatch

ANG2BOHR = 1.8897259886


def _pool_layer(features: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return features.mean(dim=1)
    if reduction == "sum":
        return features.sum(dim=1)
    if reduction == "last":
        return features[:, -1]
    raise ValueError(f"Unknown reduction {reduction!r}.")


def _num_graphs(batch: torch.Tensor) -> int:
    return int(batch.max().item()) + 1 if batch.numel() else 0


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    if src.numel() == 0:
        return out
    expand_index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, expand_index, src)


def _relative_atom_indices(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    num = _num_graphs(batch)
    counts = torch.zeros(num, dtype=torch.long, device=batch.device)
    counts.scatter_add_(0, batch, torch.ones_like(batch))
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    rel = torch.arange(batch.shape[0], device=batch.device) - offsets[batch]
    return rel, counts, int(counts.max().item()) if counts.numel() else 0


def _dense_positions(pos: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rel, counts, nmax = _relative_atom_indices(batch)
    num = counts.shape[0]
    dense = torch.zeros(num, nmax, 3, dtype=pos.dtype, device=pos.device)
    dense[batch, rel] = pos
    return dense, counts


def _recompute_edges(batch: GraphBatch) -> None:
    pos = batch["atoms", "pos"]
    ei = batch["edges", "edge_index"]
    diff = pos[ei[:, 1]] - pos[ei[:, 0]]
    batch["edges", "bond_diff"] = diff
    batch["edges", "bond_dist"] = diff.norm(dim=-1).clamp(min=1e-8)


class PiNetPolarizability(nn.Module):
    """PiNet charge-response and polarizability prediction model.

    Variants:

    * ``"localchi"``: symmetric local charge-response kernel.
    * ``"local"``: local polarizability basis with localchi kernel.
    * ``"etainv"``: direct positive ``eta^-1`` construction.
    * ``"eem"``: electronegativity-equalization kernel.
    * ``"acks2"``: ACKS2 Dyson update.

    Args:
        encoder: :class:`~molzoo.PiNet` encoder.
        atom_types: Supported atomic numbers for sigma parameters.
        variant: Charge-response kernel variant.
        iso: If True, add isotropic atomic polarizability term.
        hidden_dim: Hidden dimension for MLPs.
        layer_reduction: How to pool across GC-block layers.
        epsilon: Regularization for eta-inverse construction.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        atom_types: list[int] | None = None,
        variant: str = "localchi",
        iso: bool = False,
        hidden_dim: int = 64,
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        epsilon: float = 0.01,
        sigma: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant.endswith("_iso"):
            variant = variant[:-4]
            iso = True
        if variant not in {"localchi", "local", "etainv", "eem", "acks2"}:
            raise ValueError(f"Unsupported PiNet polarizability variant {variant!r}.")
        atom_types = atom_types or [1, 6, 7, 8]

        self.encoder = encoder
        self.variant = variant
        self.iso = iso
        self.layer_reduction = layer_reduction
        self.epsilon = float(epsilon)

        input_dim: int = getattr(encoder, "output_dim", 16)
        edge_dim: int = getattr(encoder, "edge_output_dim", input_dim)

        self.atom_diag_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.edge_scalar_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.edge_vector_mlp = nn.Linear(input_dim, 1, dtype=config.ftype)
        if self.iso:
            self.iso_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
            )

        _default_sigma = {1: 0.312, 6: 0.730, 7: 0.709, 8: 0.661, 16: 1.048, 17: 1.016}
        sigma = sigma or _default_sigma
        init_sigma = [float(sigma.get(z, 0.7)) for z in atom_types]
        self.register_buffer("atom_types", torch.tensor(atom_types, dtype=torch.long), persistent=False)
        self.atom_types: torch.Tensor
        self.sigma_raw = nn.Parameter(torch.tensor(init_sigma, dtype=config.ftype))

    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        _recompute_edges(batch)
        batch = self.encoder(batch)

        pos = batch["atoms", "pos"]
        atom_batch = batch["atoms", "batch"]
        edge_index = batch["edges", "edge_index"]
        node = _pool_layer(batch["atoms", "node_features"], self.layer_reduction)
        i1 = _pool_layer(batch["edges", "i1_features"], self.layer_reduction)

        atom_diag = self.atom_diag_mlp(node).squeeze(-1)
        edge_response = self.edge_scalar_mlp(i1).squeeze(-1)
        if "i3_features" in batch["edges"].keys():
            i3 = _pool_layer(batch["edges", "i3_features"], self.layer_reduction)
            edge_response = edge_response + self.edge_vector_mlp(i3.square().sum(dim=1)).squeeze(-1)

        rel, counts, nmax = _relative_atom_indices(atom_batch)
        atom_mask = torch.arange(nmax, device=pos.device).unsqueeze(0) < counts.unsqueeze(1)
        sigma = self._sigma_for_atoms(batch["atoms", "Z"])

        extra: dict[str, torch.Tensor] = {}

        if self.variant == "local":
            chi = self._make_local_chi(edge_index, edge_response, atom_batch, rel, counts, nmax)
            alpha = self._local_alpha(batch, edge_response)
        elif self.variant == "localchi":
            chi = self._make_local_chi(edge_index, edge_response, atom_batch, rel, counts, nmax)
            alpha = self._chi_to_alpha(pos, atom_batch, chi)
        elif self.variant == "etainv":
            chi, eta_inv = self._make_etainv(
                atom_diag, edge_index, edge_response, atom_batch, rel, counts, nmax,
            )
            alpha = self._chi_to_alpha(pos, atom_batch, chi)
            extra["eta_inv"] = eta_inv
        elif self.variant == "eem":
            chi, eta = self._make_eem(atom_diag, sigma, pos, atom_batch, counts, nmax)
            alpha = self._chi_to_alpha(pos, atom_batch, chi)
            extra["eta"] = eta
        else:  # acks2
            chi_s = self._make_local_chi(edge_index, edge_response, atom_batch, rel, counts, nmax)
            eta = self._make_eta(atom_diag, sigma, pos, atom_batch, counts, nmax)
            eye = torch.eye(nmax, dtype=eta.dtype, device=eta.device).expand_as(eta)
            system = eye - eta @ chi_s
            chi = torch.linalg.solve(system.transpose(-1, -2), chi_s)
            alpha = self._chi_to_alpha(pos, atom_batch, chi)
            extra["eta_e"] = eta
            extra["chi_s"] = chi_s

        if self.iso:
            alpha_iso_atom = F.softplus(self.iso_mlp(node).squeeze(-1))
            alpha_iso = _scatter_sum(alpha_iso_atom, atom_batch, counts.shape[0])
            eye3 = torch.eye(3, dtype=alpha.dtype, device=alpha.device)
            alpha_iso_tensor = alpha_iso[:, None, None] * eye3[None]
            alpha = alpha + alpha_iso_tensor
            extra["alpha_iso"] = alpha_iso_tensor

        return {
            "alpha": alpha, "chi": chi, "atom_mask": atom_mask,
            "atom_diag": atom_diag, "edge_response": edge_response, **extra,
        }

    # -- internal helpers -------------------------------------------------------

    def _sigma_for_atoms(self, Z: torch.Tensor) -> torch.Tensor:
        matches = Z.long().unsqueeze(-1) == self.atom_types.unsqueeze(0)
        sigma_table = self.sigma_raw.abs().clamp(min=1e-8)
        sigma = matches.to(dtype=sigma_table.dtype) @ sigma_table
        fallback = sigma_table.mean()
        return torch.where(matches.any(dim=1), sigma, fallback.expand_as(sigma))

    def _make_local_chi(self, edge_index, edge_response, atom_batch, rel, counts, nmax):
        num_graphs = counts.shape[0]
        chi = torch.zeros(num_graphs, nmax, nmax, dtype=edge_response.dtype, device=edge_response.device)
        if edge_index.numel() == 0:
            return chi
        src, dst = edge_index[:, 0], edge_index[:, 1]
        b = atom_batch[src]
        ri, rj = rel[src], rel[dst]
        y = edge_response.abs()
        chi.index_put_((b, ri, rj), y, accumulate=True)
        chi.index_put_((b, rj, ri), y, accumulate=True)
        diag = -chi.sum(dim=1)
        idx = torch.arange(nmax, device=chi.device)
        chi[:, idx, idx] = diag
        return chi

    def _make_etainv(self, atom_diag, edge_index, edge_response, atom_batch, rel, counts, nmax):
        num = counts.shape[0]
        m = torch.zeros(num, nmax, nmax, dtype=atom_diag.dtype, device=atom_diag.device)
        idx = torch.arange(nmax, device=atom_diag.device)
        diag_dense = torch.zeros(num, nmax, dtype=atom_diag.dtype, device=atom_diag.device)
        diag_dense[atom_batch, rel] = atom_diag
        m[:, idx, idx] = diag_dense
        if edge_index.numel() > 0:
            src, dst = edge_index[:, 0], edge_index[:, 1]
            b = atom_batch[src]
            m.index_put_((b, rel[src], rel[dst]), edge_response, accumulate=True)
        eta_inv = m.transpose(-1, -2) @ m
        active_eye = torch.diag_embed(
            (idx.unsqueeze(0) < counts.unsqueeze(1)).to(dtype=atom_diag.dtype)
        )
        eta_inv = eta_inv + self.epsilon * active_eye
        chi = self._make_lrf(eta_inv)
        return chi, eta_inv

    def _make_eem(self, atom_diag, sigma, pos, atom_batch, counts, nmax):
        eta = self._make_eta(atom_diag, sigma, pos, atom_batch, counts, nmax)
        chi_blocks = []
        for b_idx, n in enumerate(counts.tolist()):
            block = eta[b_idx, :n, :n]
            inv = torch.linalg.inv(block)
            chi_blocks.append(self._make_lrf(inv.unsqueeze(0))[0])
        padded_chi = torch.zeros_like(eta)
        for b_idx, chi_b in enumerate(chi_blocks):
            n = chi_b.shape[0]
            padded_chi[b_idx, :n, :n] = chi_b
        return padded_chi, eta

    def _make_eta(self, atom_diag, sigma, pos, atom_batch, counts, nmax):
        dense_pos, _ = _dense_positions(pos, atom_batch)
        dense_sigma = torch.zeros(counts.shape[0], nmax, dtype=sigma.dtype, device=sigma.device)
        rel, _, _ = _relative_atom_indices(atom_batch)
        dense_sigma[atom_batch, rel] = sigma
        r_ij = dense_pos[:, None, :, :] - dense_pos[:, :, None, :]
        r = r_ij.norm(dim=-1)
        gamma = torch.sqrt(dense_sigma[:, None, :].square() + dense_sigma[:, :, None].square())
        eta = torch.special.erf(r / gamma.clamp(min=1e-8) / math.sqrt(2.0)) / r.clamp(min=1e-8)
        eta = eta / ANG2BOHR

        idx = torch.arange(nmax, device=pos.device)
        diag = torch.zeros(counts.shape[0], nmax, dtype=pos.dtype, device=pos.device)
        diag[atom_batch, rel] = atom_diag.abs()
        sigma_diag = 1.0 / (math.sqrt(math.pi) * dense_sigma.clamp(min=1e-8)) / ANG2BOHR
        diag = diag + sigma_diag
        eta[:, idx, idx] = diag
        active = idx.unsqueeze(0) < counts.unsqueeze(1)
        eta = eta * (active[:, :, None] & active[:, None, :]).to(dtype=eta.dtype)
        return eta

    @staticmethod
    def _make_lrf(ainv: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
        rows = ainv.sum(dim=-1)
        outer = rows.unsqueeze(-1) * rows.unsqueeze(-2)
        denom = rows.sum(dim=-1).clamp(min=eps)
        return -ainv + outer / denom[:, None, None]

    def _chi_to_alpha(self, pos, atom_batch, chi):
        dense_pos, _ = _dense_positions(pos, atom_batch)
        dense_pos = dense_pos * ANG2BOHR
        return -torch.einsum("bix,bij,bjy->bxy", dense_pos, chi, dense_pos)

    def _local_alpha(self, batch, edge_response):
        pos = batch["atoms", "pos"]
        atom_batch = batch["atoms", "batch"]
        edge_index = batch["edges", "edge_index"]
        num = _num_graphs(atom_batch)
        if edge_index.numel() == 0:
            return torch.zeros(num, 3, 3, dtype=pos.dtype, device=pos.device)
        src = edge_index[:, 0]
        edge_vec = batch["edges", "bond_diff"] * ANG2BOHR
        tmp = edge_response.unsqueeze(-1) * edge_vec
        atom_tmp = _scatter_sum(tmp, src, pos.shape[0])
        alpha_i = atom_tmp.unsqueeze(-1) * atom_tmp.unsqueeze(-2)
        return _scatter_sum(alpha_i, atom_batch, num)
