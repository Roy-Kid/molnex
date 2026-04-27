"""Prediction heads.

All heads are plain PyTorch modules.
"""

from molpot.heads.edge_energy import EdgeEnergyHead
from molpot.heads.heads import AtomicEnergyMLP, EnergyHead, TypeHead
from molpot.heads.multipole import (
    VALID_DAMPINGS,
    VALID_ENERGY_TERMS,
    PermMultipoleHead,
    PermMultipoleHeadSpec,
)
from molpot.heads.rescale import GlobalRescale, PerSpeciesScaleShift

__all__ = [
    "AtomicEnergyMLP",
    "EdgeEnergyHead",
    "EnergyHead",
    "GlobalRescale",
    "PermMultipoleHead",
    "PermMultipoleHeadSpec",
    "PerSpeciesScaleShift",
    "TypeHead",
    "VALID_DAMPINGS",
    "VALID_ENERGY_TERMS",
]
