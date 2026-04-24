"""Prediction heads.

All heads are plain PyTorch modules.
"""

from molpot.heads.edge_energy import EdgeEnergyHead
from molpot.heads.heads import AtomicEnergyMLP, EnergyHead, TypeHead
from molpot.heads.rescale import GlobalRescale, PerSpeciesScaleShift

__all__ = [
    "AtomicEnergyMLP",
    "EdgeEnergyHead",
    "EnergyHead",
    "GlobalRescale",
    "PerSpeciesScaleShift",
    "TypeHead",
]
