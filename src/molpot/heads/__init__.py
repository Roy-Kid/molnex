"""Prediction heads.

All heads are plain PyTorch modules.
"""

from molpot.heads.edge_energy import EdgeEnergyHead
from molpot.heads.electrostatics import (
    HardnessHead,
    HardnessHeadSpec,
    PolarizabilityHead,
    PolarizabilityHeadSpec,
)
from molpot.heads.element_baselines import (
    ElementAlphaTable,
    ElementChargeTable,
)
from molpot.heads.heads import AtomicEnergyMLP, EnergyHead, TypeHead
from molpot.heads.multipole import (
    PermMultipoleHead,
    PermMultipoleHeadSpec,
)
from molpot.heads.rescale import GlobalRescale, PerSpeciesScaleShift

__all__ = [
    "AtomicEnergyMLP",
    "EdgeEnergyHead",
    "ElementAlphaTable",
    "ElementChargeTable",
    "EnergyHead",
    "GlobalRescale",
    "HardnessHead",
    "HardnessHeadSpec",
    "PermMultipoleHead",
    "PermMultipoleHeadSpec",
    "PolarizabilityHead",
    "PolarizabilityHeadSpec",
    "PerSpeciesScaleShift",
    "TypeHead",
]
