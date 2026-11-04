"""HetICS: heterogeneity-guided cross-center invariant covariate selection for multi-center causal effect estimation.

fit_ra    HetICS-RA for a continuous exposure and outcome
fit_aipw  HetICS-AIPW for a binary exposure and outcome
"""
from ._binary import fit_aipw
from ._continuous import fit_ra

__version__ = "0.1.0"
__all__ = ["fit_ra", "fit_aipw"]
