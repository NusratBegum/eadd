# SPDX-FileCopyrightText: 2026 Nusrat Begum <rupunzelnusrat5@gmail.com>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
"""EADD: Enhanced Adversarial Drift Detection for unsupervised feature drift monitoring."""

from eadd.detector import ExplainableAdversarialDriftDetector

EADD = ExplainableAdversarialDriftDetector

__all__ = ["EADD", "ExplainableAdversarialDriftDetector"]
__version__ = "1.0.0"
