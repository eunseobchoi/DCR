"""Paired, site-stratified cluster bootstrap for the primary comparison.

Order of aggregation: 3 seeds per image -> image mean per cluster -> paired
cluster resampling within each external site -> equal-weight site average.
The CI covers cluster sampling at C5/C6 only; seed spread is reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import DCRError

EXTERNAL_SITES = ("C5", "C6")


@dataclass(frozen=True)
class BootstrapResult:
    delta: float
    ci_low: float
    ci_high: float
    site_delta: dict
    n_clusters: dict
    n_boot: int

    def verdicts(self, target_gain: float = 0.010, max_site_drop: float = 0.005) -> dict:
        """Three separate judgements; they can disagree and are reported separately."""
        return {
            "statistics_success": self.ci_low > 0,
            "engineering_target": self.delta >= target_gain,
            "per_site_drop_exceeds_guardrail": any(d < -max_site_drop for d in self.site_delta.values()),
        }


def group_scores(records: pd.DataFrame, n_seeds: int = 3) -> pd.DataFrame:
    """records: site_id, group_id, sample_id, method, seed, dice -> one row per (method, site, group)."""
    pos = records.dropna(subset=["dice"])
    counts = pos.groupby(["method", "sample_id"]).seed.nunique()
    if (counts != n_seeds).any():
        raise DCRError("INCOMPLETE_SCORES", f"some images lack {n_seeds} seeds: {counts[counts != n_seeds].index[:5].tolist()}")
    per_image = pos.groupby(["method", "site_id", "group_id", "sample_id"], as_index=False).dice.mean()
    return per_image.groupby(["method", "site_id", "group_id"], as_index=False).dice.mean()


def paired_cluster_bootstrap(groups: pd.DataFrame, method_a: str = "M07", method_b: str = "M05",
                             sites=EXTERNAL_SITES, n_boot: int = 2000, seed: int = 1027) -> BootstrapResult:
    wide = groups.pivot_table(index=["site_id", "group_id"], columns="method", values="dice")
    if method_a not in wide or method_b not in wide or wide[[method_a, method_b]].isna().any().any():
        raise DCRError("INCOMPLETE_SCORES", f"every cluster needs both {method_a} and {method_b}")
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot)
    site_delta, n_clusters = {}, {}
    for site in sites:
        diff = (wide.loc[site, method_a] - wide.loc[site, method_b]).to_numpy()
        site_delta[site] = float(diff.mean())
        n_clusters[site] = len(diff)
        draws = rng.integers(0, len(diff), size=(n_boot, len(diff)))  # same clusters for both methods
        boot += diff[draws].mean(axis=1) / len(sites)
    low, high = np.quantile(boot, [0.025, 0.975])
    return BootstrapResult(float(np.mean(list(site_delta.values()))), float(low), float(high),
                           site_delta, n_clusters, n_boot)
