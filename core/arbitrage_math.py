# core/arbitrage_math.py
from __future__ import annotations
import math
from typing import Sequence
from core.schemas import CulturalSignal, ZoneArbitrageResult


class DemandScorer:
    """Computes cross-domain cultural demand from Qloo affinity and demographic signals."""

    def compute_weighted_demand(
        self,
        signals: Sequence[CulturalSignal],
        demographic_lift: float,
        alpha: float = 0.35,
    ) -> float:
        """Calculates expected demand given entity affinities and a demographic modifier."""
        if not signals:
            return 0.0
        
        base_affinity = sum(s.weight * s.affinity_score for s in signals) / sum(
            s.weight for s in signals
        )
        
        clamped_lift = max(-0.95, min(2.0, demographic_lift))
        return round(base_affinity * (1.0 + alpha * clamped_lift), 4)

    def normalize_saturation(
        self, competitor_affinities: Sequence[float], popularity_weights: Sequence[float]
    ) -> float:
        """Determines physical category saturation using logarithmic scaling."""
        raw_density = sum(
            aff * pop for aff, pop in zip(competitor_affinities, popularity_weights)
        )
        return round(math.log1p(max(0.0, raw_density)), 4)


class ArbitrageCalculator:
    """Synthesizes demand and saturation into a deterministic Cultural Arbitrage Yield score."""

    def __init__(self, demand_scorer: DemandScorer) -> None:
        self._scorer = demand_scorer

    def evaluate_zone(
        self,
        zone_name: str,
        signals: Sequence[CulturalSignal],
        demographic_lift: float,
        competitor_affinities: Sequence[float],
        popularity_weights: Sequence[float],
        saturation_penalty_lambda: float = 0.4,
    ) -> ZoneArbitrageResult:
        """Produces the final mathematical viability score for a physical location."""
        demand = self._scorer.compute_weighted_demand(signals, demographic_lift)
        saturation = self._scorer.normalize_saturation(
            competitor_affinities, popularity_weights
        )
        
        # Prevent division by zero if saturation is zero
        divisor = 1.0 + (saturation_penalty_lambda * saturation)
        arbitrage_yield = demand / divisor
        
        return ZoneArbitrageResult(
            zone_name=zone_name,
            demand_score=demand,
            saturation_index=saturation,
            arbitrage_yield=round(arbitrage_yield, 4),
        )