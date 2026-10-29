# core/citation_validator.py
from __future__ import annotations
from typing import Mapping, Sequence
from core.schemas import AgentClaim, ValidationVerdict


class GroundingAuditor:
    """Enforces zero-hallucination policy by verifying every claim against Qloo entity IDs."""

    def verify_claim_set(
        self,
        claims: Sequence[AgentClaim],
        verified_qloo_entities: Mapping[str, float],
    ) -> ValidationVerdict:
        """Audits LLM claims against empirical Qloo graph data."""
        rejected_ids: list[str] = []
        for claim in claims:
            if claim.qloo_entity_id not in verified_qloo_entities:
                rejected_ids.append(claim.claim_id)
                continue
            
            expected_score = verified_qloo_entities[claim.qloo_entity_id]
            # Ensure the affinity score cited by the LLM matches the graph exactly
            if abs(claim.cited_affinity - expected_score) > 1e-3:
                rejected_ids.append(claim.claim_id)
                
        return ValidationVerdict(
            is_valid=len(rejected_ids) == 0,
            rejected_claim_ids=tuple(rejected_ids),
        )

    def compute_counterfactual_delta(
        self,
        ungrounded_llm_ids: Sequence[str],
        qloo_ranked_map: Mapping[str, float],
        top_k_qloo_mean: float,
    ) -> float:
        """Calculates the exact quantitative miss-rate of using an ungrounded LLM."""
        if not ungrounded_llm_ids:
            return round(top_k_qloo_mean, 4)
            
        llm_scores = [qloo_ranked_map.get(eid, 0.0) for eid in ungrounded_llm_ids]
        llm_mean = sum(llm_scores) / len(llm_scores)
        
        # Positive delta indicates how much better the Qloo data is vs ungrounded LLM
        return round(top_k_qloo_mean - llm_mean, 4)