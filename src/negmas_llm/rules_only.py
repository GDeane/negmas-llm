"""A negotiator that is told the rules of the game and its own payoffs, nothing more.

Unlike the default class prompts from
`~negmas_llm.negotiator.LLMNegotiator`, these templates contain no strategy advice (e.g. "open strong", "concede slowly", "never accept below your reserve value", ...).

The model receives:
- The rules of the alternating-offers protocol and the round limit,
- The objective statement: "Your goal is to maximize your payoff."
- its own payoff for every possible agreement, and the payoff for no agreement,
- each round: the offer on the table, the model's payoff for that offer, and the message (if any) the other party sent with it,
- the JSON format to answer in.

The other party's payoffs are kept private.
"""

from __future__ import annotations

from typing import Any

from negmas.sao import SAOState

from negmas_llm.negotiator import VLLMNegotiator

__all__ = [
    "RulesOnlyNegotiator",
    "RULES_SYSTEM_PROMPT",
    "RULES_PREFERENCES_PROMPT",
    "RULES_START_PROMPT",
    "RULES_ROUND_PROMPT",
]

RULES_SYSTEM_PROMPT = (
    "You are one of two parties in a negotiation. "
    "Respond strictly in the requested JSON format."
)

RULES_PREFERENCES_PROMPT = """\
These are your payoffs.

{{payoffs}}

Reply with OK."""

RULES_START_PROMPT = """\
The negotiation has started. These are the rules.

{{rules}}

On each of your turns, respond with JSON only, in this shape.
```json
{
    "response_type": "accept" | "reject" | "end",
    "outcome": [value1, value2, ...] | null,
    "text": "optional message shown to the other party",
    "reasoning": "optional, not shown to the other party"
}
```
"accept" accepts the offer on the table; "outcome" is ignored.
"reject" rejects it and proposes the outcome in "outcome" instead.
"end" ends the negotiation with no agreement.
If there is no offer on the table, use "reject" with an outcome to make one."""

RULES_ROUND_PROMPT = """\
Round {step}.

{offer_info}{{{{partner-message}}}}
Your decision (JSON):"""

class RulesOnlyNegotiator(VLLMNegotiator):
    """A :class:`VLLMNegotiator` prompted with the rules and game payoffs only.

    Every prompt can still be replaced through the usual constructor arguments (``system_prompt``, ``preferences_prompt``, ``negotiation_start_prompt``,
    ``round_prompt``); these templates are only the defaults.
    """

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", RULES_SYSTEM_PROMPT)
        kwargs.setdefault("preferences_prompt", RULES_PREFERENCES_PROMPT)
        kwargs.setdefault("negotiation_start_prompt", RULES_START_PROMPT)
        kwargs.setdefault("round_prompt", RULES_ROUND_PROMPT)
        # I'm okay with max_words being left at default.
        super().__init__(model=model, **kwargs)
    
    def process_prompt(self, prompt: str, state: SAOState | None = None) -> str:
        """Fill this class's own tags, then let negmas-llm process the rest."""
        if "{{payoffs}}" in prompt:
            prompt = prompt.replace("{{payoffs}}", self.describe_payoffs())
        if "{{rules}}" in prompt:
            prompt = prompt.replace("{{rules}}", self.describe_rules())
        if "{{partner-message}}" in prompt:
            prompt = prompt.replace(
                "{{partner-message}}", self.describe_partner_message(state)
            )
        return super().process_prompt(prompt, state=state)
    
    def describe_payoffs(self) -> str:
        """A formula for the payoff of any agreement, and the no-agreement payoff.

        Works when the utility function is linear additive and every issue's
        value function is linear (AffineFun, LinearFun, IdentityFun, ConstFun):
        payoff = sum_i w_i * (slope_i * x_i + bias_i), which collapses to
        constant + sum_i (w_i * slope_i) * x_i. Anything else falls back to
        str(ufun).
        """
        ufun = self.ufun
        if ufun is None:
            return "Your payoffs are not available."
        issues = list(ufun.outcome_space.issues) if ufun.outcome_space else []
        weights = getattr(ufun, "weights", None)
        values = getattr(ufun, "values", None)

        formula = None
        if issues and weights is not None and values is not None:
            constant, terms = 0.0, []
            for issue, w, vf in zip(issues, weights, values, strict=False):
                slope = getattr(vf, "slope", 1.0 if type(vf).__name__ == "IdentityFun" else 0.0)
                if type(vf).__name__ not in ("AffineFun", "LinearFun", "IdentityFun", "ConstFun"):
                    break  # not linear in this issue: no closed-form formula
                constant += w * getattr(vf, "bias", 0.0)
                coef = w * slope
                if coef:
                    terms.append((coef, issue.name))
            else:
                parts = [f"{constant:.4g}"] if constant or not terms else []
                for coef, name in terms:
                    sign = "-" if coef < 0 else "+"
                    mag = abs(coef)
                    term = name if mag == 1 else f"{mag:.4g} * {name}"
                    if parts:
                        parts.append(f"{sign} {term}")
                    else:
                        parts.append(term if coef > 0 else f"-{term}")
                formula = " ".join(parts)

        names = ", ".join(i.name for i in issues)
        lines = [f"An agreement sets {names}."]
        if formula is not None:
            lines.append(f"Your payoff for an agreement is: {formula}")
        else:
            lines.append(f"Your payoff function is {ufun}.")
        reserved = self.reserved_value
        if reserved is not None and math.isfinite(reserved):
            lines.append(f"If no agreement is reached, your payoff is {reserved:g}.")
        return "\n".join(lines)

    def describe_rules(self) -> str:
        """The alternating-offers protocol as configured for this negotiation."""
        nmi = self.nmi
        lines = [
            "The two parties take turns. On your turn you either accept the "
            "offer on the table, reject it and make a counter-offer, or end "
            "the negotiation.",
            "An agreement is reached when one party accepts the other's offer. "
            "Proposing exactly the offer already on the table also counts as "
            "accepting it.",
        ]
        n_steps = getattr(nmi, "n_steps", None) if nmi is not None else None
        if n_steps:
            lines.append(
                f"There are at most {n_steps} rounds; in each round both parties "
                "get one turn."
            )
        lines.append(
            "If either party ends the negotiation, or the rounds run out without "
            "an agreement, each party receives its no-agreement payoff."
        )
        return "\n".join(lines)

    def describe_partner_message(self, state: SAOState | None) -> str:
        """The message the other party sent with the offer on the table, if any."""
        if state is None or state.current_proposer in (None, self.id):
            return ""
        data = state.current_data or {}
        text = data.get("text") if isinstance(data, dict) else None
        return f'The other party\'s message: "{text}"\n' if text else ""