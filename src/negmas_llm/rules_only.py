"""A negotiator that is told the rules of the game and its own payoffs, nothing more.

Unlike the default prompts of :class:`~negmas_llm.negotiator.LLMNegotiator`,
these contain no strategy advice ("open strong", "concede slowly", "never
accept below your reserved value", ...). What the model does get, all in one
briefing on its first turn:

- who it is (its party name) and, optionally, a plain-language description of
  the game supplied by the caller (e.g. "B receives 100 - share_for_A"),
- what an agreement consists of: each issue, its allowed values, and the exact
  JSON to write for an offer (keyed by issue name, so there is no question of
  which number belongs to which party),
- its own payoff formula, a few worked payoff examples, and its no-agreement
  payoff,
- the protocol: who moves first, how many turns there are in total, which
  turns are its own, what accepting / countering / ending do, and that the
  negotiation ends after the last turn,
- whether it knows the other party's payoffs (by default it does not).

Each turn it is told the turn number, the offer on the table written in the
same JSON form it answers in, who made that offer, what that offer would pay
it, and the other party's message.

Two mechanics of the base class are changed because they confused models:

1. The base class calls the LLM twice before the game starts (after the
   payoffs and after the rules) and throws the answers away. Models treated
   those calls as real turns, e.g. "proposing" an offer nobody ever saw. Here
   the setup is sent once, as part of the first real turn, so every model
   reply is an action that the mechanism actually executes.
2. An answer with a malformed outcome (wrong number of values, out of range)
   used to become "reject with no counter-offer", which ends the whole SAO
   negotiation immediately (``end_on_no_response``). Here the model is told
   what was wrong and asked again, up to ``max_format_retries`` times. The
   correction message states only the format error, never strategy.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
from typing import Any

from negmas.gb.common import ResponseType
from negmas.sao import SAOResponse, SAOState

from negmas_llm.negotiator import LLMNegotiator, VLLMNegotiator

__all__ = [
    "RulesOnlyNegotiator",
    "RULES_SYSTEM_PROMPT",
    "RULES_ROUND_PROMPT",
]

RULES_SYSTEM_PROMPT = (
    "You are a party in a negotiation. Each time it is your turn you will be "
    "told the state of the negotiation and must answer with a single JSON "
    "object in the requested format."
)

# {step} etc. are filled by LLMNegotiator.__call__ via str.format, so the
# {{tags}} of process_prompt are written with doubled braces.
RULES_ROUND_PROMPT = "{{{{briefing}}}}{{{{turn}}}}\n\nYour decision (JSON):"


class RulesOnlyNegotiator(VLLMNegotiator):
    """A :class:`VLLMNegotiator` prompted with the rules and its payoffs only.

    Args:
        game_description: Optional plain-language description of what is being
            negotiated, shown at the top of the briefing. Use it for facts the
            utility function cannot express, e.g. "A pie of 100 units is split
            between A and B. A receives share_for_A units and B receives
            100 - share_for_A units."
        reveal_opponent_payoffs: If True, the briefing also gives the other
            party's payoff formula and no-agreement payoff (complete
            information). Requires ``private_info["opponent_ufun"]`` to be
            set, as elsewhere in negmas-llm. Default False: the briefing says
            the other party's payoffs are private.
        max_format_retries: How many times to re-ask after a malformed answer
            before giving up (which ends the negotiation, as in the base class).
        **kwargs: Passed to :class:`VLLMNegotiator`. ``system_prompt`` and
            ``round_prompt`` may still be overridden.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        game_description: str | None = None,
        reveal_opponent_payoffs: bool = False,
        max_format_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("system_prompt", RULES_SYSTEM_PROMPT)
        kwargs.setdefault("round_prompt", RULES_ROUND_PROMPT)
        super().__init__(model=model, **kwargs)
        self.game_description = game_description
        self.reveal_opponent_payoffs = reveal_opponent_payoffs
        self.max_format_retries = max_format_retries
        self._briefed = False

    # ------------------------------------------------------------------
    # Structured output that matches this class's answer format
    # ------------------------------------------------------------------

    def response_schema(self) -> dict[str, Any]:
        """JSON schema for a turn's answer, built from the outcome space.

        The base class sends a fixed strict schema to OpenAI-compatible
        servers (vLLM included) that only allows "accept" / "reject" / "end"
        and a list-valued outcome. vLLM enforces it during decoding, so a
        model prompted with this class's format physically cannot write
        "offer" or {"share_for_A": 60}. This schema allows exactly the format
        the prompt describes.
        """
        props: dict[str, Any] = {}
        for i in self._issues():
            if getattr(i, "is_continuous", lambda: False)():
                props[i.name] = {"type": "number"}
            elif (
                getattr(i, "is_integer", lambda: False)()
                and getattr(i, "is_numeric", lambda: False)()
            ):
                props[i.name] = {"type": "integer"}
            else:
                props[i.name] = {"enum": list(i.all)}
        outcome = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": props,
                    "required": list(props),
                    "additionalProperties": False,
                },
                {"type": "null"},
            ]
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "negotiation_turn",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "response_type": {"enum": ["accept", "offer", "end"]},
                        "outcome": outcome,
                        "text": {"type": ["string", "null"]},
                        "reasoning": {"type": ["string", "null"]},
                    },
                    "required": ["response_type", "outcome", "text", "reasoning"],
                    "additionalProperties": False,
                },
            },
        }

    def _call_llm(self, messages, require_json: bool = False, *args, **kwargs):
        if not (require_json and self.use_structured_output):
            return super()._call_llm(messages, require_json, *args, **kwargs)
        # Send our schema instead of the base class's: switch the base
        # mechanism off for this call and pass response_format directly.
        saved_flag, saved_kwargs = self.use_structured_output, self.llm_kwargs
        self.use_structured_output = False
        self.llm_kwargs = {"response_format": self.response_schema(), **saved_kwargs}
        try:
            return super()._call_llm(messages, require_json, *args, **kwargs)
        finally:
            self.use_structured_output, self.llm_kwargs = saved_flag, saved_kwargs

    # ------------------------------------------------------------------
    # No LLM calls outside real turns
    # ------------------------------------------------------------------

    def on_preferences_changed(self, changes) -> None:
        # Skip LLMNegotiator's version (which calls the LLM); keep negmas's.
        super(LLMNegotiator, self).on_preferences_changed(changes)

    def on_negotiation_start(self, state) -> None:
        super(LLMNegotiator, self).on_negotiation_start(state)
        self._briefed = False

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def process_prompt(self, prompt: str, state: SAOState | None = None) -> str:
        if "{{briefing}}" in prompt:
            briefing = "" if self._briefed else self.describe_briefing() + "\n\n"
            self._briefed = True
            prompt = prompt.replace("{{briefing}}", briefing)
        if "{{turn}}" in prompt:
            prompt = prompt.replace("{{turn}}", self.describe_turn(state))
        return super().process_prompt(prompt, state=state)

    # ------------------------------------------------------------------
    # Briefing (sent once, on the first turn)
    # ------------------------------------------------------------------

    def describe_briefing(self) -> str:
        sections = [f"You are party {self.name}."]
        if self.game_description:
            sections.append(self.game_description.strip())
        sections += [
            "WHAT AN AGREEMENT IS\n" + self.describe_outcomes(),
            "YOUR PAYOFFS\n" + self.describe_payoffs(),
            "THE OTHER PARTY'S PAYOFFS\n" + self.describe_opponent_payoffs(),
            "RULES\n" + self.describe_rules(),
            "HOW TO ANSWER\n" + self.describe_answer_format(),
        ]
        return "\n\n".join(sections)

    def _issues(self) -> list:
        os_ = self.nmi.outcome_space if self.nmi is not None else None
        return list(getattr(os_, "issues", None) or [])

    @staticmethod
    def _describe_values(issue) -> str:
        if getattr(issue, "is_continuous", lambda: False)():
            return f"any number from {issue.min_value:g} to {issue.max_value:g}"
        if (
            getattr(issue, "is_integer", lambda: False)()
            and getattr(issue, "is_numeric", lambda: False)()
        ):
            return f"a whole number from {issue.min_value} to {issue.max_value}"
        vals = list(issue.all)
        return "one of " + ", ".join(json.dumps(v) for v in vals)

    def _outcome_json(self, outcome) -> str:
        names = [i.name for i in self._issues()]
        if len(names) != len(outcome):
            return json.dumps(list(outcome))
        return json.dumps(dict(zip(names, outcome, strict=True)))

    def describe_outcomes(self) -> str:
        issues = self._issues()
        lines = ["An agreement sets a value for each of these items:"]
        for i in issues:
            lines.append(f"- {i.name}: {self._describe_values(i)}")
        return "\n".join(lines)

    def _payoff_formula(self, ufun) -> str | None:
        """Closed form for linear-additive ufuns with linear value functions."""
        issues = self._issues()
        weights = getattr(ufun, "weights", None)
        values = getattr(ufun, "values", None)
        if not issues or weights is None or values is None:
            return None
        constant, terms = 0.0, []
        for issue, w, vf in zip(issues, weights, values, strict=False):
            kind = type(vf).__name__
            if kind not in ("AffineFun", "LinearFun", "IdentityFun", "ConstFun"):
                return None
            slope = 1.0 if kind == "IdentityFun" else getattr(vf, "slope", 0.0)
            constant += w * getattr(vf, "bias", 0.0)
            if w * slope:
                terms.append((w * slope, issue.name))
        parts = [f"{constant:.4g}"] if constant or not terms else []
        for coef, name in terms:
            mag = abs(coef)
            term = name if mag == 1 else f"{mag:.4g} * {name}"
            if parts:
                parts.append(f"{'-' if coef < 0 else '+'} {term}")
            else:
                parts.append(term if coef > 0 else f"-{term}")
        return " ".join(parts)

    def _payoff_examples(self, ufun, k: int = 5) -> list[str]:
        """Payoffs of a few evenly spaced outcomes (single-issue spaces only)."""
        issues = self._issues()
        if len(issues) != 1:
            return []
        try:
            vals = list(issues[0].all)
        except Exception:
            return []
        if len(vals) > k:
            vals = [vals[round(j * (len(vals) - 1) / (k - 1))] for j in range(k)]
        return [
            f"{self._outcome_json((v,))} pays {float(ufun((v,))):.4g}" for v in vals
        ]

    def _describe_ufun(self, ufun, reserved, whose: str) -> str:
        lines = []
        formula = self._payoff_formula(ufun)
        if formula is not None:
            lines.append(f"{whose} payoff for an agreement is {formula}.")
        else:
            lines.append(f"{whose} payoff function is {ufun}.")
        examples = self._payoff_examples(ufun)
        if examples:
            lines.append("Examples: " + "; ".join(examples) + ".")
        if reserved is not None and math.isfinite(reserved):
            lines.append(
                f"If the negotiation ends without an agreement, {whose.lower()} "
                f"payoff is {reserved:g}."
            )
        return "\n".join(lines)

    def describe_payoffs(self) -> str:
        if self.ufun is None:
            return "Your payoffs are not available."
        return self._describe_ufun(self.ufun, self.reserved_value, "Your")

    def describe_opponent_payoffs(self) -> str:
        opp = (self.private_info or {}).get("opponent_ufun")
        if self.reveal_opponent_payoffs and opp is not None:
            return self._describe_ufun(
                opp, getattr(opp, "reserved_value", None), "Their"
            )
        return "You are not told the other party's payoffs."

    def _turn_numbers(self) -> tuple[int | None, bool]:
        """(total turns, whether this negotiator moves first)."""
        nmi = self.nmi
        first = True
        with contextlib.suppress(Exception):
            first = nmi.negotiator_index(self.id) == 0  # type: ignore[union-attr]
        n_steps = getattr(nmi, "n_steps", None)
        n_neg = getattr(nmi, "n_negotiators", 2) or 2
        if n_steps is None or getattr(nmi, "one_offer_per_step", False):
            return (n_steps, first)
        return (n_steps * n_neg, first)

    def describe_rules(self) -> str:
        total, first = self._turn_numbers()
        nmi = self.nmi
        lines = [
            "The two parties take turns. "
            + ("You move first." if first else "The other party moves first."),
        ]
        if total:
            mine = "odd-numbered" if first else "even-numbered"
            lines.append(
                f"There are {total} turns in total, numbered 1 to {total}; "
                f"yours are the {mine} turns. After turn {total} the "
                "negotiation is over."
            )
        lines += [
            "On your turn you do exactly one of these:",
            "- accept the offer the other party made on their last turn, which "
            "ends the negotiation with that offer as the agreement;",
            "- make an offer of your own, which replaces any offer on the "
            "table and which the other party can accept on their next turn;",
            "- end the negotiation without an agreement.",
        ]
        if getattr(nmi, "offering_is_accepting", False):
            lines.append(
                "Making an offer identical to the one on the table counts as "
                "accepting it."
            )
        lines.append(
            "If the negotiation is ended, or the last turn passes without an "
            "agreement, each party gets its no-agreement payoff."
        )
        lines.append(
            "Each turn you may also send the other party a short message; they "
            "see it with your offer."
        )
        return "\n".join(lines)

    def describe_answer_format(self) -> str:
        issues = self._issues()
        template = "{" + ", ".join(f'"{i.name}": <value>' for i in issues) + "}"
        return (
            "Answer with one JSON object:\n"
            "```json\n"
            "{\n"
            '  "response_type": "accept" | "offer" | "end",\n'
            f'  "outcome": {template} or null,\n'
            '  "text": "message shown to the other party (optional)",\n'
            '  "reasoning": "private notes, not shown to anyone (optional)"\n'
            "}\n"
            "```\n"
            '- "accept": accept the offer on the table; set "outcome" to null. '
            "Only possible when there is an offer from the other party on the "
            "table.\n"
            '- "offer": make an offer; "outcome" gives the value of every item, '
            "keyed by item name.\n"
            '- "end": end the negotiation without an agreement; set "outcome" '
            "to null."
        )

    # ------------------------------------------------------------------
    # Per-turn state
    # ------------------------------------------------------------------

    def describe_turn(self, state: SAOState | None) -> str:
        if state is None:
            return ""
        total, first = self._turn_numbers()
        n_neg = getattr(self.nmi, "n_negotiators", 2) or 2
        turn = state.step * n_neg + (1 if first else 2)
        head = f"TURN {turn}" + (f" of {total}" if total else "") + " (your turn)."
        lines = [head]
        offer, proposer = state.current_offer, state.current_proposer
        if offer is None or proposer in (None, self.id):
            lines.append("There is no offer from the other party on the table.")
        else:
            lines.append(f"The other party offers {self._outcome_json(offer)}.")
            if self.ufun is not None:
                with contextlib.suppress(TypeError, ValueError):
                    lines.append(
                        f"If you accept it, your payoff is "
                        f"{float(self.ufun(offer)):.4g}."
                    )
            data = state.current_data or {}
            text = data.get("text") if isinstance(data, dict) else None
            if text:
                lines.append(f'Their message: "{text}"')
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parsing: accept "offer" as a synonym for "reject"; re-ask on bad format
    # ------------------------------------------------------------------

    def _parse_llm_response(self, response_text: str, state: SAOState):
        text = re.sub(
            r'("response_type"\s*:\s*)"offer"', r'\1"reject"', response_text, count=1
        )
        return super()._parse_llm_response(text, state)

    def _format_problem(self, response_type, outcome, state: SAOState) -> str | None:
        has_offer = (
            state.current_offer is not None and state.current_proposer != self.id
        )
        if response_type == ResponseType.REJECT_OFFER and outcome is None:
            names = ", ".join(i.name for i in self._issues())
            return (
                "Your answer did not contain a valid offer. An offer needs "
                f'"outcome" with a valid value for each item ({names}), keyed '
                "by item name."
            )
        if response_type == ResponseType.ACCEPT_OFFER and not has_offer:
            return (
                "There is no offer from the other party on the table, so you "
                'cannot accept. Use "offer" or "end".'
            )
        return None

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        response = super().__call__(state, dest)
        for _ in range(self.max_format_retries):
            problem = self._format_problem(response.response, response.outcome, state)
            if problem is None:
                break
            reply = self._send_to_llm(
                f"{problem}\nIt is still your turn. Your decision (JSON):",
                require_json=True,
            )
            rtype, outcome, text, _reasoning = self._parse_llm_response(reply, state)
            response = SAOResponse(
                response=rtype,
                outcome=outcome,
                data={"text": text} if text else None,
            )
        return response