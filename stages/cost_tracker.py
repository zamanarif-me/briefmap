"""
Cost Tracker for BriefMap

Tracks token usage and calculates cost for every API call.
Supports Anthropic Claude and Gemini models.

Per-run tracking:
    Each pipeline run creates its own CostTracker and installs it with
    use_tracker(). Stage code calls current_tracker() — concurrent runs
    (multiple Streamlit users / threads) no longer intermix their costs,
    and reset() of one run cannot wipe another run's data.

Usage:
    from stages.cost_tracker import CostTracker, use_tracker, current_tracker

    run_tracker = CostTracker()
    use_tracker(run_tracker)          # at the start of the worker thread
    ...
    run_tracker.print_report()
    run_tracker.save_report("output/cost_report.json")
"""

import contextvars
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ── Pricing per 1M tokens (June 2026) ────────────────────────────────────────

PRICING = {
    # Anthropic
    "claude-sonnet-5": {
        "input":  3.00,   # $3.00 per 1M input tokens ($2.00 intro through 2026-08-31)
        "output": 15.00,  # $15.00 per 1M output tokens ($10.00 intro)
    },
    "claude-sonnet-4-6": {
        "input":  3.00,
        "output": 15.00,
    },
    "claude-haiku-4-5": {
        "input":  1.00,
        "output": 5.00,
    },
    # Gemini (1.5 family is retired — do not use)
    "gemini-2.5-flash": {
        "input":  0.15,
        "output": 0.60,
    },
    "gemini-2.5-pro": {
        "input":  1.25,
        "output": 10.00,
    },
    "gemini-2.0-flash": {
        "input":  0.10,   # Free tier available; paid is ~$0.10/1M
        "output": 0.40,
    },
    # Serper.dev
    "serper": {
        "per_call": 0.001,  # ~$1 per 1000 calls on paid plan; free tier = $0
    },
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class APICall:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_write_tokens: int = 0   # billed at 1.25x input price (5-min TTL)
    cache_read_tokens: int = 0    # billed at 0.1x input price
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


@dataclass
class SerperCall:
    stage: str
    queries: int
    cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── Cost Calculator ───────────────────────────────────────────────────────────

def calculate_llm_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """
    Calculate cost in USD for an LLM call.

    Prompt-cache tokens are billed against the input price:
    writes at 1.25x (5-minute TTL), reads at 0.1x.
    """
    pricing = PRICING.get(model)
    if not pricing:
        # Unknown model — estimate at Sonnet price
        pricing = PRICING["claude-sonnet-5"]

    input_cost  = (input_tokens  / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    cache_cost  = (
        (cache_write_tokens / 1_000_000) * pricing["input"] * 1.25
        + (cache_read_tokens / 1_000_000) * pricing["input"] * 0.10
    )
    return round(input_cost + output_cost + cache_cost, 6)


def calculate_serper_cost(num_calls: int, on_paid_plan: bool = False) -> float:
    """Serper.dev cost. Free tier = $0. Paid = ~$0.001/call."""
    if not on_paid_plan:
        return 0.0
    return round(num_calls * PRICING["serper"]["per_call"], 4)


# ── Global Tracker ────────────────────────────────────────────────────────────

class CostTracker:
    """
    Cost tracker. One instance per pipeline run (see use_tracker /
    current_tracker at the bottom of this module).
    """

    def __init__(self):
        self.llm_calls:    list[APICall]   = []
        self.serper_calls: list[SerperCall] = []
        self.run_start:    str = datetime.now().isoformat()
        self.serper_on_paid_plan: bool = False

    def reset(self):
        """Call at the start of each pipeline run."""
        self.llm_calls    = []
        self.serper_calls = []
        self.run_start    = datetime.now().isoformat()

    def log_llm_call(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        notes: str = "",
    ) -> float:
        """Log one LLM API call and return its cost."""
        cost = calculate_llm_cost(
            model, input_tokens, output_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
        )
        self.llm_calls.append(APICall(
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            notes=notes,
        ))
        return cost

    def log_serper_call(self, stage: str, num_queries: int = 1):
        """Log Serper.dev API calls."""
        cost = calculate_serper_cost(num_queries, self.serper_on_paid_plan)
        self.serper_calls.append(SerperCall(
            stage=stage,
            queries=num_queries,
            cost_usd=cost,
        ))

    # ── Aggregations ──────────────────────────────────────────────────────────

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def total_cache_write_tokens(self) -> int:
        return sum(c.cache_write_tokens for c in self.llm_calls)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(c.cache_read_tokens for c in self.llm_calls)

    @property
    def total_llm_cost(self) -> float:
        return round(sum(c.cost_usd for c in self.llm_calls), 4)

    @property
    def total_serper_calls(self) -> int:
        return sum(c.queries for c in self.serper_calls)

    @property
    def total_serper_cost(self) -> float:
        return round(sum(c.cost_usd for c in self.serper_calls), 4)

    @property
    def total_cost(self) -> float:
        return round(self.total_llm_cost + self.total_serper_cost, 4)

    def by_stage(self) -> dict[str, dict]:
        """Cost and token breakdown per stage."""
        stages: dict[str, dict] = {}
        for call in self.llm_calls:
            if call.stage not in stages:
                stages[call.stage] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "model": call.model,
                }
            stages[call.stage]["calls"]         += 1
            stages[call.stage]["input_tokens"]  += call.input_tokens
            stages[call.stage]["output_tokens"] += call.output_tokens
            stages[call.stage]["cost_usd"]      += call.cost_usd

        for entry in stages.values():
            entry["cost_usd"] = round(entry["cost_usd"], 4)

        return stages

    def by_model(self) -> dict[str, dict]:
        """Cost breakdown per model."""
        models: dict[str, dict] = {}
        for call in self.llm_calls:
            if call.model not in models:
                models[call.model] = {"calls": 0, "cost_usd": 0.0}
            models[call.model]["calls"]    += 1
            models[call.model]["cost_usd"] += call.cost_usd

        for entry in models.values():
            entry["cost_usd"] = round(entry["cost_usd"], 4)

        return models

    # ── Report ────────────────────────────────────────────────────────────────

    def print_report(self):
        """Print a human-readable cost report to console."""
        sep = "=" * 60

        print(f"\n{sep}")
        print("  BRIEFMAP — COST REPORT")
        print(f"  Run started: {self.run_start}")
        print(sep)

        # Per-stage breakdown
        print("\n── By Stage ──────────────────────────────────────────")
        print(f"{'Stage':<35} {'Calls':>5} {'In Tok':>8} {'Out Tok':>8} {'Cost':>8}")
        print("-" * 68)
        for stage, data in self.by_stage().items():
            print(
                f"{stage:<35} "
                f"{data['calls']:>5} "
                f"{data['input_tokens']:>8,} "
                f"{data['output_tokens']:>8,} "
                f"${data['cost_usd']:>7.4f}"
            )

        # Serper
        if self.serper_calls:
            print("-" * 68)
            serper_label = "Stage 3.5 — Serper.dev"
            plan = "paid" if self.serper_on_paid_plan else "free tier"
            print(
                f"{serper_label:<35} "
                f"{self.total_serper_calls:>5} "
                f"{'—':>8} "
                f"{'—':>8} "
                f"${self.total_serper_cost:>7.4f}  ({plan})"
            )

        # Totals
        print("=" * 68)
        print(
            f"{'TOTAL':<35} "
            f"{len(self.llm_calls):>5} "
            f"{self.total_input_tokens:>8,} "
            f"{self.total_output_tokens:>8,} "
            f"${self.total_cost:>7.4f}"
        )

        # Per-model summary
        print("\n── By Model ──────────────────────────────────────────")
        for model, data in self.by_model().items():
            print(f"  {model:<35} {data['calls']} calls   ${data['cost_usd']:.4f}")

        # Projection
        print("\n── Monthly Projection ────────────────────────────────")
        for n in [10, 50, 100, 200]:
            monthly = round(self.total_cost * n, 2)
            serper_note = ""
            if n > 50 and not self.serper_on_paid_plan:
                serper_note = " (add $50 Serper paid)"
            print(f"  {n:>4} maps/month → ${monthly:>8.2f}{serper_note}")

        print(f"\n{sep}\n")

    def save_report(self, path: str | Path):
        """Save full cost report as JSON."""
        report = {
            "run_start":           self.run_start,
            "summary": {
                "total_llm_calls":     len(self.llm_calls),
                "total_input_tokens":  self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cache_write_tokens": self.total_cache_write_tokens,
                "total_cache_read_tokens":  self.total_cache_read_tokens,
                "total_llm_cost_usd":  self.total_llm_cost,
                "total_serper_calls":  self.total_serper_calls,
                "total_serper_cost_usd": self.total_serper_cost,
                "total_cost_usd":      self.total_cost,
            },
            "by_stage":  self.by_stage(),
            "by_model":  self.by_model(),
            "all_calls": [
                {
                    "stage":         c.stage,
                    "model":         c.model,
                    "input_tokens":  c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cache_write_tokens": c.cache_write_tokens,
                    "cache_read_tokens":  c.cache_read_tokens,
                    "cost_usd":      c.cost_usd,
                    "timestamp":     c.timestamp,
                    "notes":         c.notes,
                }
                for c in self.llm_calls
            ],
            "serper_calls": [
                {
                    "stage":     c.stage,
                    "queries":   c.queries,
                    "cost_usd":  c.cost_usd,
                    "timestamp": c.timestamp,
                }
                for c in self.serper_calls
            ],
        }
        Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Cost report saved: {path}")


# ── Per-run tracker management ────────────────────────────────────────────────

# Global default tracker — used when no per-run tracker is installed
# (e.g. ad-hoc brief generation outside a pipeline run).
tracker = CostTracker()

_current_tracker: contextvars.ContextVar[CostTracker | None] = contextvars.ContextVar(
    "briefmap_cost_tracker", default=None
)


def use_tracker(t: CostTracker) -> None:
    """Install a tracker for the current thread/context (call at run start)."""
    _current_tracker.set(t)


def current_tracker() -> CostTracker:
    """Return the tracker for this context, falling back to the global one."""
    return _current_tracker.get() or tracker
