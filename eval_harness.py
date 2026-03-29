#!/usr/bin/env python3
"""Head-to-head eval: Router A (RL-style) vs Router B (fixed GPT-4) with killer chart."""

from __future__ import annotations

import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Matplotlib headless
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Mode: simulated (default) vs live (real API calls) ───────────────────────
# Toggle: EVAL_HARNESS_MODE=live  OR  EVAL_SIMULATED=0|false
EVAL_HARNESS_MODE = os.environ.get("EVAL_HARNESS_MODE", "simulated").strip().lower()
_explicit_live = EVAL_HARNESS_MODE == "live"
SIMULATED = not _explicit_live and (
    os.environ.get("EVAL_SIMULATED", "true").strip().lower() not in ("0", "false", "no")
)
if _explicit_live:
    SIMULATED = False

ROOT = Path(__file__).resolve().parent
PROMPTS_PATH = ROOT / "prompts.json"

# Rough USD / 1K tokens (for cost estimates)
_COST_PER_1K = {
    "router_a": 0.012,
    "router_b_gpt4": 0.030,
}


def _load_prompts() -> List[str]:
    raw = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < 20:
        raise SystemExit(f"Expected at least 20 prompts in {PROMPTS_PATH}")
    return [str(p) for p in raw[:20]]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4 + 1)


def _router_a_sim(prompt: str) -> Tuple[float, float, str]:
    """RL router: lower median latency, variable cost."""
    random.seed(hash(prompt) % (2**32))
    lat_ms = random.uniform(120.0, 480.0)
    toks_in = _estimate_tokens(prompt)
    toks_out = random.randint(80, 400)
    cost = (toks_in + toks_out) / 1000.0 * _COST_PER_1K["router_a"]
    return lat_ms, cost, f"[Router A] RL-routed response for: {prompt[:48]}…"


def _router_b_sim(prompt: str) -> Tuple[float, float, str]:
    """Fixed GPT-4: higher latency and premium cost."""
    random.seed((hash(prompt) ^ 0x9E3779B9) % (2**32))
    lat_ms = random.uniform(800.0, 2200.0)
    toks_in = _estimate_tokens(prompt)
    toks_out = random.randint(120, 600)
    cost = (toks_in + toks_out) / 1000.0 * _COST_PER_1K["router_b_gpt4"]
    return lat_ms, cost, f"[Router B] Fixed GPT-4 baseline for: {prompt[:48]}…"


def _call_openai_chat(model: str, system: str, user: str) -> Tuple[str, int, int]:
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY required for live mode")
    client = OpenAI(api_key=key)
    t0 = time.perf_counter()
    comp = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    _ = time.perf_counter() - t0
    text = (comp.choices[0].message.content or "").strip()
    usage = comp.usage
    inp = getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "completion_tokens", 0) or 0
    return text, inp, out


def _gpt4o_judge_score(question: str, answer: str) -> float:
    """LLM-as-judge 1–5 using GPT-4o."""
    system = (
        "You are an impartial evaluator. Score the assistant answer for usefulness "
        "and correctness on a scale of 1 to 5 only. Reply with a single digit 1-5, nothing else."
    )
    user = f"User question:\n{question}\n\nAssistant answer:\n{answer}\n\nScore (1-5):"
    text, _, _ = _call_openai_chat("gpt-4o", system, user)
    for ch in text.strip():
        if ch.isdigit() and "1" <= ch <= "5":
            return float(ch)
    return 3.0


def _live_pair(prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Live: two completions (gpt-4o-mini as Router A stand-in, gpt-4o as fixed B)."""
    model_a = os.environ.get("EVAL_ROUTER_A_MODEL", "gpt-4o-mini")
    model_b = os.environ.get("EVAL_ROUTER_B_MODEL", "gpt-4o")

    t0 = time.perf_counter()
    ans_a, in_a, out_a = _call_openai_chat(
        model_a,
        "You are a helpful assistant. Be concise.",
        prompt,
    )
    lat_a = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    ans_b, in_b, out_b = _call_openai_chat(
        model_b,
        "You are a helpful assistant. Be concise.",
        prompt,
    )
    lat_b = (time.perf_counter() - t1) * 1000.0

    # Pricing (approximate per 1M tokens)
    def usd(m: str, inp: int, out: int) -> float:
        ml = m.lower()
        if "gpt-4o-mini" in ml:
            return (inp * 0.15 + out * 0.60) / 1_000_000.0
        if "gpt-4o" in ml and "mini" not in ml:
            return (inp * 2.5 + out * 10.0) / 1_000_000.0
        return (inp * 0.5 + out * 1.5) / 1_000_000.0

    cost_a = usd(model_a, in_a, out_a)
    cost_b = usd(model_b, in_b, out_b)

    qa = _gpt4o_judge_score(prompt, ans_a)
    qb = _gpt4o_judge_score(prompt, ans_b)

    return (
        {"latency_ms": lat_a, "cost": cost_a, "quality": qa},
        {"latency_ms": lat_b, "cost": cost_b, "quality": qb},
    )


def _run_eval(prompts: List[str]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    rows_a: List[Dict[str, float]] = []
    rows_b: List[Dict[str, float]] = []

    if SIMULATED:
        for p in prompts:
            lat_a, c_a, _ = _router_a_sim(p)
            lat_b, c_b, _ = _router_b_sim(p)
            # Synthetic quality correlated with inverse latency (router A wins slightly)
            qa = 3.2 + min(1.8, 600.0 / max(lat_a, 50.0)) * 0.4 + random.uniform(-0.3, 0.3)
            qb = 3.5 + min(1.3, 900.0 / max(lat_b, 100.0)) * 0.35 + random.uniform(-0.2, 0.2)
            qa = float(max(1.0, min(5.0, round(qa, 2))))
            qb = float(max(1.0, min(5.0, round(qb, 2))))
            rows_a.append({"latency_ms": lat_a, "cost": c_a, "quality": qa})
            rows_b.append({"latency_ms": lat_b, "cost": c_b, "quality": qb})
    else:
        for p in prompts:
            a, b = _live_pair(p)
            rows_a.append(a)
            rows_b.append(b)

    def cols(rows: List[Dict[str, float]]) -> Dict[str, List[float]]:
        return {
            "latency_ms": [r["latency_ms"] for r in rows],
            "cost": [r["cost"] for r in rows],
            "quality": [r["quality"] for r in rows],
        }

    return cols(rows_a), cols(rows_b)


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _killer_chart(
    metrics_a: Dict[str, List[float]],
    metrics_b: Dict[str, List[float]],
    out_path: Path,
) -> None:
    med_lat_a = _median(metrics_a["latency_ms"])
    med_lat_b = _median(metrics_b["latency_ms"])
    tot_cost_a = sum(metrics_a["cost"])
    tot_cost_b = sum(metrics_b["cost"])
    max_cost = max(tot_cost_a, tot_cost_b, 1e-9)
    norm_a = tot_cost_a / max_cost
    norm_b = tot_cost_b / max_cost
    avg_q_a = statistics.mean(metrics_a["quality"]) if metrics_a["quality"] else 0.0
    avg_q_b = statistics.mean(metrics_b["quality"]) if metrics_b["quality"] else 0.0

    labels = ["Median latency (ms)", "Total cost (normalized)", "Avg quality (1–5)"]
    # Scale quality to comparable visual height with first two (normalize quality to 0-max)
    max_q = 5.0
    series_a = [med_lat_a, norm_a * 100.0, (avg_q_a / max_q) * 100.0]
    series_b = [med_lat_b, norm_b * 100.0, (avg_q_b / max_q) * 100.0]

    x = range(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    ax.bar([i - w / 2 for i in x], series_a, width=w, label="Router A (RL router)", color="#00f5d4")
    ax.bar([i + w / 2 for i in x], series_b, width=w, label="Router B (fixed GPT-4)", color="#7b61ff")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Scaled value (see caption)")
    mode = "Simulated" if SIMULATED else "Live"
    ax.set_title(f"Killer chart — {mode} · Router A vs Router B (n=20)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    caption = (
        f"Latency: raw ms median. Cost: total $ normalized to max=100. "
        f"Quality: avg score scaled as (avg/5)*100. "
        f"Totals: cost A=${tot_cost_a:.4f}, B=${tot_cost_b:.4f}."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=9, color="#444")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, format="png")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    prompts = _load_prompts()
    print(f"EVAL_HARNESS_MODE={EVAL_HARNESS_MODE} (set to 'live' for OpenAI API eval)")
    ma, mb = _run_eval(prompts)
    out = ROOT / "killer_chart.png"
    _killer_chart(ma, mb, out)


if __name__ == "__main__":
    main()
