"""Aggregate booking case-study trajectories into a per-model/per-task table.

Reads JSONL files produced by ``run_all.py`` and reports outcome rates,
violation labels, and per-task breakdowns. Mirrors the sweep tools in
the playwright case-study repo.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_trajectory(path: Path) -> dict:
    meta: dict = {}
    outcome = "incomplete"
    monitor_events: list[dict] = []
    first_violation: dict | None = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "meta":
            meta = rec
        elif rec.get("type") == "outcome":
            outcome = rec.get("outcome", "incomplete")
        elif rec.get("type") == "monitor":
            monitor_events.append(rec)
            if rec.get("verdict") == "Violation" and first_violation is None:
                first_violation = rec
    return {
        "path": path,
        "meta": meta,
        "outcome": outcome,
        "monitor_events": monitor_events,
        "first_violation": first_violation,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: sweep.py <trajectories-dir>", file=sys.stderr)
        return 2

    directory = Path(argv[1])
    rows = [load_trajectory(p) for p in sorted(directory.glob("*.jsonl"))]
    if not rows:
        print(f"no .jsonl files under {directory}", file=sys.stderr)
        return 1

    # ── Per-trajectory grid ──
    headers = ["task", "model", "trial", "outcome", "first_violation"]
    widths = [22, 32, 6, 15, 32]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for r in rows:
        m = r["meta"]
        fv = r["first_violation"]
        fv_str = f"{fv['kind']}!{fv['label']}" if fv else ""
        print("".join(c.ljust(w) for c, w in zip([
            m.get("task_id", "?"),
            m.get("model", "?"),
            str(m.get("trial", "?")),
            r["outcome"],
            fv_str,
        ], widths)))

    # ── Per-model summary ──
    by_model: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_model[r["meta"].get("model", "?")][r["outcome"]] += 1

    print()
    print("per-model outcome distribution:")
    print(f"{'model':<32}{'n':>4}{'ok':>7}{'violated':>10}{'unrecognized':>14}{'incomplete':>12}")
    print("-" * 79)
    for model, counts in sorted(by_model.items()):
        n = sum(counts.values())
        print(
            f"{model:<32}{n:>4}"
            f"{counts.get('ok', 0):>7}"
            f"{counts.get('violated', 0):>10}"
            f"{counts.get('unrecognized', 0):>14}"
            f"{counts.get('incomplete', 0):>12}"
        )

    # ── Per-task summary (collapse across models, group by category) ──
    print()
    print("per-task outcome distribution (collapsed across models):")
    by_task: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_task[r["meta"].get("task_id", "?")][r["outcome"]] += 1
    print(f"{'task':<22}{'n':>4}{'ok':>7}{'violated':>10}{'unrecognized':>14}{'incomplete':>12}")
    print("-" * 69)
    for task, counts in sorted(by_task.items()):
        n = sum(counts.values())
        print(
            f"{task:<22}{n:>4}"
            f"{counts.get('ok', 0):>7}"
            f"{counts.get('violated', 0):>10}"
            f"{counts.get('unrecognized', 0):>14}"
            f"{counts.get('incomplete', 0):>12}"
        )

    # ── Aggregate ──
    total = len(rows)
    counts = Counter(r["outcome"] for r in rows)
    print()
    print(f"TOTAL: {total} trajectories")
    for outcome, n in counts.most_common():
        pct = 100 * n / total
        print(f"  {outcome:<14}{n:>4}  ({pct:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
