"""Orchestrate the empirical booking case study: tasks × models × trials.

Resumable — skips trajectories whose JSONL output already exists. Single
bash invocation so macOS / Claude-Code permission prompts that fire on
shell ``for`` loops don't bite.

Usage:

    python3 run_all.py \\
        --models claude-haiku-4-5-20251001,claude-sonnet-4-6,claude-opus-4-7 \\
        --trials 2 \\
        --out trajectories/v1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure local imports work when called from the example dir.
sys.path.insert(0, str(Path(__file__).parent))

from booking_agent import run_real_agent  # noqa: E402
from tasks import TASKS  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="claude-haiku-4-5-20251001",
        help="comma-separated model ids",
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("trajectories/v1"))
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args(argv[1:])

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    total = len(TASKS) * len(models) * args.trials
    print(f"matrix: {len(TASKS)} tasks × {len(models)} models × {args.trials} trials = {total} trajectories")

    started = time.time()
    done = 0
    skipped = 0
    failed = 0

    for task in TASKS:
        for model in models:
            for trial in range(args.trials):
                done += 1
                out_path = args.out / f"{task.id}__{model}__trial{trial}.jsonl"
                if out_path.exists():
                    skipped += 1
                    print(f"  [{done:>3}/{total}] SKIP {out_path.name}")
                    continue
                print(f"  [{done:>3}/{total}] RUN  {out_path.name}", flush=True)
                try:
                    outcome = run_real_agent(
                        system_prompt=task.system_prompt,
                        user_request=task.user_request,
                        user_replies=task.user_replies,
                        max_turns=args.max_turns,
                        model=model,
                        out_path=out_path,
                        quiet=True,
                        task_id=task.id,
                        trial=trial,
                    )
                    print(f"        outcome: {outcome}", flush=True)
                except Exception as exc:  # pragma: no cover
                    failed += 1
                    print(f"        FAILED: {exc!r}", flush=True)
                    if out_path.exists():
                        out_path.rename(out_path.with_suffix(".jsonl.failed"))

    elapsed = time.time() - started
    print(f"\ndone in {elapsed:.0f}s — generated: {done - skipped - failed}, skipped: {skipped}, failed: {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
