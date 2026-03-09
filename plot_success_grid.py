from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def _is_success(block: dict[str, Any], require_valid: bool) -> bool:
    if not isinstance(block, dict):
        return False
    solved = bool(block.get("solved", False))
    if not solved:
        return False
    if not require_valid:
        return True
    valid = block.get("valid", True)
    return bool(valid)


def _safe_float_rate(num: int, den: int) -> float:
    return (num / den) if den > 0 else float("nan")

def _iter_result_rows(path: Path):
    """
    Yield result objects from either:
    - JSONL: one JSON object per line (recommended; what run_scen_benchmark.py writes)
    - JSON: a list of objects, or {"rows": [...]}.
    """
    # Heuristic: if it starts with '[' or '{' we can try json.load; otherwise default to JSONL.
    try:
        with path.open("r", encoding="utf-8") as f:
            # Skip whitespace
            while True:
                ch = f.read(1)
                if not ch:
                    return
                if not ch.isspace():
                    break
            f.seek(0)
            if ch in ("[", "{"):
                data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            yield obj
                    return
                if isinstance(data, dict) and isinstance(data.get("rows"), list):
                    for obj in data["rows"]:
                        if isinstance(obj, dict):
                            yield obj
                    return
                # Fallback: treat as single object
                if isinstance(data, dict):
                    yield data
                    return
    except Exception:
        pass

    # JSONL fallback
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Results file(s). Pass multiple times to aggregate shards.",
    )
    ap.add_argument("--output", type=Path, default=Path("success_rate_grid.png"))
    ap.add_argument(
        "--require-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Count success only when (solved and valid).",
    )
    ap.add_argument(
        "--groups",
        type=str,
        default="random-32-32-10,den520d,brc202d,maze-128-128-1",
        help="Comma-separated scen group folder names in desired column order.",
    )
    ap.add_argument(
        "--k-min",
        type=int,
        default=1,
    )
    ap.add_argument(
        "--k-max",
        type=int,
        default=4,
    )
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Max agents on x-axis. If omitted, inferred from the input file.",
    )
    ap.add_argument(
        "--expected-per-cell",
        type=int,
        default=25,
        help="Expected number of instances per (group,k,n) cell (usually 25).",
    )
    ap.add_argument(
        "--denominator",
        choices=["seen", "expected", "auto"],
        default="auto",
        help=(
            "Success-rate denominator: 'seen' uses instances seen so far; "
            "'expected' uses --expected-per-cell; "
            "'auto' uses seen until expected is reached."
        ),
    )
    ap.add_argument(
        "--which",
        choices=["cbs", "star_first", "star_last", "cbs+star_last", "all"],
        default="cbs+star_last",
        help="Which success-rate curves to draw in each subplot.",
    )
    ap.add_argument(
        "--dodge-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Slightly offset x positions per series so overlapping lines are visible.",
    )
    ap.add_argument(
        "--marker-every",
        type=int,
        default=5,
        help="Place a marker every N x-ticks (0 disables markers).",
    )
    args = ap.parse_args()

    inputs = args.input or [Path("results.jsonl")]

    # Ensure Matplotlib/fontconfig caches are writable (important in sandboxed runs).
    repo_root = Path(__file__).resolve().parent
    cache_root = repo_root / ".cache"
    mpl_cache = cache_root / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    if len(groups) != 4:
        raise SystemExit("--groups must contain exactly 4 group names (4 columns).")

    # key: (group, k, n, series_name) -> [success, total]
    agg: dict[tuple[str, int, int, str], list[int]] = defaultdict(lambda: [0, 0])
    seen_n_max = 0
    seen_run_keys: set[str] = set()

    missing = [p for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f"Input not found: {missing[0]}")

    for p in inputs:
        for obj in _iter_result_rows(p):
            rk = obj.get("run_key")
            if isinstance(rk, str):
                if rk in seen_run_keys:
                    continue
                seen_run_keys.add(rk)
            spec = obj.get("spec") or {}
            group = spec.get("scen_group")
            k = spec.get("k")
            n = spec.get("n")
            if not isinstance(group, str) or not isinstance(k, int) or not isinstance(n, int):
                continue

            if group not in groups:
                continue
            if not (args.k_min <= k <= args.k_max and args.n_min <= n and (args.n_max is None or n <= args.n_max)):
                continue
            seen_n_max = max(seen_n_max, n)

            # CBS
            cbs = obj.get("k_robust_cbs") or {}
            cbs_ok = _is_success(cbs, require_valid=args.require_valid)
            a = agg[(group, k, n, "CBS")]
            a[1] += 1
            if cbs_ok:
                a[0] += 1

            # RobustLaCAMStar (first/last)
            star = obj.get("robust_lacam_star") or {}
            star_first = star.get("first") or {}
            star_last = star.get("last") or {}

            s1_ok = _is_success(star_first, require_valid=args.require_valid)
            a = agg[(group, k, n, "LaCAM*(first)")]
            a[1] += 1
            if s1_ok:
                a[0] += 1

            sL_ok = _is_success(star_last, require_valid=args.require_valid)
            a = agg[(group, k, n, "LaCAM*(last)")]
            a[1] += 1
            if sL_ok:
                a[0] += 1

    def series_to_plot() -> list[str]:
        if args.which == "cbs":
            return ["CBS"]
        if args.which == "star_first":
            return ["LaCAM*(first)"]
        if args.which == "star_last":
            return ["LaCAM*(last)"]
        if args.which == "cbs+star_last":
            return ["CBS", "LaCAM*(last)"]
        return ["CBS", "LaCAM*(first)", "LaCAM*(last)"]

    series = series_to_plot()
    colors = {
        "CBS": "tab:blue",
        "LaCAM*(first)": "tab:orange",
        "LaCAM*(last)": "tab:green",
    }
    linestyles = {
        "CBS": "-",
        "LaCAM*(first)": "--",
        "LaCAM*(last)": "-.",
    }
    markers = {
        "CBS": "o",
        "LaCAM*(first)": "s",
        "LaCAM*(last)": "^",
    }

    ks = list(range(args.k_min, args.k_max + 1))
    if len(ks) != 4:
        raise SystemExit("This script is designed for exactly 4 rows (k=1..4).")
    inferred_n_max = args.n_max if args.n_max is not None else seen_n_max
    if inferred_n_max <= 0:
        raise SystemExit("No matching rows found in input (cannot infer n range).")
    ns = list(range(args.n_min, inferred_n_max + 1))

    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True, sharey=True)

    pretty = {
        "random-32-32-10": "random",
        "den520d": "den520d",
        "brc202d": "brc202d",
        "maze-128-128-1": "maze-128-128-1",
    }

    for r, k in enumerate(ks):
        for c, group in enumerate(groups):
            ax = axes[r][c]
            for s in series:
                ys = []
                dens = []
                for n in ns:
                    succ, tot = agg.get((group, k, n, s), [0, 0])
                    if args.denominator == "expected":
                        den = args.expected_per_cell
                    elif args.denominator == "seen":
                        den = tot
                    else:
                        # auto: use seen until we have the full expected count
                        den = args.expected_per_cell if tot >= args.expected_per_cell else tot
                    ys.append(_safe_float_rate(succ, den))
                    dens.append(tot)
                if args.dodge_x and len(series) > 1:
                    idx = series.index(s)
                    center = (len(series) - 1) / 2.0
                    offset = (idx - center) * 0.12
                    xs = [x + offset for x in ns]
                else:
                    xs = ns
                markevery = None
                if args.marker_every and args.marker_every > 0:
                    markevery = max(1, int(args.marker_every))
                ax.plot(
                    xs,
                    ys,
                    label=s,
                    color=colors.get(s),
                    linestyle=linestyles.get(s, "-"),
                    marker=(markers.get(s) if markevery is not None else None),
                    markersize=3,
                    markevery=markevery,
                    linewidth=1.6,
                    alpha=0.9,
                )

            ax.set_ylim(0.0, 1.1)
            ax.grid(True, alpha=0.25)
            ax.set_xticks(ns)

            if r == 0:
                ax.set_title(pretty.get(group, group))
            if c == 0:
                ax.set_ylabel(f"k={k}\nsuccess rate")
            if r == 3:
                ax.set_xlabel("agents (n)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False)
    fig.suptitle("Success rate vs agents (25 scenarios per map folder)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    args.output.parent.mkdir(parents=True, exist_ok=True) if args.output.parent != Path(".") else None
    fig.savefig(args.output, dpi=200)
    print(f"Wrote: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

