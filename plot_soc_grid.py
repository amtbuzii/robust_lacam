from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _iter_result_rows(path: Path) -> Iterable[dict[str, Any]]:
    """
    Yield result objects from either:
    - JSONL: one JSON object per line (recommended; what run_scen_benchmark.py writes)
    - JSON: a list of objects, or {"rows": [...]}.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            # Peek first non-whitespace
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
                if isinstance(data, dict):
                    yield data
                    return
    except Exception:
        pass

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


def _is_success(block: dict[str, Any], require_valid: bool) -> bool:
    if not isinstance(block, dict):
        return False
    if not bool(block.get("solved", False)):
        return False
    if not require_valid:
        return True
    valid = block.get("valid", True)
    return bool(valid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Results file(s). Pass multiple times to aggregate shards.",
    )
    ap.add_argument("--output", type=Path, default=Path("soc_grid.png"))
    ap.add_argument(
        "--require-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average SOC only over (solved and valid).",
    )
    ap.add_argument(
        "--groups",
        type=str,
        default="random-32-32-10,den520d,brc202d,maze-128-128-1",
        help="Comma-separated scen group folder names in desired column order.",
    )
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=4)
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
        "--show-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show how many instances contributed to each average.",
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
    missing = [p for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f"Input not found: {missing[0]}")

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    if len(groups) != 4:
        raise SystemExit("--groups must contain exactly 4 group names (4 columns).")

    ks = list(range(args.k_min, args.k_max + 1))
    if len(ks) != 4:
        raise SystemExit("This script is designed for exactly 4 rows (k=1..4).")
    # Determine n-range from file (for live/partial runs).
    seen_n_max = 0

    # Ensure Matplotlib caches are writable (important in sandboxed runs).
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

    # (group,k,n,series) -> [sum_soc, count]
    agg: dict[tuple[str, int, int, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    seen_run_keys: set[str] = set()

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

            # Only include this instance if ALL 3 series have SOC (and are solved/valid).
            cbs = obj.get("k_robust_cbs") or {}
            star = obj.get("robust_lacam_star") or {}
            first = star.get("first") or {}
            best = star.get("last") or {}

            if not (_is_success(cbs, require_valid=args.require_valid)):
                continue
            if not (_is_success(first, require_valid=args.require_valid)):
                continue
            if not (_is_success(best, require_valid=args.require_valid)):
                continue

            cbs_soc = cbs.get("soc", None)
            first_soc = first.get("soc", None)
            best_soc = best.get("soc", None)
            if not isinstance(cbs_soc, (int, float)):
                continue
            if not isinstance(first_soc, (int, float)):
                continue
            if not isinstance(best_soc, (int, float)):
                continue

            for series_name, soc in (
                ("CBS", cbs_soc),
                ("LaCAM*(first)", first_soc),
                ("LaCAM*(best)", best_soc),
            ):
                a = agg[(group, k, n, series_name)]
                a[0] += float(soc)
                a[1] += 1.0

    series = ["CBS", "LaCAM*(first)", "LaCAM*(best)"]
    colors = {"CBS": "tab:blue", "LaCAM*(first)": "tab:orange", "LaCAM*(best)": "tab:green"}
    linestyles = {"CBS": "-", "LaCAM*(first)": "--", "LaCAM*(best)": "-."}
    markers = {"CBS": "o", "LaCAM*(first)": "s", "LaCAM*(best)": "^"}

    inferred_n_max = args.n_max if args.n_max is not None else seen_n_max
    if inferred_n_max <= 0:
        raise SystemExit("No matching rows found in input (cannot infer n range).")
    ns = list(range(args.n_min, inferred_n_max + 1))

    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True)

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
                cs = []
                for n in ns:
                    sum_soc, cnt = agg.get((group, k, n, s), [0.0, 0.0])
                    if cnt > 0:
                        ys.append(sum_soc / cnt)
                    else:
                        ys.append(float("nan"))
                    cs.append(int(cnt))
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

                if args.show_coverage:
                    # Light markers where we have any data (helps in live/partial runs).
                    mx = [n for n, cnt in zip(ns, cs) if cnt > 0]
                    my = [y for y, cnt in zip(ys, cs) if cnt > 0]
                    ax.scatter(mx, my, s=10, color=colors.get(s), alpha=0.35)

            ax.grid(True, alpha=0.25)
            ax.set_xticks(ns)

            if r == 0:
                ax.set_title(pretty.get(group, group))
            if c == 0:
                ax.set_ylabel(f"k={k}\navg SOC")
            if r == 3:
                ax.set_xlabel("agents (n)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False)
    fig.suptitle("Average SOC vs agents (computed from results file; live as file grows)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    args.output.parent.mkdir(parents=True, exist_ok=True) if args.output.parent != Path(".") else None
    fig.savefig(args.output, dpi=200)
    print(f"Wrote: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

