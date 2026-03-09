from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.k_robust_cbs_wrapper import KRobustCBS
from src.pycam.mapf_utils import (
    get_grid as get_grid_std,
    get_scenario as get_scenario_std,
    get_sum_of_loss as get_sum_of_loss_std,
    validate_mapf_solution as validate_mapf_solution_std,
)
from src.robust_pycam.mapf_utils import validate_robust_mapf_solution
from src.robust_pycam_star import LaCAM as RobustLaCAMStar
from src.robust_pycam_star.mapf_utils import (
    Config as RobustStarConfig,
    get_sum_of_loss as get_sum_of_loss_star,
    validate_robust_mapf_solution as validate_robust_mapf_solution_star,
)


@dataclass(frozen=True)
class RunSpec:
    scen_group: str
    map_file: str
    scen_file: str
    k: int
    n: int
    time_limit_ms: int
    seed: int


def _stable_int_seed(base_seed: int, key: str) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (base_seed + int(h[:8], 16)) % (2**31 - 1)


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _scen_sort_key(p: Path) -> tuple[int, str]:
    # Expect filenames like xxx-random-12.scen or xxx-even-3.scen
    stem = p.stem
    num = -1
    for part in reversed(stem.split("-")):
        if part.isdigit():
            num = int(part)
            break
    return (num, p.name)


def _compute_metrics_std(grid, starts, goals, solution, k: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "solved": bool(solution),
        "valid": False,
        "soc": None,
        "makespan": None,
        "error": None,
    }
    if not solution:
        return out
    try:
        validate_mapf_solution_std(grid, starts, goals, solution)
        validate_robust_mapf_solution(grid, starts, goals, solution, k)
        out["valid"] = True
        out["makespan"] = len(solution) - 1
        out["soc"] = get_sum_of_loss_std(solution)
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def _compute_metrics_star(grid, starts_star, goals_star, solution, k: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "solved": bool(solution),
        "valid": False,
        "soc": None,
        "makespan": None,
        "error": None,
    }
    if not solution:
        return out
    try:
        validate_robust_mapf_solution_star(grid, starts_star, goals_star, solution, k)
        out["valid"] = True
        out["makespan"] = len(solution) - 1
        out["soc"] = get_sum_of_loss_star(solution)
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def _read_existing_keys(jsonl_path: Path) -> set[str]:
    keys: set[str] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("run_key")
                if isinstance(k, str):
                    keys.add(k)
            except Exception:
                continue
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen-root", type=Path, default=Path("scen"))
    ap.add_argument("--output", type=Path, default=Path("results.jsonl"))
    ap.add_argument("--append", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--time-limit-ms", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=60)
    ap.add_argument("--num-scen", type=int, default=25, help="How many .scen files to take from scen-random/")
    ap.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument(
        "--print-every-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print full scenario parameters and per-algorithm results for every run.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    scen_root = (repo_root / args.scen_root).resolve() if not args.scen_root.is_absolute() else args.scen_root

    scen_groups = sorted([p for p in scen_root.iterdir() if p.is_dir()])
    if args.verbose:
        print(f"Found {len(scen_groups)} scenario groups under {scen_root}")

    map_by_group: dict[Path, Path] = {}
    scen_random_by_group: dict[Path, list[Path]] = {}
    for g in scen_groups:
        maps = sorted(g.glob("*.map"))
        if not maps:
            continue
        map_file = maps[0]
        scen_random_dir = g / "scen-random"
        if not scen_random_dir.exists():
            continue
        scen_files = sorted(scen_random_dir.glob("*.scen"), key=_scen_sort_key)
        if not scen_files:
            continue
        map_by_group[g] = map_file
        scen_random_by_group[g] = scen_files[: args.num_scen]

    selected_groups = sorted(map_by_group.keys())
    total_scen_files = sum(len(scen_random_by_group[g]) for g in selected_groups)
    total_runs = total_scen_files * (args.k_max - args.k_min + 1) * (args.n_max - args.n_min + 1)

    if args.verbose:
        print(f"Selected {len(selected_groups)} groups with maps and scen-random/")
        print(f"Total .scen files (limited): {total_scen_files}")
        print(f"Planned runs: {total_runs} (k={args.k_min}..{args.k_max}, n={args.n_min}..{args.n_max})")

    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True) if args.output.parent != Path(".") else None
    mode = "a" if args.append else "w"
    existing: set[str] = _read_existing_keys(args.output) if (args.resume and args.append) else set()
    if args.verbose and existing:
        print(f"Resume enabled: skipping {len(existing)} already-recorded runs in {args.output}")

    # Cache grids per map file.
    grid_cache: dict[str, Any] = {}

    planner_cbs = KRobustCBS()  # verifies executable exists

    with args.output.open(mode, encoding="utf-8") as out_f:
        written = 0
        skipped = 0
        started_at = time.time()

        for group_dir in selected_groups:
            map_file = map_by_group[group_dir]
            grid_key = str(map_file.resolve())
            if grid_key not in grid_cache:
                grid_cache[grid_key] = get_grid_std(map_file)
            grid = grid_cache[grid_key]

            scen_files = scen_random_by_group[group_dir]
            for scen_file in scen_files:
                for k in range(args.k_min, args.k_max + 1):
                    for n in range(args.n_min, args.n_max + 1):
                        row_started = time.perf_counter()
                        run_key = "|".join(
                            [
                                group_dir.name,
                                _safe_rel(map_file, repo_root),
                                _safe_rel(scen_file, repo_root),
                                f"k={k}",
                                f"n={n}",
                                f"tl={args.time_limit_ms}",
                            ]
                        )
                        if run_key in existing:
                            skipped += 1
                            continue

                        seed_i = _stable_int_seed(args.seed, run_key)
                        spec = RunSpec(
                            scen_group=group_dir.name,
                            map_file=_safe_rel(map_file, repo_root),
                            scen_file=_safe_rel(scen_file, repo_root),
                            k=k,
                            n=n,
                            time_limit_ms=args.time_limit_ms,
                            seed=seed_i,
                        )
                        if args.verbose and args.print_every_run:
                            print(
                                f"RUN  group={spec.scen_group} map={spec.map_file} scen={spec.scen_file} "
                                f"k={spec.k} n={spec.n} tl_ms={spec.time_limit_ms} seed={spec.seed}",
                                flush=True,
                            )

                        # Build instance (starts/goals) from MovingAI .scen
                        try:
                            starts_std, goals_std = get_scenario_std(scen_file, n)
                        except Exception as e:
                            obj = {
                                "run_key": run_key,
                                "spec": asdict(spec),
                                "error": f"ScenarioParseError: {type(e).__name__}: {e}",
                            }
                            out_f.write(json.dumps(obj) + "\n")
                            out_f.flush()
                            written += 1
                            existing.add(run_key)
                            if args.verbose and args.print_every_run:
                                print(f"ERR  scenario_parse {type(e).__name__}: {e}", flush=True)
                            continue

                        # KRobustCBS
                        if args.verbose and args.print_every_run:
                            print("  CBS  start", flush=True)
                        cbs_start = time.perf_counter()
                        cbs_solution = []
                        cbs_error = None
                        try:
                            cbs_solution = planner_cbs.solve(
                                grid=grid,
                                starts=starts_std,
                                goals=goals_std,
                                seed=seed_i,
                                time_limit_ms=args.time_limit_ms,
                                flg_star=True,
                                verbose=0,
                                k_robust=k,
                            )
                        except Exception as e:
                            cbs_error = f"{type(e).__name__}: {e}"
                            cbs_solution = []
                        cbs_elapsed_ms = (time.perf_counter() - cbs_start) * 1000.0
                        cbs_metrics = _compute_metrics_std(grid, starts_std, goals_std, cbs_solution, k=k)
                        cbs_metrics["runtime_ms"] = cbs_elapsed_ms
                        if cbs_error is not None and cbs_metrics.get("error") is None:
                            cbs_metrics["error"] = cbs_error
                        if args.verbose and args.print_every_run:
                            print(
                                f"  CBS  solved={bool(cbs_metrics.get('solved'))} valid={bool(cbs_metrics.get('valid'))} "
                                f"soc={cbs_metrics.get('soc')} makespan={cbs_metrics.get('makespan')} "
                                f"rt_ms={cbs_metrics.get('runtime_ms'):.2f}",
                                flush=True,
                            )

                        # RobustLaCAMStar (first + best/last)
                        star = RobustLaCAMStar()
                        starts_star = RobustStarConfig(list(starts_std))
                        goals_star = RobustStarConfig(list(goals_std))

                        if args.verbose and args.print_every_run:
                            print("  STAR start", flush=True)
                        star_start = time.perf_counter()
                        star_solution_last = []
                        star_error = None
                        try:
                            star_solution_last = star.solve(
                                grid=grid,
                                starts=starts_star,
                                goals=goals_star,
                                seed=seed_i,
                                time_limit_ms=args.time_limit_ms,
                                flg_star=True,
                                verbose=0,
                                k_robust=k,
                            )
                        except Exception as e:
                            star_error = f"{type(e).__name__}: {e}"
                            star_solution_last = []
                        star_elapsed_ms = (time.perf_counter() - star_start) * 1000.0

                        # First solution reconstruction (if any)
                        star_solution_first = []
                        try:
                            n_first = getattr(star, "_N_goal_first", None)
                            if n_first is not None:
                                star_solution_first = star.backtrack(n_first)
                        except Exception:
                            star_solution_first = []

                        star_first_time_ms = getattr(star, "first_solution_time_ms", None)
                        star_best_time_ms = getattr(star, "best_solution_time_ms", None)

                        star_first_metrics = _compute_metrics_star(
                            grid, starts_star, goals_star, star_solution_first, k=k
                        )
                        star_last_metrics = _compute_metrics_star(
                            grid, starts_star, goals_star, star_solution_last, k=k
                        )
                        if args.verbose and args.print_every_run:
                            print(
                                "  STAR first"
                                f" solved={bool(star_first_metrics.get('solved'))}"
                                f" valid={bool(star_first_metrics.get('valid'))}"
                                f" soc={star_first_metrics.get('soc')}"
                                f" makespan={star_first_metrics.get('makespan')}",
                                flush=True,
                            )
                            print(
                                "  STAR best "
                                f" solved={bool(star_last_metrics.get('solved'))}"
                                f" valid={bool(star_last_metrics.get('valid'))}"
                                f" soc={star_last_metrics.get('soc')}"
                                f" makespan={star_last_metrics.get('makespan')}"
                                f" best_time_ms={star_best_time_ms}",
                                flush=True,
                            )

                        star_payload = {
                            "runtime_ms": star_elapsed_ms,
                            "first_solution_time_ms": star_first_time_ms,
                            "best_solution_time_ms": star_best_time_ms,
                            # Convenience fields for analytics (runtime-to-solution).
                            "first_runtime_ms": star_first_time_ms,
                            "last_runtime_ms": star_best_time_ms,
                            "first": star_first_metrics,
                            "last": star_last_metrics,
                        }
                        if star_error is not None:
                            star_payload["error"] = star_error

                        obj = {
                            "run_key": run_key,
                            "spec": asdict(spec),
                            "k_robust_cbs": cbs_metrics,
                            "robust_lacam_star": star_payload,
                        }
                        out_f.write(json.dumps(obj) + "\n")
                        out_f.flush()
                        written += 1
                        existing.add(run_key)
                        if args.verbose and args.print_every_run:
                            elapsed_s = time.perf_counter() - row_started
                            print(
                                f"DONE elapsed_s={elapsed_s:.2f} written={written} skipped={skipped}",
                                flush=True,
                            )

    if args.verbose:
        elapsed = time.time() - started_at
        print(f"Done. Wrote {written} rows to {args.output} (skipped {skipped}), elapsed {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

