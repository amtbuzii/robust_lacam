import argparse
import subprocess
import time
from pathlib import Path

from src.pycam import LaCAM
from src.pycam.mapf_utils import get_sum_of_loss as get_sum_of_loss_std
from src.pycam.mapf_utils import (
    get_grid as get_grid_std,
    get_scenario as get_scenario_std,
    validate_mapf_solution as validate_mapf_solution_std,
)

from src.k_robust_cbs_wrapper import KRobustCBS

from src.robust_pycam import LaCAM as RobustLaCAM
from src.robust_pycam.mapf_utils import (
    get_grid as get_grid_robust,
    get_scenario as get_scenario_robust,
    validate_mapf_solution as validate_mapf_solution_robust,
    validate_robust_mapf_solution,
    get_sum_of_loss as get_sum_of_loss_robust,
)

# from src.alt_robust_pycam import LaCAM as AltRobustLaCAM
# from src.alt_robust_pycam.mapf_utils import (
#     get_grid as get_grid_alt,
#     get_scenario as get_scenario_alt,
#     validate_robust_mapf_solution as validate_robust_mapf_solution_alt,
#     get_sum_of_loss as get_sum_of_loss_alt,
# )

from src.robust_pycam_star import LaCAM as RobustLaCAMStar
from src.robust_pycam_star.mapf_utils import (
    get_grid as get_grid_alt,
    get_scenario as get_scenario_alt,
    validate_robust_mapf_solution as validate_robust_mapf_solution_alt,
    get_sum_of_loss as get_sum_of_loss_alt,
)

from plot import (
    plot_solution, 
    plot_solutions_comparison, 
    plot_three_solutions_comparison,
    animate_solution,
)


def export_search_tree_and_render(planner, dot_basename: str, label: str) -> None:
    """Save search tree to .dot and render to .png (if dot is available)."""
    dot_path = Path(f"{dot_basename}.dot")
    png_path = Path(f"{dot_basename}.png")
    try:
        planner.export_search_tree_dot(str(dot_path))
        print(f"Search tree ({label}) written to: {dot_path.resolve()}")
        try:
            subprocess.run(
                ["dot", "-Tpng", "-o", str(png_path), str(dot_path)],
                check=True,
                capture_output=True,
            )
            print(f"  Rendered to: {png_path.resolve()}")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(f"  Render with: dot -Tpng -o {png_path.name} {dot_path.name}")
    except RuntimeError as e:
        print(f"  (Search tree not exported: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--map-file",
        type=Path,
        default=Path(__file__).parent / "assets" / "empty-4-4.map",
    )
    parser.add_argument(
        "-i",
        "--scen-file",
        type=Path,
        default=Path(__file__).parent / "assets"/ "empty-4-4.scen",
    )
    parser.add_argument(
        "-N",
        "--num-agents",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="output.txt",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=4,
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-t", "--time_limit_ms", type=int, default=500)
    parser.add_argument(
        "-k",
        "--k-robust",
        type=int,
        default=2,
        help="k-robust parameter for robust LaCAM versions",
    )
    parser.add_argument(
        "--flg_star",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="choose LaCAM* (default) or vanilla LaCAM",
    )
    parser.add_argument(
        "--export-search-tree",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="export search tree .dot and .png for Robust and Robust LaCAM*",
    )

    args = parser.parse_args()

    # define problem instance - use standard version for all (they're compatible)
    grid = get_grid_std(args.map_file)
    starts_std, goals_std = get_scenario_std(args.scen_file, args.num_agents)
    # Also get from robust versions to ensure type compatibility
    starts_robust, goals_robust = get_scenario_robust(args.scen_file, args.num_agents)
    starts_alt, goals_alt = get_scenario_alt(args.scen_file, args.num_agents)

    print("=" * 80)
    print("LaCAM Comparison: k-Robust CBS vs Robust LaCAM vs Robust LaCAM*")
    print("=" * 80)
    print(f"Map: {args.map_file}")
    print(f"Scenario: {args.scen_file}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Time limit: {args.time_limit_ms}ms")
    print(f"Seed: {args.seed}")
    print(f"k-robust: {args.k_robust}")
    print(f"LaCAM*: {args.flg_star}")
    print(f"Export search tree: {args.export_search_tree}")
    print("-" * 80)

    planner_k_robust_cbs = KRobustCBS()
    planner_robust = RobustLaCAM()
    # planner_alt_robust = AltRobustLaCAM()
    planner_robust_star = RobustLaCAMStar()
    
    # Run k-Robust CBS (optimal solution)
    print("\n" + "=" * 80)
    print("Running k-Robust CBS (Optimal)...")
    print("=" * 80)

    start_time_k_robust_cbs = time.time()
    solution_k_robust_cbs = planner_k_robust_cbs.solve(
        grid=grid,
        starts=starts_std,
        goals=goals_std,
        seed=args.seed,
        time_limit_ms=args.time_limit_ms,
        flg_star=args.flg_star,
        verbose=args.verbose,
        k_robust=args.k_robust,
    )
    elapsed_time_k_robust_cbs = (time.time() - start_time_k_robust_cbs) * 1000  # Convert to ms
    
    # Validate solutions (same k-robust check as LaCAM) and calculate metrics
    valid_k_robust_cbs = True
    cost_k_robust_cbs = None
    soc_k_robust_cbs = None
    try:
        validate_mapf_solution_std(grid, starts_std, goals_std, solution_k_robust_cbs)
        validate_robust_mapf_solution(
            grid, starts_std, goals_std, solution_k_robust_cbs, args.k_robust
        )
        cost_k_robust_cbs = len(solution_k_robust_cbs) - 1 if solution_k_robust_cbs else None
        soc_k_robust_cbs = get_sum_of_loss_std(solution_k_robust_cbs) if solution_k_robust_cbs else None
    except Exception as e:
        valid_k_robust_cbs = False
        cost_k_robust_cbs = None
        soc_k_robust_cbs = None
        print(f"  ⚠️  k-Robust CBS solution validation failed: {e}")
    
    if valid_k_robust_cbs:
        print(f"Solution cost (makespan): {cost_k_robust_cbs}")
        print(f"Sum of Costs (SOC): {soc_k_robust_cbs}")
        print(f"Runtime: {elapsed_time_k_robust_cbs:.2f}ms")

    #_ = planner_robust.solve(grid, starts_alt, goals_alt, seed=args.seed, time_limit_ms=1, flg_star=False, verbose=0, k_robust=0)

    # Run Robust LaCAM once
    print("\n" + "=" * 80)
    print("Running Robust LaCAM...")
    print("=" * 80)
    start_time_robust = time.time()
    solution_robust = planner_robust.solve(
        grid=grid,
        starts=starts_robust,
        goals=goals_robust,
        seed=args.seed,
        time_limit_ms=args.time_limit_ms,
        flg_star=args.flg_star,
        verbose=args.verbose,
        k_robust=args.k_robust,  # k-robust parameter
    )
    elapsed_time_robust = (time.time() - start_time_robust) * 1000  # Convert to ms
    solution_time_robust = planner_robust.best_solution_time_ms  # Time when best solution was found

    # Export Robust LaCAM search tree and render PNG (if requested)
    if args.export_search_tree:
        export_search_tree_and_render(
            planner_robust,
            "search_tree_robust_lacam",
            "Robust LaCAM",
        )

    # Validate solutions (including k-robustness) and calculate metrics
    valid_robust = True
    cost_robust = None
    soc_robust = None
    try:
        validate_robust_mapf_solution(
            grid, starts_robust, goals_robust, solution_robust, args.k_robust
        )
        cost_robust = len(solution_robust) - 1 if solution_robust else None
        soc_robust = get_sum_of_loss_robust(solution_robust) if solution_robust else None
    except Exception as e:
        valid_robust = False
        cost_robust = None
        soc_robust = None
        print(f"  ⚠️  Robust LaCAM solution validation failed: {e}")
    
    if valid_robust:
        print(f"Solution cost (makespan): {cost_robust}")
        print(f"Sum of Costs (SOC): {soc_robust}")
        print(f"Runtime: {elapsed_time_robust:.2f}ms")
    
    # Run Robust LaCAM* once (replaces Alternative Robust LaCAM)
    print("\n" + "=" * 80)
    print("Running Robust LaCAM*...")
    print("=" * 80)
    start_time_robust_star = time.time()
    solution_robust_star = planner_robust_star.solve(
        grid=grid,
        starts=starts_alt,
        goals=goals_alt,
        seed=args.seed,
        time_limit_ms=args.time_limit_ms,
        flg_star=args.flg_star,
        verbose=args.verbose,
        k_robust=args.k_robust,  # k-robust parameter
    )
    elapsed_time_robust_star = (time.time() - start_time_robust_star) * 1000  # Convert to ms
    solution_time_robust_star = planner_robust_star.best_solution_time_ms  # Time when best solution was found

    # Export Robust LaCAM* search tree and render PNG (if requested)
    if args.export_search_tree:
        export_search_tree_and_render(
            planner_robust_star,
            "search_tree_robust_lacam_star",
            "Robust LaCAM*",
        )

    # Validate solutions (including k-robustness) and calculate metrics
    valid_robust_star = True
    cost_robust_star = None
    soc_robust_star = None
    if not solution_robust_star:
        valid_robust_star = False
        print(f"  ⚠️  Robust LaCAM* found no solution (time limit or search exhausted)")
    else:
        try:
            validate_robust_mapf_solution_alt(
                grid, starts_alt, goals_alt, solution_robust_star, args.k_robust
            )
            cost_robust_star = len(solution_robust_star) - 1
            soc_robust_star = get_sum_of_loss_alt(solution_robust_star)
        except Exception as e:
            valid_robust_star = False
            cost_robust_star = None
            soc_robust_star = None
            print(f"  ⚠️  Robust LaCAM* solution validation failed: {e}")
    
    if valid_robust_star:
        print(f"Solution cost (makespan): {cost_robust_star}")
        print(f"Sum of Costs (SOC): {soc_robust_star}")
        print(f"Runtime: {elapsed_time_robust_star:.2f}ms")
    
   
    # Plot all three solutions side by side for comparison
    print("\nPlotting three solutions side by side for comparison...")
    plot_three_solutions_comparison(
        grid,
        solution_k_robust_cbs,
        solution_robust,
        solution_robust_star,
        title1=f"k-Robust CBS (k={args.k_robust})",
        title2=f"Robust LaCAM (k={args.k_robust})",
        title3=f"Robust LaCAM* (k={args.k_robust})",
        main_title="Solutions Comparison",
        soc1=soc_k_robust_cbs,
        soc2=soc_robust,
        soc3=soc_robust_star,
        runtime1=elapsed_time_k_robust_cbs,
        runtime2=elapsed_time_robust,
        runtime3=elapsed_time_robust_star,
        solution_time1=None,  # k-robust CBS doesn't track this yet
        solution_time2=solution_time_robust,
        solution_time3=solution_time_robust_star,
    )
    
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print(f"{'Metric':<20} {'k-Robust CBS':<20} {'Robust LaCAM':<20} {'Robust LaCAM*':<20}")
    print("-" * 80)
    
    # Makespan
    print(f"{'Makespan':<20} {str(cost_k_robust_cbs):<20} {str(cost_robust):<20} {str(cost_robust_star):<20}")
    
    # SOC
    print(f"{'SOC':<20} {str(soc_k_robust_cbs):<20} {str(soc_robust):<20} {str(soc_robust_star):<20}")
    
    # Runtime
    runtime_k_robust_cbs_str = f"{elapsed_time_k_robust_cbs:.2f}ms" if elapsed_time_k_robust_cbs is not None else "N/A"
    runtime_robust_str = f"{elapsed_time_robust:.2f}ms" if elapsed_time_robust is not None else "N/A"
    runtime_robust_star_str = f"{elapsed_time_robust_star:.2f}ms" if elapsed_time_robust_star is not None else "N/A"
    print(f"{'Runtime':<20} {runtime_k_robust_cbs_str:<20} {runtime_robust_str:<20} {runtime_robust_star_str:<20}")
    
    print("-" * 80)
    if cost_k_robust_cbs is not None and cost_robust is not None:
        diff_robust = cost_robust - cost_k_robust_cbs
        print(f"Robust LaCAM vs k-Robust CBS makespan difference: {diff_robust:+d}")
    if cost_k_robust_cbs is not None and cost_robust_star is not None:
        diff_star = cost_robust_star - cost_k_robust_cbs
        print(f"Robust LaCAM* vs k-Robust CBS makespan difference: {diff_star:+d}")
    if cost_robust is not None and cost_robust_star is not None:
        diff_robust_star = cost_robust_star - cost_robust
        print(f"Robust LaCAM* vs Robust makespan difference: {diff_robust_star:+d}")
    print("=" * 80)

 