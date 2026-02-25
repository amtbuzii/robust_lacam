"""Experiment script for empty-16-16 map with varying k-robustness and agent counts."""

import argparse
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.alt_robust_pycam import LaCAM as AltRobustLaCAM
from src.alt_robust_pycam.mapf_utils import (
    get_grid,
    get_scenario,
    validate_robust_mapf_solution,
    get_sum_of_loss,
)


def run_experiments(
    map_file: Path,
    scen_dir: Path,
    num_agents_list: list[int],
    k_values: list[int],
    time_limit_ms: int = 1000,
    verbose: int = 0,
    flg_star: bool = False,
    log_file: Path | None = None,
) -> dict:
    """
    Run experiments with different k values and agent counts.
    
    Returns a dictionary with results organized by (num_agents, k_robust):
    {
        (5, 0): {'success': 45, 'costs': [...], 'times': [...]},
        (5, 1): {'success': 43, 'costs': [...], 'times': [...]},
        ...
    }
    """
    # Get all scenario files
    scen_files = sorted(scen_dir.glob("*.scen"))
    print(f"Found {len(scen_files)} scenario files")
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"Found {len(scen_files)} scenario files\n")
    
    if len(scen_files) == 0:
        raise ValueError(f"No scenario files found in {scen_dir}")
    
    # Load grid once
    grid = get_grid(map_file)
    print(f"Grid loaded: {grid.shape[0]}x{grid.shape[1]}")
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"Grid loaded: {grid.shape[0]}x{grid.shape[1]}\n\n")
    
    # Initialize planner (reuse for efficiency)
    planner = AltRobustLaCAM()
    
    results = defaultdict(lambda: {'success_count': 0, 'costs': [], 'times': []})
    
    total_runs = len(num_agents_list) * len(k_values) * len(scen_files)
    run_count = 0
    
    for num_agents in num_agents_list:
        print(f"\n{'='*80}")
        print(f"Testing with {num_agents} agents")
        print(f"{'='*80}")
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Testing with {num_agents} agents\n")
                f.write(f"{'='*80}\n\n")
        
        for k_robust in k_values:
            print(f"\n  Testing k={k_robust}...")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"Testing k={k_robust}...\n")
            success_count = 0
            costs = []
            socs = []
            times = []
            
            for scen_file in scen_files:
                run_count += 1
                if run_count % 10 == 0:
                    progress_msg = f"    Progress: {run_count}/{total_runs} runs ({run_count*100//total_runs}%)"
                    print(progress_msg)
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write(progress_msg + "\n")
                
                try:
                    # Load scenario
                    starts, goals = get_scenario(scen_file, num_agents)
                    
                    # Run planner
                    start_time = time.time()
                    solution = planner.solve(
                        grid=grid,
                        starts=starts,
                        goals=goals,
                        seed=0,
                        time_limit_ms=time_limit_ms,
                        flg_star=flg_star,
                        verbose=verbose,
                        k_robust=k_robust,
                    )
                    elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Validate solution
                    if solution and len(solution) > 0:
                        try:
                            validate_robust_mapf_solution(
                                grid, starts, goals, solution, k_robust
                            )
                            # Valid solution
                            cost = len(solution) - 1  # Makespan
                            soc = get_sum_of_loss(solution)  # Sum of Costs
                            success_count += 1
                            costs.append(cost)
                            socs.append(soc)
                            times.append(elapsed_time)
                            
                            # Log success
                            if log_file:
                                with open(log_file, 'a') as f:
                                    f.write(f"      {scen_file.name}: SUCCESS (makespan={cost}, SOC={soc}, time={elapsed_time:.2f}ms)\n")
                        except Exception as e:
                            # Invalid solution
                            if log_file:
                                with open(log_file, 'a') as f:
                                    f.write(f"      {scen_file.name}: FAILED validation - {e}\n")
                    else:
                        # No solution found
                        if log_file:
                            with open(log_file, 'a') as f:
                                f.write(f"      {scen_file.name}: NO SOLUTION\n")
                        
                except Exception as e:
                    if verbose > 0:
                        print(f"      Error with {scen_file.name}: {e}")
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write(f"      {scen_file.name}: ERROR - {e}\n")
                    pass
            
            # Store results
            results[(num_agents, k_robust)] = {
                'success_count': success_count,
                'costs': costs,  # Makespan
                'socs': socs,    # Sum of Costs
                'times': times,
            }
            
            result_msg = f"    k={k_robust}: {success_count}/{len(scen_files)} successful"
            if costs:
                result_msg += f" (avg makespan: {np.mean(costs):.2f}, avg SOC: {np.mean(socs):.2f}, avg time: {np.mean(times):.2f}ms)"
            print(result_msg)
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(result_msg + "\n\n")
    
    return results


def create_results_tables(
    results: dict,
    num_agents_list: list[int],
    k_values: list[int],
    output_dir: Path,
) -> None:
    """Create four tables: success rate, average makespan, average SOC, and average time."""
    
    # Prepare data for tables
    success_data = []
    cost_data = []  # Makespan
    soc_data = []   # Sum of Costs
    time_data = []
    
    for num_agents in num_agents_list:
        success_row = {'Agents': num_agents}
        cost_row = {'Agents': num_agents}
        soc_row = {'Agents': num_agents}
        time_row = {'Agents': num_agents}
        
        for k in k_values:
            key = (num_agents, k)
            data = results[key]
            
            # Success rate (out of 50 scenarios)
            success_rate = data['success_count']
            success_row[f'k={k}'] = success_rate
            
            # Average makespan (only for successful runs)
            if data['costs']:
                avg_cost = np.mean(data['costs'])
                cost_row[f'k={k}'] = f"{avg_cost:.2f}"
            else:
                cost_row[f'k={k}'] = "N/A"
            
            # Average SOC (only for successful runs)
            if data['socs']:
                avg_soc = np.mean(data['socs'])
                soc_row[f'k={k}'] = f"{avg_soc:.2f}"
            else:
                soc_row[f'k={k}'] = "N/A"
            
            # Average time (only for successful runs)
            if data['times']:
                avg_time = np.mean(data['times'])
                time_row[f'k={k}'] = f"{avg_time:.2f}"
            else:
                time_row[f'k={k}'] = "N/A"
        
        success_data.append(success_row)
        cost_data.append(cost_row)
        soc_data.append(soc_row)
        time_data.append(time_row)
    
    # Create DataFrames
    df_success = pd.DataFrame(success_data)
    df_cost = pd.DataFrame(cost_data)
    df_soc = pd.DataFrame(soc_data)
    df_time = pd.DataFrame(time_data)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    df_success.to_csv(output_dir / "success_rate.csv", index=False)
    df_cost.to_csv(output_dir / "average_makespan.csv", index=False)
    df_soc.to_csv(output_dir / "average_soc.csv", index=False)
    df_time.to_csv(output_dir / "average_time.csv", index=False)
    
    # Create formatted table plots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Table 1: Success Rate
    ax1 = axes[0, 0]
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(
        cellText=df_success.values,
        colLabels=df_success.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 2)
    ax1.set_title('Success Rate (out of 50 scenarios)', fontsize=12, fontweight='bold', pad=15)
    
    # Color header row
    for i in range(len(df_success.columns)):
        table1[(0, i)].set_facecolor('#4472C4')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    # Table 2: Average Makespan
    ax2 = axes[0, 1]
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(
        cellText=df_cost.values,
        colLabels=df_cost.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2)
    ax2.set_title('Average Makespan', fontsize=12, fontweight='bold', pad=15)
    
    # Color header row
    for i in range(len(df_cost.columns)):
        table2[(0, i)].set_facecolor('#4472C4')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    # Table 3: Average SOC
    ax3 = axes[1, 0]
    ax3.axis('tight')
    ax3.axis('off')
    table3 = ax3.table(
        cellText=df_soc.values,
        colLabels=df_soc.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table3.auto_set_font_size(False)
    table3.set_fontsize(9)
    table3.scale(1, 2)
    ax3.set_title('Average Sum of Costs (SOC)', fontsize=12, fontweight='bold', pad=15)
    
    # Color header row
    for i in range(len(df_soc.columns)):
        table3[(0, i)].set_facecolor('#4472C4')
        table3[(0, i)].set_text_props(weight='bold', color='white')
    
    # Table 4: Average Time
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    table4 = ax4.table(
        cellText=df_time.values,
        colLabels=df_time.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table4.auto_set_font_size(False)
    table4.set_fontsize(9)
    table4.scale(1, 2)
    ax4.set_title('Average Planning Time (ms)', fontsize=12, fontweight='bold', pad=15)
    
    # Color header row
    for i in range(len(df_time.columns)):
        table4[(0, i)].set_facecolor('#4472C4')
        table4[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(output_dir / "results_tables.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "results_tables.pdf", bbox_inches='tight')
    print(f"\nTables saved to {output_dir}/")
    print(f"  - results_tables.png (high resolution)")
    print(f"  - results_tables.pdf (vector format)")
    print(f"  - success_rate.csv")
    print(f"  - average_makespan.csv")
    print(f"  - average_soc.csv")
    print(f"  - average_time.csv")
    
    # Print tables to console
    print("\n" + "="*80)
    print("SUCCESS RATE (out of 50 scenarios)")
    print("="*80)
    print(df_success.to_string(index=False))
    print("\n" + "="*80)
    print("AVERAGE MAKESPAN")
    print("="*80)
    print(df_cost.to_string(index=False))
    print("\n" + "="*80)
    print("AVERAGE SUM OF COSTS (SOC)")
    print("="*80)
    print(df_soc.to_string(index=False))
    print("\n" + "="*80)
    print("AVERAGE PLANNING TIME (ms)")
    print("="*80)
    print(df_time.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run k-robust MAPF experiments on empty-16-16 map"
    )
    parser.add_argument(
        "--map-file",
        type=Path,
        default=Path(__file__).parent / "assets" / "empty-16-16.map",
    )
    parser.add_argument(
        "--scen-dir",
        type=Path,
        default=Path(__file__).parent / "assets" / "scen-empty-16-16",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="Number of agents to test (default: 5 10 15)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(range(8)),  # 0 to 7 inclusive
        help="k-robust values to test (default: 0 1 2 3 4 5 6 7)",
    )
    parser.add_argument(
        "--time-limit-ms",
        type=int,
        default=5000,
        help="Time limit per run in milliseconds (default: 60000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results_empty_16_16",
        help="Output directory for results",
    )
    parser.add_argument(
        "--flg-star",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LaCAM* (default: True)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level (default: 0)",
    )
    
    args = parser.parse_args()
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.output_dir / f"experiment_log_{timestamp}.txt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("k-Robust MAPF Experiment: empty-16-16")
    print("="*80)
    print(f"Map file: {args.map_file}")
    print(f"Scenario directory: {args.scen_dir}")
    print(f"Number of agents: {args.num_agents}")
    print(f"k values: {args.k_values}")
    print(f"Time limit: {args.time_limit_ms}ms")
    print(f"LaCAM*: {args.flg_star}")
    print(f"Output directory: {args.output_dir}")
    print(f"Log file: {log_file}")
    print("="*80)
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("k-Robust MAPF Experiment: empty-16-16\n")
        f.write("="*80 + "\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Map file: {args.map_file}\n")
        f.write(f"Scenario directory: {args.scen_dir}\n")
        f.write(f"Number of agents: {args.num_agents}\n")
        f.write(f"k values: {args.k_values}\n")
        f.write(f"Time limit: {args.time_limit_ms}ms\n")
        f.write(f"LaCAM*: {args.flg_star}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write("="*80 + "\n\n")
    
    # Run experiments
    results = run_experiments(
        map_file=args.map_file,
        scen_dir=args.scen_dir,
        num_agents_list=args.num_agents,
        k_values=args.k_values,
        time_limit_ms=args.time_limit_ms,
        verbose=args.verbose,
        flg_star=args.flg_star,
        log_file=log_file,
    )
    
    # Create tables
    create_results_tables(
        results=results,
        num_agents_list=args.num_agents,
        k_values=args.k_values,
        output_dir=args.output_dir,
    )
    
    # Write completion to log file
    with open(log_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Experiment completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    print(f"\nExperiment completed!")
    print(f"Log file saved to: {log_file}")
