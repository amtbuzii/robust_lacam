"""
Profiling script for the alternative robust LaCAM algorithm.

This script runs the alternative robust algorithm with various test cases to generate
profiling data. Run with:
    python -m cProfile -o robust_profiler_output.prof robust_profiler.py
"""

from pathlib import Path

from src.alt_robust_pycam import LaCAM as AltRobustLaCAM
from src.alt_robust_pycam.mapf_utils import (
    get_grid,
    get_scenario,
    validate_robust_mapf_solution,
)


def run_profiling_tests():
    """Run multiple test cases to profile the robust algorithm."""
    
    # Test cases: (map_file, scen_file, num_agents, k_robust, time_limit_ms)
    test_cases = [
        # Small test cases
        ("assets/empty-8-8.map", "assets/empty-8-8-random-1.scen", 4, 3, 5000),
        ("assets/empty-8-8.map", "assets/empty-8-8-random-1.scen", 4, 5, 5000),
        
        # Medium test cases
        ("assets/empty-16-16.map", "assets/scen-empty-16-16/empty-16-16-random-1.scen", 5, 3, 10000),
        ("assets/empty-16-16.map", "assets/scen-empty-16-16/empty-16-16-random-2.scen", 5, 5, 10000),
        ("assets/empty-16-16.map", "assets/scen-empty-16-16/empty-16-16-random-3.scen", 10, 3, 15000),
        ("assets/empty-16-16.map", "assets/scen-empty-16-16/empty-16-16-random-4.scen", 10, 5, 15000),
        
        # Larger test cases
        ("assets/empty-16-16.map", "assets/scen-empty-16-16/empty-16-16-random-5.scen", 15, 3, 20000),
        ("assets/empty-16-16.map", "assets/scen-empty-16-16/empty-16-16-random-6.scen", 15, 5, 20000),
        
        # Maze test cases
        ("assets/maze-32-32-4.map", "assets/maze-32-32-4-random-1.scen", 6, 3, 30000),
        ("assets/maze-32-32-4.map", "assets/maze-32-32-4-random-1.scen", 6, 5, 30000),
    ]
    
    planner = AltRobustLaCAM()
    base_path = Path(__file__).parent
    
    print("=" * 80)
    print("Profiling Alternative Robust LaCAM Algorithm")
    print("=" * 80)
    
    for i, (map_file, scen_file, num_agents, k_robust, time_limit_ms) in enumerate(test_cases, 1):
        map_path = base_path / map_file
        scen_path = base_path / scen_file
        
        if not map_path.exists():
            print(f"  Test {i}: SKIPPED - Map file not found: {map_path}")
            continue
        if not scen_path.exists():
            print(f"  Test {i}: SKIPPED - Scenario file not found: {scen_path}")
            continue
        
        print(f"\n  Test {i}/{len(test_cases)}: {map_path.name} | {num_agents} agents | k={k_robust}")
        
        try:
            # Load grid and scenario
            grid = get_grid(map_path)
            starts, goals = get_scenario(scen_path, num_agents)
            
            # Run the alternative robust algorithm
            solution = planner.solve(
                grid=grid,
                starts=starts,
                goals=goals,
                seed=0,
                time_limit_ms=time_limit_ms,
                flg_star=False,  # Use LaCAM*
                verbose=0,  # No verbose output during profiling
                k_robust=k_robust,
            )
            
            # Validate solution if found
            if solution and len(solution) > 0:
                try:
                    validate_robust_mapf_solution(grid, starts, goals, solution, k_robust)
                    cost = len(solution) - 1
                    print(f"    ✓ SUCCESS - Makespan: {cost}")
                except Exception as e:
                    print(f"    ✗ VALIDATION FAILED: {e}")
            else:
                print(f"    ✗ NO SOLUTION FOUND")
                
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("Profiling complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_profiling_tests()
