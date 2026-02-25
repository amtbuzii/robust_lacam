"""
Wrapper for k-robust CBS solution from Lazy-Train-and-K-CBS repository.
This module provides a Python interface to the k-robust CBS solver.
"""
import subprocess
import re
import tempfile
import os
from pathlib import Path
from typing import Optional
import time

from src.pycam.mapf_utils import Grid, Config, Configs, Deadline


class KRobustCBS:
    """
    Wrapper for k-robust CBS solver from Lazy-Train-and-K-CBS.
    
    This class provides an interface similar to LaCAM for easy integration.
    """
    
    def __init__(
        self,
        executable_path: Optional[Path] = None,
        solver: str = "CBSH-RM",
        corridor: bool = True,
        target: bool = True,
    ):
        """
        Initialize the k-robust CBS wrapper.
        
        Args:
            executable_path: Path to the compiled CBS-K executable.
                           If None, will try to find it in k_robust_cbs/build/
            solver: Solver algorithm to use (CBS, ICBS, CBSH, CBSH-RM, etc.)
            corridor: Enable corridor reasoning
            target: Enable target reasoning
        """
        if executable_path is None:
            # Try to find the executable in the cloned repo
            repo_root = Path(__file__).parent.parent.parent / "k_robust_cbs"
            possible_paths = [
                repo_root / "build" / "CBS-K",
                repo_root / "CBSH-rect-cmake" / "build" / "CBS-K",
            ]
            for path in possible_paths:
                if path.exists() and os.access(path, os.X_OK):
                    executable_path = path
                    break
            
            if executable_path is None:
                raise FileNotFoundError(
                    f"CBS-K executable not found. Please compile it first:\n"
                    f"  cd {repo_root}/CBSH-rect-cmake\n"
                    f"  mkdir -p build && cd build\n"
                    f"  cmake ..\n"
                    f"  make -j\n"
                )
        
        self.executable_path = Path(executable_path)
        if not self.executable_path.exists():
            raise FileNotFoundError(f"Executable not found: {executable_path}")
        
        self.solver = solver
        self.corridor = corridor
        self.target = target
    
    def solve(
        self,
        grid: Grid,
        starts: Config,
        goals: Config,
        time_limit_ms: int = 3000,
        deadline: Optional[Deadline] = None,
        flg_star: bool = True,  # Not used for CBS, but kept for compatibility
        seed: int = 0,
        verbose: int = 1,
        k_robust: int = 3,
    ) -> Configs:
        """
        Solve k-robust MAPF problem.
        
        Args:
            grid: Grid map (numpy array)
            starts: Starting positions for agents
            goals: Goal positions for agents
            time_limit_ms: Time limit in milliseconds
            deadline: Deadline object (not used, kept for compatibility)
            flg_star: Not used for CBS (kept for compatibility)
            seed: Random seed
            verbose: Verbosity level
            k_robust: k value for k-robustness
            
        Returns:
            Configs: List of Config objects representing the solution
        """
        # Create temporary files for map and scenario
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            map_file = tmp_path / "map.map"
            scen_file = tmp_path / "scenario.scen"
            output_file = tmp_path / "output.txt"
            
            # Write map file
            self._write_map_file(map_file, grid)
            
            # Write scenario file
            height, width = grid.shape
            self._write_scenario_file(scen_file, starts, goals, width, height)
            
            # Build command
            # Optimal CBS needs more time - use at least 10 seconds or the provided limit
            time_limit_sec = max(10.0, time_limit_ms / 1000.0)
            cmd = [
                str(self.executable_path),
                "-m", str(map_file),
                "-a", str(scen_file),
                "-o", str(output_file),
                "-s", self.solver,
                "-t", str(time_limit_sec),
                "-d", str(seed),  # Use -d for seed (not --seed)
                "--screen", str(verbose),
                "--kDelay", str(k_robust),
                "--agentNum", str(len(starts)),  # Specify number of agents
                "--ignore-train",  # This enables k-robust CBS mode
                "--corridor", "True" if self.corridor else "False",
                "--target", "True" if self.target else "False",
                "--printPath",  # Print paths to stdout
            ]
            
            if verbose >= 1:
                print(f"Running k-robust CBS: {' '.join(cmd)}")
            
            # Run the solver
            start_time = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=time_limit_sec + 1,
                    cwd=tmp_path,
                )
                elapsed_time = (time.time() - start_time) * 1000
                
                if verbose >= 1:
                    print(f"k-robust CBS completed in {elapsed_time:.2f}ms")
                    if result.returncode != 0:
                        print(f"Warning: CBS-K returned code {result.returncode}")
                    if result.stderr:
                        print(f"Stderr: {result.stderr}")
                    if verbose >= 1:
                        print(f"Stdout length: {len(result.stdout)} chars")
                        if len(result.stdout) > 0:
                            print(f"Stdout (first 1000 chars):\n{result.stdout[:1000]}")
                        else:
                            print("Stdout is empty")
                        # Also check output file
                        if output_file.exists():
                            with open(output_file, 'r') as f:
                                output_content = f.read()
                                print(f"Output file content:\n{output_content}")
                
                # Parse paths from stdout
                solution = self._parse_paths(result.stdout, len(starts))
                
                if solution is None or len(solution) == 0:
                    if verbose >= 1:
                        print("Warning: No solution found or failed to parse paths")
                        print(f"Full stdout:\n{result.stdout}")
                    return []
                
                return solution
                
            except subprocess.TimeoutExpired:
                if verbose >= 1:
                    print(f"k-robust CBS timed out after {time_limit_sec}s")
                return []
            except Exception as e:
                if verbose >= 1:
                    print(f"Error running k-robust CBS: {e}")
                return []
    
    def _write_map_file(self, map_file: Path, grid: Grid) -> None:
        """Write grid to .map file format."""
        height, width = grid.shape
        
        with open(map_file, 'w') as f:
            f.write(f"type octile\n")
            f.write(f"height {height}\n")
            f.write(f"width {width}\n")
            f.write(f"map\n")
            
            for row in grid:
                line = ""
                for cell in row:
                    if cell == 0:  # obstacle
                        line += "@"
                    else:  # free space
                        line += "."
                f.write(line + "\n")
    
    def _write_scenario_file(self, scen_file: Path, starts: Config, goals: Config, width: int, height: int) -> None:
        """Write scenario to .scen file format.
        
        Format: bucket map_name width height start_x start_y goal_x goal_y optimal_length
        Note: starts and goals are in (y, x) format, but .scen file uses (x, y)
        """
        map_name = "temp.map"
        
        with open(scen_file, 'w') as f:
            f.write("version 1\n")
            for i, (start, goal) in enumerate(zip(starts, goals)):
                # .scen format: bucket map_name width height start_x start_y goal_x goal_y optimal_length
                # starts/goals are in (y, x) format, but .scen uses (x, y)
                start_x, start_y = start[1], start[0]  # Convert (y, x) to (x, y)
                goal_x, goal_y = goal[1], goal[0]
                optimal_length = abs(start_x - goal_x) + abs(start_y - goal_y)
                f.write(f"0\t{map_name}\t{width}\t{height}\t{start_x}\t{start_y}\t{goal_x}\t{goal_y}\t{optimal_length:.8f}\n")
    
    def _parse_paths(self, stdout: str, num_agents: int) -> Optional[Configs]:
        """
        Parse paths from CBS-K stdout.
        
        Format: Agent 0 (size --> size): (y,x)->(y,x)->...
        """
        solution = []
        
        # Pattern to match: Agent N (size --> size): (y,x)->(y,x)->...
        pattern = re.compile(r'Agent (\d+).*?:\s*(\([^)]+\)(?:->\([^)]+\))*)')
        
        matches = pattern.findall(stdout)
        
        if len(matches) == 0:
            # Try alternative parsing - look for lines with coordinates
            lines = stdout.split('\n')
            agent_paths = {}
            
            for line in lines:
                if 'Agent' in line and '->' in line:
                    # Extract agent number and path
                    agent_match = re.search(r'Agent (\d+)', line)
                    if agent_match:
                        agent_id = int(agent_match.group(1))
                        # Extract all (y,x) coordinates
                        coords = re.findall(r'\((\d+),(\d+)\)', line)
                        if coords:
                            agent_paths[agent_id] = [(int(y), int(x)) for y, x in coords]
            
            if len(agent_paths) == num_agents:
                # Convert to Configs format
                max_length = max(len(path) for path in agent_paths.values())
                
                for t in range(max_length):
                    config = Config()
                    for agent_id in range(num_agents):
                        if agent_id in agent_paths:
                            path = agent_paths[agent_id]
                            if t < len(path):
                                config.append(path[t])
                            else:
                                # Stay at goal
                                config.append(path[-1])
                        else:
                            # Agent not found, use goal as fallback
                            config.append((0, 0))
                    solution.append(config)
                
                return solution
        
        # If we have matches, use them
        if len(matches) > 0:
            agent_paths = {}
            for agent_str, path_str in matches:
                agent_id = int(agent_str)
                coords = re.findall(r'\((\d+),(\d+)\)', path_str)
                if coords:
                    agent_paths[agent_id] = [(int(y), int(x)) for y, x in coords]
            
            if len(agent_paths) > 0:
                max_length = max(len(path) for path in agent_paths.values())
                
                for t in range(max_length):
                    config = Config()
                    for agent_id in range(num_agents):
                        if agent_id in agent_paths:
                            path = agent_paths[agent_id]
                            if t < len(path):
                                config.append(path[t])
                            else:
                                config.append(path[-1])
                        else:
                            config.append((0, 0))
                    solution.append(config)
                
                return solution
        
        return None
