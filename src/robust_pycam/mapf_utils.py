import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeAlias

import numpy as np

Grid: TypeAlias = np.ndarray
Coord: TypeAlias = tuple[int, int]  # y, x


@dataclass
class Config:
    positions: list[Coord] = field(default_factory=lambda: [])

    def __getitem__(self, k: int) -> Coord:
        return self.positions[k]

    def __setitem__(self, k: int, coord: Coord) -> None:
        self.positions[k] = coord

    def __len__(self) -> int:
        return len(self.positions)

    def __hash__(self) -> int:
        return hash(tuple(self.positions))

    def append(self, coord: Coord) -> None:
        self.positions.append(coord)


Configs: TypeAlias = list[Config]


@dataclass
class Deadline:
    time_limit_ms: int

    def __post_init__(self) -> None:
        self.start_time = time.time()

    @property
    def elapsed(self) -> float:
        return (time.time() - self.start_time) * 1000

    @property
    def is_expired(self) -> bool:
        return self.elapsed > self.time_limit_ms


def get_grid(map_file: str | Path) -> Grid:
    width, height = 0, 0
    with open(map_file, "r") as f:
        # retrieve map size
        for row in f:
            # get width
            res = re.match(r"width\s(\d+)", row)
            if res:
                width = int(res.group(1))

            # get height
            res = re.match(r"height\s(\d+)", row)
            if res:
                height = int(res.group(1))

            if width > 0 and height > 0:
                break

        # retrieve map
        grid = np.zeros((height, width), dtype=bool)
        y = 0
        for row in f:
            row = row.strip()
            if len(row) == width and row != "map":
                grid[y] = [s == "." for s in row]
                y += 1

    # simple error check
    assert y == height, f"map format seems strange, check {map_file}"

    # grid[y, x] -> True: available, False: obstacle
    return grid


def get_scenario(scen_file: str | Path, N: int | None = None) -> tuple[Config, Config]:
    with open(scen_file, "r") as f:
        starts, goals = Config(), Config()
        for row in f:
            res = re.match(
                r"\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+", row
            )
            if res:
                x_s, y_s, x_g, y_g = [int(res.group(k)) for k in range(1, 5)]
                starts.append((y_s, x_s))  # align with grid
                goals.append((y_g, x_g))

                # check the number of agents
                if (N is not None) and len(starts) >= N:
                    break

    return starts, goals


def is_valid_coord(grid: Grid, coord: Coord) -> bool:
    y, x = coord
    if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1] or not grid[coord]:
        return False
    return True


def get_neighbors(grid: Grid, coord: Coord) -> list[Coord]:
    # coord: y, x
    neigh: list[Coord] = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh

    y, x = coord

    if x > 0 and grid[y, x - 1]:
        neigh.append((y, x - 1))

    if x < grid.shape[1] - 1 and grid[y, x + 1]:
        neigh.append((y, x + 1))

    if y > 0 and grid[y - 1, x]:
        neigh.append((y - 1, x))

    if y < grid.shape[0] - 1 and grid[y + 1, x]:
        neigh.append((y + 1, x))

    return neigh


def save_configs_for_visualizer(configs: Configs, filename: str | Path) -> None:
    output_dirname = Path(filename).parent
    if not output_dirname.exists():
        output_dirname.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        for t, config in enumerate(configs):
            row = f"{t}:" + "".join([f"({x},{y})," for (y, x) in config]) + "\n"
            f.write(row)

def check_coord_k_robust(
    cand: Coord,
    agent_id: int,
    history: tuple[Config, ...] | list[Config],
    k_robust: int | None = None,
) -> bool:
    """
    Single-agent k-robust check: candidate next position is safe iff for every
    past config in history, either the cell was not occupied or it was this
    agent's position. Returns (True, []) when safe, (False, intermediate) when
    not safe, with intermediate configs from the conflict point (same as
    check_k_robust in lacam).
    """
    configs_to_check = history[max(0, len(history) - k_robust) :]
  
    for hist_Q in configs_to_check:
        if cand in hist_Q and cand != hist_Q[agent_id]:
            return False

    return True

def export_search_tree_dot(
    filepath: str | Path,
    explored: dict,
    N_init,
    N_goal,
    label_fn: Callable[[object], str],
) -> None:
    """
    Write a high-level search tree to a Graphviz DOT file.
    Each node is a state; edges are parent -> child.
    explored: dict mapping state_key -> node (node must have .parent, .Q, .history).
    N_init, N_goal: optional root and goal nodes (for styling).
    label_fn(node): returns label string for the node.
    Nodes are keyed by state (Q, history) so the graph is one tree even when
    the same state is stored under different node objects.
    Render with: dot -Tpng -o tree.png <filepath>
    """
    lines = ["digraph search_tree {", "  node [shape=box, fontsize=10];"]
    state_to_id: dict = {}
    for i, (state_key, N) in enumerate(explored.items()):
        nid = f"n{i}"
        state_to_id[state_key] = nid
        label = label_fn(N).replace('"', '\\"')
        extra = ""
        if N_init is not None and N is N_init:
            extra = ", style=bold, color=green"
        if N_goal is not None and N is N_goal:
            extra = ", style=bold, color=blue"
        lines.append(f'  {nid} [label="{label}"{extra}];')
    for state_key, N in explored.items():
        nid = state_to_id[state_key]
        if N.parent is not None:
            parent_key = (N.parent.Q, N.parent.history)
            if parent_key in state_to_id:
                pid = state_to_id[parent_key]
                lines.append(f"  {pid} -> {nid};")
    lines.append("}")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def validate_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> None:
    assert len(solution) > 0, "invalid solution, empty"

    # starts
    assert all(
        [u == v for (u, v) in zip(starts, solution[0])]
    ), "invalid solution, check starts"

    # goals
    assert all(
        [u == v for (u, v) in zip(goals, solution[-1])]
    ), "invalid solution, check goals"

    T = len(solution)
    N = len(starts)

    for t in range(T):
        for i in range(N):
            v_i_now = solution[t][i]
            v_i_pre = solution[max(t - 1, 0)][i]

            # check continuity
            assert v_i_now in [v_i_pre] + get_neighbors(
                grid, v_i_pre
            ), "invalid solution, check connectivity"

            # check collision
            for j in range(i + 1, N):
                v_j_now = solution[t][j]
                v_j_pre = solution[max(t - 1, 0)][j]
                assert not (v_i_now == v_j_now), "invalid solution, vertex collision"
                assert not (
                    v_i_now == v_j_pre and v_i_pre == v_j_now
                ), "invalid solution, edge collision"


def validate_k_robust_solution(
    solution: Configs,
    k_robust: int,
) -> None:
    """
    Validate that a solution satisfies k-robustness.
    
    k-robustness: If an agent occupies vertex v at time t, 
    no other agent can occupy v during the interval [t, t+k].
    
    Args:
        solution: List of Config objects representing the solution path
        k_robust: The robustness parameter k
        
    Raises:
        AssertionError: If the solution violates k-robustness
    """
    if k_robust == 0:
        return  # No k-robustness constraint
    
    assert len(solution) > 0, "invalid solution, empty"
    
    T = len(solution)
    N = len(solution[0]) if solution else 0
    
    if N == 0:
        return  # No agents to check
    
    for t in range(T):
        config_t = solution[t]
        
        # For each agent at time t
        for agent_i in range(N):
            loc_i = config_t[agent_i]
            
            # Check historical time steps: t-1, t-2, ..., t-k (but not before 0)
            start_time = max(0, t - k_robust)
            
            for t_hist in range(start_time, t):
                config_hist = solution[t_hist]
                
                # Check if any OTHER agent was at loc_i at time t_hist
                for agent_j in range(N):
                    if agent_i != agent_j:  # Different agent
                        loc_j = config_hist[agent_j]
                        if loc_i == loc_j:
                            raise AssertionError(
                                f"invalid k-robust solution: agent {agent_i} at location {loc_i} "
                                f"at time {t} conflicts with agent {agent_j} at the same location "
                                f"at time {t_hist} (k={k_robust}, time difference: {t - t_hist})"
                            )


def validate_robust_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
    k_robust: int,
) -> None:
    """
    Validate a MAPF solution with k-robustness constraints.
    
    First validates standard MAPF constraints (starts, goals, continuity, collisions),
    then validates k-robustness.
    
    Args:
        grid: Grid array where True = free cell, False = obstacle
        starts: Starting configuration
        goals: Goal configuration
        solution: List of Config objects representing the solution path
        k_robust: The robustness parameter k
        
    Raises:
        AssertionError: If the solution violates any constraint
    """
    # First validate standard MAPF constraints
    validate_mapf_solution(grid, starts, goals, solution)
    
    # Then validate k-robustness
    validate_k_robust_solution(solution, k_robust)


def is_valid_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> bool:
    try:
        validate_mapf_solution(grid, starts, goals, solution)
        return True
    except Exception as e:
        print(e)
        return False


def get_sum_of_loss(configs: Configs) -> int:
    cost = 0
    for t in range(1, len(configs)):
        cost += sum(
            [
                not (v_from == v_to == goal)
                for (v_from, v_to, goal) in zip(configs[t - 1], configs[t], configs[-1])
            ]
        )
    return cost
