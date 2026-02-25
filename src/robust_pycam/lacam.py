from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from .dist_table import DistTable
from .mapf_utils import (
    Config,
    Configs,
    Coord,
    Deadline,
    Grid,
    export_search_tree_dot as _export_search_tree_dot,
    export_search_tree_dot_with_low_level as _export_search_tree_dot_with_low_level,
    get_neighbors,check_coord_k_robust
)
from .pibt import PIBT

@dataclass
class LowLevelNode:
    who: list[int] = field(default_factory=lambda: [])
    where: list[Coord] = field(default_factory=lambda: [])
    depth: int = 0

    def get_child(self, who: int, where: Coord) -> LowLevelNode:
        return LowLevelNode(
            who=self.who + [who],
            where=self.where + [where],
            depth=self.depth + 1,
        )

@dataclass
class HighLevelNode:
    Q: Config
    parent: HighLevelNode | None = None
    order: list[int] = field(default_factory=list)
    tree: deque[LowLevelNode] = field(default_factory=lambda: deque([LowLevelNode()]))
    g: int = 0
    h: int = 0
    f: int = 0
    neighbors: set[HighLevelNode] = field(default_factory=lambda: set())

    history: tuple[Config, ...] = field(default_factory=tuple)

    # Low-level nodes that were actually popped/processed ("tried") while expanding this state.
    # We keep them even if they later led to invalid configs / k-robust rejection / duplicates.
    ll_tried: list[LowLevelNode] = field(default_factory=list)
    # Maps low-level key -> status string.
    ll_status: dict[tuple[tuple[int, ...], tuple[Coord, ...]], str] = field(default_factory=dict)
    # Maps low-level key -> resulting high-level state_key (Q_to, history_to) when accepted.
    ll_to_state: dict[tuple[tuple[int, ...], tuple[Coord, ...]], tuple[Config, tuple[Config, ...]]] = field(
        default_factory=dict
    )

    def __post_init__(self): #todo - why we need it?
        self.f = self.g + self.h

    def __eq__(self, other) -> bool:
        if isinstance(other, HighLevelNode):
            return self.Q == other.Q and self.history == other.history
        return False

    def __hash__(self) -> int:
        return hash((self.Q, self.history))

class LaCAM:
    def __init__(self) -> None:
        self.best_solution_time_ms: float | None = None

    def solve(
        self,
        grid: Grid,
        starts: Config,
        goals: Config,
        time_limit_ms: int = 3000,
        deadline: Deadline | None = None,
        flg_star: bool = True,
        seed: int = 0,
        verbose: int = 1,
        k_robust: int = 5,
    ) -> Configs:
        self.num_agents: int = len(starts)
        self.grid: Grid = grid
        self.starts: Config = starts
        self.goals: Config = goals
        self.deadline: Deadline = deadline if deadline is not None else Deadline(time_limit_ms)
        self.flg_star: bool = flg_star
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)
        self.verbose = verbose
        self.k_robust = k_robust
        return self._solve()

    def _solve(self) -> Configs:
        self.info(1, f"start solving MAPF (k-robust={self.k_robust}, {self.num_agents} agents)")

        self.best_solution_time_ms = None

        self.dist_tables = [DistTable(self.grid, g) for g in self.goals]
        self.pibt = PIBT(self.dist_tables)

        OPEN: deque[HighLevelNode] = deque([])
        EXPLORED: dict[tuple[Config, tuple[Config, ...]], HighLevelNode] = {}
        N_goal: HighLevelNode | None = None

        Q_init = self.starts
        history_init = (Q_init,)
        N_init = HighLevelNode(
            Q=Q_init, 
            order=self.get_order(Q_init), 
            h=self.get_h_value(Q_init),
            history=history_init
        )
        
        OPEN.appendleft(N_init)
        EXPLORED[(N_init.Q, N_init.history)] = N_init

        while len(OPEN) > 0 and not self.deadline.is_expired:

            N: HighLevelNode = OPEN[0]

            # Track best goal node (min g among all nodes with Q==goals)
            if N.Q == self.goals and (N_goal is None or N.g < N_goal.g):
                N_goal = N
                self.best_solution_time_ms = self.deadline.elapsed
                self.info(1, f"goal found/improved, cost={N_goal.g}")
                if not self.flg_star:
                    break

            # lower bound check
            if N_goal is not None and N_goal.g <= N.f:
                OPEN.popleft()
                continue

            # low-level search end
            if len(N.tree) == 0:
                OPEN.popleft()
                continue

            # low-level search
            C: LowLevelNode = N.tree.popleft()
            ll_key = (tuple(C.who), tuple(C.where))
            N.ll_tried.append(C)
            N.ll_status.setdefault(ll_key, "popped")
            if C.depth < self.num_agents:
                i = N.order[C.depth]
                v = N.Q[i]
                cands = [v] + get_neighbors(self.grid, v)
                self.rng.shuffle(cands)
                for u in cands:
                    N.tree.append(C.get_child(i, u))

            Q_to = self.configuration_generaotr(N, C)
            if Q_to is None: 
                # invalid configuration
                N.ll_status[ll_key] = "invalid_config"
                continue
            if not self.check_k_robust(N, Q_to):
                    N.ll_status[ll_key] = "k_robust_reject"
                    continue

            history_to = (
                (N.history + (Q_to,))[-(self.k_robust + 1) :] if self.k_robust > 0 else (Q_to,)
            )
            state_key = (Q_to, history_to)
    
            if state_key in EXPLORED:
                N.ll_status[ll_key] = "accepted_known_state"
                N.ll_to_state[ll_key] = state_key
                N_known = EXPLORED[state_key]
                N.neighbors.add(N_known)
                OPEN.appendleft(N_known)

                # Dijkstra update
                D = deque([N])
                while len(D) > 0 and self.flg_star:
                    N_from = D.popleft()
                    for N_to in N_from.neighbors:
                        # IMPORTANT: only relax along edges that are history-compatible.
                        # State is (Q, history). A transition from N_from to N_to is only valid
                        # if N_to.history equals the truncated window of (N_from.history + (N_to.Q,)).
                        if self.k_robust > 0:
                            expected_history = (N_from.history + (N_to.Q,))[-(self.k_robust + 1) :]
                        else:
                            expected_history = (N_to.Q,)
                        if expected_history != N_to.history:
                            continue

                        g = N_from.g + self.get_edge_cost(N_from.Q, N_to.Q)
                        if g < N_to.g:
                            N_to.g = g
                            N_to.f = N_to.g + N_to.h
                            N_to.parent = N_from
                            D.append(N_to)
                            if N_to.Q == self.goals and (N_goal is None or g < N_goal.g):
                                N_goal = N_to
                                self.best_solution_time_ms = self.deadline.elapsed
                                self.info(1, f"cost update: goal g={g:4d}")
                            if N_goal is not None and N_to.f < N_goal.g:
                                OPEN.appendleft(N_to) 

            else:
                

                N_new = HighLevelNode(
                    Q=Q_to,
                    parent=N,
                    order=self.get_order(Q_to),
                    g=N.g + self.get_edge_cost(N.Q, Q_to),
                    h=self.get_h_value(Q_to),
                    history=history_to
                )
                N.ll_status[ll_key] = "accepted_new_state"
                N.ll_to_state[ll_key] = state_key
                N.neighbors.add(N_new)
                OPEN.appendleft(N_new)
                EXPLORED[state_key] = N_new

        # store for search tree export (each node = state (Q, history))
        self._explored: dict = EXPLORED
        self._N_init: HighLevelNode | None = N_init
        self._N_goal: HighLevelNode | None = N_goal
        if len(OPEN) == 0:
            self.info(1, f"open == 0")
        self.info(1, f"size of explored: {len(EXPLORED)}")

        return self.backtrack(N_goal)

    def _state_label(self, N: HighLevelNode) -> str:
        """State label: Q then last k_robust history configs (the window used for k-robust check)."""
        def _config_line(Q: Config) -> str:
            return "|".join(f"{Q[i][0]},{Q[i][1]}" for i in range(len(Q)))
        lines = [f"Q: {_config_line(N.Q)} g:{N.g}"]
        # Last k steps (same as k-robust validity check)
        k = max(0, self.k_robust)
        last_k = N.history[-k:] if k else []
        for t, h in enumerate(last_k):
            lines.append(f"h-{len(last_k)-t}: {_config_line(h)}")
        return "\n".join(lines)

    def export_search_tree_dot(self, filepath: str, include_low_level: bool = False) -> None:
        """
        Write the high-level search tree to a Graphviz DOT file.
        Each node is a state (Q, history); edges are parent -> child.
        Render with: dot -Tpng -o tree.png <filepath>
        """
        if not hasattr(self, "_explored") or self._explored is None:
            raise RuntimeError("Run solve() before export_search_tree_dot()")
        export_fn = _export_search_tree_dot_with_low_level if include_low_level else _export_search_tree_dot
        export_fn(filepath, self._explored, self._N_init, self._N_goal, self._state_label)

    def check_k_robust(self, N: HighLevelNode, Q_next: Config) -> bool:
        if self.k_robust == 0:
            return True
        for agent_id, cand in enumerate(Q_next):
            if not check_coord_k_robust(
                cand=cand,
                agent_id=agent_id,
                history=N.history,
                k_robust=self.k_robust,
            ):
                return False
        return True

    def backtrack(self, _N: HighLevelNode | None) -> Configs:
        configs = []
        N = _N
        while N is not None:
            configs.append(N.Q)
            N = N.parent
        configs.reverse()
        return configs

    def get_edge_cost(self, Q_from: Config, Q_to: Config) -> int:
        cost = 0
        for i in range(self.num_agents):
            if not (self.goals[i] == Q_from[i] == Q_to[i]):
                cost += 1
        return cost

    def get_h_value(self, Q: Config) -> int:
        # e.g., \sum_i dist(Q[i], g_i)
        cost = 0
        for agent_idx, loc in enumerate(Q):
            c = self.dist_tables[agent_idx].get(loc)
            if c is None:
                return np.iinfo(np.int32).max
            cost += c
        return cost

    def get_order(self, Q: Config) -> list[int]:
        # e.g., by descending order of dist(Q[i], g_i)
        # Note that this is not an effective PIBT prioritization scheme
        order = list(range(self.num_agents))
        self.rng.shuffle(order)
        order.sort(key=lambda i: self.dist_tables[i].get(Q[i]), reverse=True)
        return order

    def configuration_generaotr(self, N: HighLevelNode, C: LowLevelNode) -> Config | None:
        Q_to = Config([self.pibt.NIL_COORD for _ in range(self.num_agents)])
        for k in range(C.depth):
            Q_to[C.who[k]] = C.where[k]

        # apply PIBT
        success = self.pibt.step(N.Q, Q_to, N.order)
        return Q_to if success else None

    def info(self, level: int, msg: str) -> None:
        if self.verbose < level:
            return
        logger.debug(f"{int(self.deadline.elapsed):4d}ms  {msg}")
