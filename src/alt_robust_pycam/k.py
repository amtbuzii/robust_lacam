from __future__ import annotations

from alt_robust_pycam.mapf_utils import Coord
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from .dist_table import DistTable
from .mapf_utils import Config, Configs, Coord, Deadline, Grid, get_neighbors, get_neighbors_safe_k_robust
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
        self.info(1, f"start solving MAPF (k-robust={self.k_robust})")
        self.best_solution_time_ms = None

        self.dist_tables = [DistTable(self.grid, g) for g in self.goals]
        self.pibt = PIBT(self.dist_tables, k_robust=self.k_robust)

        OPEN: deque[HighLevelNode] = deque([])
        EXPLORED: dict[tuple[Config, tuple[Config, ...]], HighLevelNode] = {}
        
        SEEN_ALT: dict[Config, list[HighLevelNode]] = {}
        NEED_ALT: set[Config] = set()
        SEEN_CONFIGS: dict[Config, int] = {}
        

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
        SEEN_CONFIGS[Q_init] = 1
        while len(OPEN) > 0 and not self.deadline.is_expired:
            N: HighLevelNode = OPEN[0]

            if N_goal is None and N.Q == self.goals:
                N_goal = N
                self.best_solution_time_ms = self.deadline.elapsed
                self.info(1, f"initial solution found, cost={N_goal.g}")
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
            if C.depth < self.num_agents:
                i = N.order[C.depth]
                v = N.Q[i]
                cands, _invalid = get_neighbors_safe_k_robust(
                    self.grid, v, self.k_robust, i, N.history
                )

                if _invalid:                
                    for invalid_cand, intermediate in _invalid.items():
                        if invalid_cand in NEED_ALT and invalid_cand in SEEN_ALT:
                                N_alt = SEEN_ALT[invalid_cand].pop()
                                if not SEEN_ALT[invalid_cand]:
                                    del SEEN_ALT[invalid_cand]
                                OPEN.appendleft(N_alt)
                                NEED_ALT.remove(invalid_cand)
                        else:
                            NEED_ALT.add(invalid_cand)

                self.rng.shuffle(cands)
                for u in cands:
                    N.tree.append(C.get_child(i, u))

            Q_to = self.configuration_generaotr(N, C)
            if Q_to is None: 
                # invalid configuration
                continue

            history_to = (N.history + (Q_to,))[-self.k_robust-1:]
            state_key = (Q_to, history_to)

            Q_only_seen = Q_to in SEEN_CONFIGS
    
            if state_key in EXPLORED:
                N_known = EXPLORED[state_key]
                N.neighbors.add(N_known)
                OPEN.appendleft(N_known)

                # Dijkstra update
                D = deque([N_known])
                while len(D) > 0 and self.flg_star:
                    N_from = D.popleft()
                    for N_to in N_from.neighbors:
                        g = N_from.g + self.get_edge_cost(N_from.Q, N_to.Q)
                        if g < N_to.g:
                            N_to.g = g
                            N_to.f = N_to.g + N_to.h
                            N_to.parent = N_from
                            D.append(N_to)
                            if N_goal is not None and N_to is N_goal:
                                self.best_solution_time_ms = self.deadline.elapsed
                                self.info(2, f"cost update: {N_goal.g:4d} -> {g:4d}")
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
                N.neighbors.add(N_new)

                if not Q_only_seen:
                    OPEN.appendleft(N_new)
                    EXPLORED[state_key] = N_new
                    SEEN_CONFIGS[Q_to] = 1
                else:
                    if Q_to in NEED_ALT:
                        OPEN.appendleft(N_new)
                        NEED_ALT.remove(Q_to)
                        EXPLORED[state_key] = N_new
                        SEEN_CONFIGS[Q_to] = 1
                    else:
                        SEEN_ALT.setdefault(Q_to, []).append(N_new)

        if len(OPEN) == 0:
            self.info(1, f"open == 0")

        return self.backtrack(N_goal)

    def check_k_robust(self, N: HighLevelNode, Q_next: Config) -> tuple[bool, list[Config]]:
        if self.k_robust == 0:
            return True, []

        for history_index, hist_Q in enumerate(N.history):
            hist_set = set(hist_Q) 
            for i, loc_next in enumerate(Q_next):
                if loc_next in hist_set:
                    if loc_next != hist_Q[i]:
                        intermediate = list(N.history[history_index:])
                        return False, intermediate
        return True, []

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
        success = self.pibt.step(N.Q, Q_to, N.order, history=N.history)
        return Q_to if success else None

    def info(self, level: int, msg: str) -> None:
        if self.verbose < level:
            return
        logger.debug(f"{int(self.deadline.elapsed):4d}ms  {msg}")