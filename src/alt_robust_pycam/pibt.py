"""
a toy PIBT implementation taken from
https://github.com/Kei18/pypibt
"""

import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Coord, get_neighbors, get_neighbors_safe_k_robust


class PIBT:
    def __init__(
        self,
        dist_tables: list[DistTable],
        seed: int = 0,
        k_robust: int = 0,
    ) -> None:
        self.N = len(dist_tables)
        assert self.N > 0
        self.dist_tables = dist_tables
        self.grid = self.dist_tables[0].grid
        self.k_robust = k_robust

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(self.grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(self.grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def _config_with(self, Q: Config, i: int, new_coord: Coord) -> Config:
        """Build new Config with position i set to new_coord."""
        new_positions = tuple(
            new_coord if j == i else Q.positions[j] for j in range(len(Q.positions))
        )
        return Config(positions=new_positions)

    def funcPIBT(
        self,
        Q_from: Config,
        Q_to: Config,
        i: int,
        history: tuple[Config, ...] | list[Config] = (),
    ) -> tuple[bool, Config]:
        # (success, updated Q_to)

        # get candidate next vertices (k-robust safe when k_robust > 0 and history given)
        valid, invalid = get_neighbors_safe_k_robust(
            self.grid, Q_from[i], self.k_robust, i, history
        )
        cands = valid or (list(invalid.keys()) if invalid else get_neighbors(self.grid, Q_from[i]))
        C = cands

        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: self.dist_tables[i].get(u))

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(
        self,
        Q_from: Config,
        Q_to: list[Coord],
        order: list[int],
        history: tuple[Config, ...] | list[Config] = (),
    ) -> bool:
        flg_success = True

        # setup
        for i, (v_i_from, v_i_to) in enumerate(zip(Q_from, Q_to)):
            self.occupied_now[v_i_from] = i
            if v_i_to != self.NIL_COORD:
                #  check vertex collision
                if self.occupied_nxt[v_i_to] != self.NIL:
                    flg_success = False
                    break
                # check edge collision
                j = self.occupied_now[v_i_to]
                if j != self.NIL and j != i and Q_to[j] == v_i_from:
                    flg_success = False
                    break
                self.occupied_nxt[v_i_to] = i

        # perform PIBT
        if flg_success:
            for i in order:
                if Q_to[i] == self.NIL_COORD:
                    flg_success = self.funcPIBT(
                        Q_from, Q_to, i, history
                    )
                    if not flg_success:
                        break

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            if q_to != self.NIL_COORD:
                self.occupied_nxt[q_to] = self.NIL

        return flg_success
