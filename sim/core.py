from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

Move = Tuple[int, int, int]   # (i -> j, k units)

@dataclass
class SimConfig:
    """Minimal config for a docked shared-mobility sim."""
    dt_min: int                         # minutes per tick
    horizon_h: int                      # total hours to simulate
    capacity: np.ndarray                # (N,) max vehicles per station
    travel_min: np.ndarray              # (N,N) travel time matrix (minutes)
    charge_rate: np.ndarray             # (N,) SoC/hour when plugged (e.g. 0.25 => +25%/h)
    cost_km: np.ndarray                 # (N,N) relocation distance or cost
    chargers: np.ndarray                # (N,) plugs per station
    battery_kwh: float                  # kWh per vehicle @ 100% SoC
    energy_cost_per_kwh: float          # €/kWh

class Sim:
    """
    Discrete-time simulator for station-based e-scooters/e-bikes.
    State per station i: stock x[i], average SoC s[i].
    Trips consume SoC on departure; charging raises average SoC for plugged fraction.
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

        N = cfg.capacity.shape[0]
        assert cfg.travel_min.shape == (N, N), "travel_min must be NxN"
        assert cfg.cost_km.shape == (N, N), "cost_km must be NxN"
        for arr, name in ((cfg.capacity, "capacity"),
                          (cfg.charge_rate, "charge_rate"),
                          (cfg.chargers, "chargers")):
            assert arr.shape == (N,), f"{name} must be shape (N,)"

        # State
        self.t = 0  # minutes since start
        self.x = np.minimum(cfg.capacity // 3, cfg.capacity).astype(int)  # initial stock ~1/3 full
        self.s = rng.uniform(0.5, 0.9, size=N)                             # avg SoC in [50%, 90%]
        self._trip_queue: List[Tuple[int, int, float]] = []                # (t_arrive, dest_j, soc_use)

        # Accumulators / logs
        self.logs: List[dict] = []

    # --------------------------------------------------

    def step(
        self,
        lam_t: np.ndarray,
        P_t: np.ndarray,
        *,
        weather_fac: float = 1.0,
        event_fac: Optional[np.ndarray] = None,
        reloc_plan: Optional[List[Move]] = None,
        charging_plan: Optional[np.ndarray] = None,
    ) -> None:
        """Advance simulation by one tick."""
        self._validate_inputs(lam_t, P_t, event_fac, charging_plan)

        dt_h = self.cfg.dt_min / 60.0
        N = self.x.shape[0]

        # 1) Demand arrivals and service
        lam_eff = lam_t * weather_fac
        if event_fac is not None:
            lam_eff = lam_eff * event_fac
        A = self.rng.poisson(lam_eff)                 # arrivals per station (int), depois de fatores aplica-se o poisson
        served = np.minimum(self.x, A)
        unmet = A - served
        self.x -= served

        # 2) Spawn trips (schedule arrivals)
        if served.sum():
            for i, k in enumerate(served):
                if k <= 0:
                    continue
                # NOTA, assume que P_t[i] é um valid probability vector
                dests = self.rng.choice(N, size=int(k), p=P_t[i])
                # draw per-trip SoC usage (tempo de viagem ainda nao importa, adicionar depois)
                uses = self.rng.uniform(0.03, 0.08, size=int(k))
                tij = self.cfg.travel_min[i, dests].astype(int)
                self._trip_queue.extend((self.t + int(tt), int(j), float(u)) for tt, j, u in zip(tij, dests, uses))

        # 3) Complete trips due by next tick
        t_next = self.t + self.cfg.dt_min
        if self._trip_queue:
            due, future = [], []
            for rec in self._trip_queue:
                (due if rec[0] <= t_next else future).append(rec)
            self._trip_queue = future
            for _, j, soc_use in due:
                # add vehicle if not at capacity (se estiver em capacity perde-se a scooter); always apply SoC drop on avg
                if self.x[j] < self.cfg.capacity[j]:
                    self.x[j] += 1
                self.s[j] = max(0.0, self.s[j] - soc_use)

        # 4) Charging (operator decision)
        plan = self._resolve_charging_plan(charging_plan)
        plug_frac = np.divide(plan, np.maximum(self.x, 1), where=self.x > 0, out=np.zeros_like(self.s, dtype=float))
        soc_gain = self.cfg.charge_rate * dt_h * plug_frac
        self.s = np.minimum(1.0, self.s + soc_gain)

        # Energy & cost from charging (Mantemos medias e aproxim por razão de simplificar)
        energy_kwh = float(np.sum(plan * self.cfg.battery_kwh * (self.cfg.charge_rate * dt_h)))
        charge_cost = energy_kwh * self.cfg.energy_cost_per_kwh

        # 5) Apply relocation plan (instantâneo; Adicionar maybe tempo de deslocação dos camiões)
        reloc_km = 0.0
        if reloc_plan:
            for i, j, k in reloc_plan:
                if k <= 0:
                    continue
                move = int(min(k, self.x[i], self.cfg.capacity[j] - self.x[j]))
                if move <= 0:
                    continue
                self.x[i] -= move
                self.x[j] += move
                reloc_km += move * self.cfg.cost_km[i, j]

        # 6) Manter Capacity e SoC feasible, logs
        self.x = np.clip(self.x, 0, self.cfg.capacity)
        self.s = np.clip(self.s, 0.0, 1.0)

        self.logs.append({
            "t_min": self.t,
            "unmet": int(unmet.sum()),
            "availability": float((self.x > 0).mean()),
            "reloc_km": reloc_km,
            "plugged": int(plan.sum()),
            "charge_energy_kwh": energy_kwh,
            "charge_cost_eur": charge_cost,
        })
        self.t = t_next

    # ------------------------- helpers -------------------------

    def _resolve_charging_plan(self, charging_plan: Optional[np.ndarray]) -> np.ndarray:
        """Clip to [0, chargers, x]. If None, default: plug as many as possible."""
        if charging_plan is None:
            return np.minimum(self.cfg.chargers, self.x).astype(int)
        plan = np.asarray(charging_plan, dtype=int).copy()
        plan = np.clip(plan, 0, None)
        plan = np.minimum(plan, self.cfg.chargers)
        plan = np.minimum(plan, self.x)
        return plan

    def _validate_inputs(
        self,
        lam_t: np.ndarray,
        P_t: np.ndarray,
        event_fac: Optional[np.ndarray],
        charging_plan: Optional[np.ndarray],
    ) -> None:
        N = self.x.shape[0]
        assert lam_t.shape == (N,), f"lam_t must be shape (N,), got {lam_t.shape}"
        assert P_t.shape == (N, N), f"P_t must be shape (N,N), got {P_t.shape}"
        # Não sou maluco: rows of P sum to ~1
        row_sum = P_t.sum(axis=1)
        if not np.allclose(row_sum, 1.0, atol=1e-6):
            raise ValueError("Each row of P_t must sum to 1.0")
        if event_fac is not None and event_fac.shape != (N,):
            raise ValueError("event_fac must be shape (N,) or None")
        if charging_plan is not None and charging_plan.shape != (N,):
            raise ValueError("charging_plan must be shape (N,) or None")
