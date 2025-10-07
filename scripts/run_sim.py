import argparse
import sys

import numpy as np
import yaml

from control.greedy import plan_charging_greedy, plan_greedy
from sim.core import Sim, SimConfig
from sim.demand import effective_lambda


def main(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    N = int(cfg["network"]["n_stations"])
    C = np.full(N, int(cfg["network"]["capacity_default"]))
    tmin = np.array(cfg["network"]["travel_time_min"], dtype=float).reshape(N, N)
    km   = np.array(cfg["network"]["distance_km"], dtype=float).reshape(N, N)

    energy = cfg.get("energy", {})
    chargers    = np.array(energy.get("chargers_per_station", [2]*N), dtype=int)
    charge_rate = np.array(energy.get("charge_rate_soc_per_hour", [0.25]*N), dtype=float)
    battery_kwh = float(energy.get("battery_kwh_per_vehicle", 0.5))
    energy_cost = float(energy.get("energy_cost_per_kwh_eur", 0.20))

    simcfg = SimConfig(
        dt_min=int(cfg["time"]["dt_minutes"]),
        horizon_h=int(cfg["time"]["horizon_hours"]),
        capacity=C, travel_min=tmin, charge_rate=charge_rate,
        cost_km=km, chargers=chargers,
        battery_kwh=battery_kwh, energy_cost_per_kwh=energy_cost,
    )
    sim = Sim(simcfg, np.random.default_rng(int(cfg.get("seed", 42))))

    base_lambda = np.full(N, float(cfg["demand"]["base_lambda_per_dt"]))
    P = np.full((N, N), 1.0 / N, dtype=float)  # placeholder OD

    steps = int(simcfg.horizon_h * 60 / simcfg.dt_min)

    if steps <= 0:
        print({"error": "No steps to run", "horizon_h": simcfg.horizon_h, "dt_min": simcfg.dt_min}, flush=True)
        return

    total_reloc_km = 0.0
    try:
        for step in range(steps):
            hour = (step * simcfg.dt_min / 60.0) % 24
            lam_t = effective_lambda(base_lambda, hour, weather_fac=1.0, event_fac_vec=None)
            reloc = plan_greedy(sim.x, simcfg.capacity, simcfg.travel_min, low=0.2, high=0.8)
            charge_plan = plan_charging_greedy(sim.x, sim.s, simcfg.chargers, lam_t)

            sim.step(lam_t, P, weather_fac=1.0, event_fac=None, reloc_plan=reloc, charging_plan=charge_plan)

            total_reloc_km += sim.logs[-1]["reloc_km"]
            # lightweight progress every simulated 6 hours
            if step % max(1, int((6*60)/simcfg.dt_min)) == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
    except Exception as e:
        print("\nRuntime error:", repr(e), file=sys.stderr, flush=True)
        raise

    unmet_total = int(sum(r["unmet"] for r in sim.logs))
    avail_avg   = float(np.mean([r["availability"] for r in sim.logs]))
    energy_kwh  = float(sum(r.get("charge_energy_kwh", 0.0) for r in sim.logs))
    energy_eur  = float(sum(r.get("charge_cost_eur", 0.0) for r in sim.logs))

    print("\n", flush=True)
    print({
        "unmet_total": unmet_total,
        "availability_avg": round(avail_avg, 3),
        "relocation_km_total": float(round(total_reloc_km, 2)),
        "charging_energy_kwh_total": round(energy_kwh, 2),
        "charging_cost_eur_total": round(energy_eur, 2),
        "ticks": steps, "dt_min": simcfg.dt_min, "stations": N,
    }, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
