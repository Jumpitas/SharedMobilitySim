import numpy as np


def plan_greedy(x, C, travel_min, low=0.2, high=0.8, max_moves=50):
    x = x.astype(float).copy()
    need = np.where(x < low*C)[0]
    have = np.where(x > high*C)[0]
    plan = []
    for i in need:
        donors = sorted(have, key=lambda j: travel_min[j, i])
        for j in donors:
            surplus = x[j] - high*C[j]
            gap = low*C[i] - x[i]
            k = int(max(0, min(surplus, gap)))
            if k > 0:
                plan.append((j, i, k))
                x[j] -= k
                x[i] += k
                if len(plan) >= max_moves:
                    return plan
    return plan

# simple strategic charging (prioritize high demand & low SoC)
def plan_charging_greedy(x, s, chargers, lam_t):
    """
    Return per-station number of vehicles to plug this tick.
    Prioritizes stations with high expected demand and low SoC.
    Local constraint only: plan[i] <= min(chargers[i], x[i]).
    """
    score = lam_t * (1.0 - s)               # higher = more urgent to charge
    order = np.argsort(-score)
    plan = np.zeros_like(x, dtype=int)
    for i in order:
        plan[i] = int(min(chargers[i], x[i]))
    return plan
