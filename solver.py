"""
Algoritmi per Trappist Tour:
  1. Differential Evolution - optimizon 4 parametrat e cdo udhetimi
  2. Beam Search - zgjedh renditjen e planeteve
  3. Fronti Pareto + hypervolume
"""
import json
import math
from scipy.optimize import differential_evolution

import problem as P

REF = (2500.0, 4000.0)      # pika referuese e ESA-s [m/s, dite]
LAM = 0.5                   # sa "kushton" nje dite kohe [m/s per dite]


def optimize_entry(planet, seed):
    """Hapi hyres (nga 10 AU te planeti i pare) me DV minimale.
    T0 kerkohet rreth 1500 dite, kur hyrja eshte pothuajse falas."""
    def f(z):
        dv, viol, _ = P.entry_leg(z[0], z[1], z[2], planet)
        return dv + (1e7 if viol > 0 else 0)  # ndeshkim nese i afrohet yllit
    res = differential_evolution(f, [(0, 1), (0.25, 0.75), (1400, 1600)],
                                 maxiter=60, popsize=10, seed=seed, polish=False, tol=0)
    dv, viol, st = P.entry_leg(*res.x, planet)
    return dv, list(res.x), st


def optimize_leg(state, planet, tcap, seed):
    """Gjen 4 parametrat [beta, rp, eta, T] te udhetimit te radhes, me T <= tcap."""
    def f(z):
        dv, viol, _ = P.next_leg(state, z[0], z[1], z[2], z[3], planet)
        return dv + LAM * z[3] + (1e7 if viol > 0 else 0)
    bounds = list(zip(P.LEG_LB, P.LEG_UB))
    bounds[3] = (5.0, tcap)  # kufiri i kohes per kete udhetim
    res = differential_evolution(f, bounds, maxiter=60, popsize=10, seed=seed, polish=False, tol=0)
    dv, viol, st = P.next_leg(state, *res.x, planet)
    if viol > 0:
        return math.inf, list(res.x), None
    return dv, list(res.x), st