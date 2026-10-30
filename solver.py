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

def beam_search(tcap, width=8, seed=0):
    """Kthen listen e tour-eve te plota: (DV, T, x)."""
    # niveli 1: nje tour i pjesshem per cdo planet te pare
    beam = []                               # elementet: (DV, x, state, renditja)
    for p in range(7):
        dv, x, st = optimize_entry(p, seed + p)
        beam.append((dv, x, st, [p]))
    # nivelet 2..7: shtojme nga nje planet
    for level in range(6):
        children = []
        for i, (dv, x, st, seq) in enumerate(beam):
            for p in range(7):
                if p in seq:
                    continue
                d, z, st2 = optimize_leg(st, p, tcap, seed + 100 * level + 10 * i + p)
                if st2 is not None:
                    children.append((dv + d, x + z, st2, seq + [p]))
        children.sort(key=lambda c: c[0])
        beam = children[:width]             # ruajme vetem me te mirat
    tours = []
    for dv, x, st, seq in beam:
        full = x + [float(s) for s in seq]
        f = P.fitness(full)
        if f[2] == 0 and f[3] <= 0:
            tours.append((f[0], f[1], full))
    return tours