"""
Krahasimi i algoritmeve (analiza DAA):

  1. Random Search    - plane udhetimi krejt te rastit (pa asnje strategji)
  2. Greedy           - Beam Search me gjeresi 1 (merr gjithmone vetem me te mirin)
  3. Beam Search W=4  - mban 4 rruget me te mira ne cdo nivel
  4. Beam Search W=8  - mban 8 rruget me te mira (algoritmi yne kryesor)

Per secilin masim: hypervolume (cilesia), numrin e zgjidhjeve dhe kohen.
"""
import random
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import problem
import solver


def random_search(n=20000, seed=0):
    """Gjeneron n vektore krejt te rastit dhe mban ata qe jane te pranueshem."""
    rng = random.Random(seed)
    tours = []
    for _ in range(n):
        x = [rng.uniform(problem.LB[i], problem.UB[i]) for i in range(27)]
        x += [float(s) for s in rng.sample(range(7), 7)]     # radhe e rastit e planeteve
        dv, T, eq, ineq = problem.fitness(x)
        if eq == 0 and ineq <= 0:
            tours.append((dv, T, x))
    return tours


def _beam(args):
    tcap, width = args
    return solver.beam_search(tcap, width=width)


def run_beam(width, tcaps):
    """Ekzekuton Beam Search me gjeresine `width` per te gjithe kufijte kohore `tcaps`."""
    with Pool() as pool:
        results = pool.map(_beam, [(t, width) for t in tcaps])
    return [t for r in results for t in r]


if __name__ == "__main__":
    t = random_search(n=2000)
    print("random:", len(t), "tour-e")