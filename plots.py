"""
Grafiket e rezultateve (me matplotlib).
  plot_pareto(front) - zgjidhjet: karburanti kundrejt kohes
  plot_tour(x)       - radha e planeteve qe viziton anija
"""
import math
import matplotlib.pyplot as plt
import problem
import solver

AU = 149597870691.0          # 1 njesi astronomike ne metra


def plot_pareto(front):
    # marrim vetem zgjidhjet brenda pikes referuese [2500, 4000]
    karburanti = [dv for dv, T, x in front if dv < 2500 and T < 4000]
    koha = [T for dv, T, x in front if dv < 2500 and T < 4000]

    plt.figure(figsize=(8, 5))
    plt.scatter(karburanti, koha, color="blue", label="zgjidhjet")
    plt.scatter(2500, 4000, color="black", marker="x", label="pika referuese")
    plt.xlabel("Karburanti ΔV [m/s]")
    plt.ylabel("Koha e misionit [dite]")
    plt.title("Fronti Pareto, hypervolume = " + str(round(solver.hypervolume(front))))
    plt.legend()
    plt.grid(True)
    plt.savefig("pareto_front.png")
    plt.show()