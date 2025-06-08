
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def amplitude_closed_form(m: float, c: float, k: float, F0: float, omega: np.ndarray) -> np.ndarray:
    denom = np.sqrt((k - m * omega ** 2) ** 2 + (c * omega) ** 2)
    return F0 / denom


def zeta(m: float, c: float, k: float) -> float:
    return c / (2.0 * np.sqrt(k * m))


def sdof_rhs(t: float, y: Tuple[float, float], m: float, c: float, k: float, F0: float, omega: float):
    x, v = y
    force = F0 * np.sin(omega * t)
    dxdt = v
    dvdt = (force - c * v - k * x) / m
    return [dxdt, dvdt]


def simulate_amplitude(m: float, c: float, k: float, F0: float, omega: float,
                       t_end: float = 200.0, discard_ratio: float = 0.8) -> float:
    """Measures the steady-state amplitude using time-domain simulation.

    discard_ratio: Used to discard the transient part of the simulation.
    """
    t_eval = np.linspace(0.0, t_end, 5000)
    sol = solve_ivp(
        fun=sdof_rhs,
        t_span=(0.0, t_end),
        y0=[0.0, 0.0],
        args=(m, c, k, F0, omega),
        t_eval=t_eval,
        method="RK45",
    )
    x = sol.y[0]
    # Discard transient
    start_idx = int(len(t_eval) * discard_ratio)
    x_ss = x[start_idx:]
    return np.max(np.abs(x_ss))


# -------------------------------------------------
# Main Flow
# -------------------------------------------------

def main():
    # System parameters
    m = 1.0
    k = 20.0
    F0 = 5.0
    omega_n = np.sqrt(k / m)

    # Damping cases to investigate
    damping_cases = {
        "c = 0.1": 0.1,
        "c = 0.5": 0.5,
        "c = 5.0": 5.0,
        "c = 15.0": 15.0,
    }

    # Frequency sweep (for analytical curve)
    omega_range = np.linspace(0.01, 3 * omega_n, 2000)

    # Plot: Analytical + Simulation Points
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]
    markers = ["o", "s", "^", "D"]

    # Selected frequencies for simulation
    freq_samples = np.array([0.5, 0.9, 1.0, 1.1, 1.5])

    for (label, c_val), color, mk in zip(damping_cases.items(), colors, markers):
        # Analytical
        A = amplitude_closed_form(m, c_val, k, F0, omega_range)
        ax.plot(omega_range / omega_n, A, color=color, lw=1.2, label=f"{label} (Analytical)")

        # Simulation points
        sim_amps = []
        for r in freq_samples:
            amp = simulate_amplitude(m, c_val, k, F0, r * omega_n)
            sim_amps.append(amp)
        ax.scatter(freq_samples, sim_amps, color=color, marker=mk, s=35, edgecolors="k",
                   label=f"{label} (Sim)")

    ax.set_xlabel(r"Relative frequency $\omega/\omega_n$")
    ax.set_ylabel(r"Amplitude $A$ (m)")
    ax.set_title("Analytical Curve vs. Time-Domain Simulation")
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize="small", ncol=2)

    os.makedirs("figs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("figs/freq_response_validation.png", dpi=300)

    # -------------------------------------------------
    # Q vs. c Plot
    # -------------------------------------------------
    c_vals = np.logspace(-2, 2, 200)
    zetas = zeta(m, c_vals, k)
    Qs = 1.0 / (2.0 * zetas)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.loglog(c_vals, Qs, "k-")
    ax2.set_xlabel(r"Damping coefficient $c$ (N·s/m)")
    ax2.set_ylabel(r"Quality factor $Q = 1/(2\zeta)$")
    ax2.set_title("Q Factor – Damping Coefficient Relationship (log-log)")
    ax2.grid(True, which="both", ls=":", lw=0.5)

    plt.tight_layout()
    plt.savefig("figs/Q_vs_c.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
