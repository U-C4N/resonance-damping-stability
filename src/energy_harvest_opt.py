import os
import csv
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------------------------------
# Parameters
# -------------------------------------------------
M = 1.0        # kg
K = 20.0       # N/m
F0 = 5.0       # N
ETA = 0.9      # energy conversion efficiency (0..1)
OMEGA = np.sqrt(K / M)  # resonance case (worst case for energy)

X_LIM = 0.5    # safety amplitude limit (m)

# Scan range
C_MIN = 1e-2   # 0.01 N·s/m
C_MAX = 1e+2   # 100  N·s/m
N_POINTS = 60  # resolution (higher => finer scan)

T_END = 60.0       # simulation time (s)
DISCARD_RATIO = 0.7  # transient portion (first %70 discarded)

DATA_DIR = "data"
FIG_DIR = "figs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------------------------------
# Model functions
# -------------------------------------------------

def sdof_rhs(t: float, y: Tuple[float, float], m: float, c: float, k: float, F0: float, omega: float):
    x, v = y
    force = F0 * np.sin(omega * t)
    dxdt = v
    dvdt = (force - c * v - k * x) / m
    return [dxdt, dvdt]


def simulate(m: float, c: float, k: float, F0: float, omega: float) -> Tuple[float, float]:
    """Returns average power and max |x| for given c.

    Tuple: (avg_power, max_abs_x)
    """
    t_eval = np.linspace(0.0, T_END, 6000)
    sol = solve_ivp(
        fun=sdof_rhs,
        t_span=(0.0, T_END),
        y0=[0.0, 0.0],
        args=(m, c, k, F0, omega),
        t_eval=t_eval,
        method="RK45",
    )

    x, v = sol.y

    # Transient after
    start_idx = int(len(t_eval) * DISCARD_RATIO)
    v_ss = v[start_idx:]
    x_ss = x[start_idx:]

    # Average power (steady-state)
    avg_power = ETA * c * np.mean(v_ss ** 2)

    # Max. absolute displacement (entire simulation)
    max_abs_x = np.max(np.abs(x))

    return avg_power, max_abs_x

# -------------------------------------------------
# Main flow
# -------------------------------------------------

def main():
    c_values = np.logspace(np.log10(C_MIN), np.log10(C_MAX), N_POINTS)

    rows = []  # csv rows

    best_power = -np.inf
    c_opt = None
    best_row = None

    print("[INFO] Scanning…", N_POINTS, "points")

    for c_val in c_values:
        avg_p, max_x = simulate(M, c_val, K, F0, OMEGA)
        status = "SAFE" if max_x <= X_LIM else "UNSAFE"

        rows.append({
            "c": c_val,
            "avg_power": avg_p,
            "max_abs_x": max_x,
            "status": status,
        })

        if status == "SAFE" and avg_p > best_power:
            best_power = avg_p
            c_opt = c_val
            best_row = rows[-1]

    # ---------------- CSV Save ----------------
    csv_path = os.path.join(DATA_DIR, "energy_harvest_summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["c", "avg_power", "max_abs_x", "status"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Results saved to {csv_path}.")

    # ---------------- Plot ----------------
    c_arr = np.array([r["c"] for r in rows])
    p_arr = np.array([r["avg_power"] for r in rows])
    safe_mask = np.array([r["status"] == "SAFE" for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(c_arr[safe_mask], p_arr[safe_mask], "o-g", label="SAFE (|x| ≤ %.2f m)" % X_LIM)
    ax.loglog(c_arr[~safe_mask], p_arr[~safe_mask], "x-r", label="UNSAFE")

    if c_opt is not None:
        ax.axvline(c_opt, color="k", ls="--", label="c_opt ≈ %.3g" % c_opt)
        ax.annotate("c_opt", xy=(c_opt, best_power), xytext=(1.1*c_opt, 1.2*best_power),
                    arrowprops=dict(arrowstyle="->"), fontsize="small")

    ax.set_xlabel("Damping coefficient c (N·s/m)")
    ax.set_ylabel(r"Average harvested power $\overline{P}$ (W)")
    ax.set_title("Energy Harvesting – Safety Constrained Optimization")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize="small")

    fig_path = os.path.join(FIG_DIR, "energy_harvest_opt.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.show()

    if best_row:
        print("\n[RESULT] Best point with safe amplitude:")
        for k, v in best_row.items():
            print(f"  {k:>12}: {v}")
    else:
        print("[WARN] No safe point found – increase X_LIM or decrease F0.")

if __name__ == "__main__":
    main()

