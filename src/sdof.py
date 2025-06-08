
from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "rhs",
    "amplitude_closed_form",
    "zeta",
    "simulate_amplitude",
    "simulate_full",
]

# --------------------------------------------
# 1. ODE Right-Hand Side
# --------------------------------------------

def rhs(t: float, y: Tuple[float, float], m: float, c: float, k: float, F0: float, omega: float):
    """Mass-Spring-Damper System ODE Right-Hand Sides.

    Parameters
    ----------
    t : float
        Time (s)
    y : Tuple[x, v]
        Position and velocity
    m, c, k : float
        Mass (kg), viscous damping (N·s/m), spring constant (N/m)
    F0, omega : float
        Force amplitude (N) and angular frequency (rad/s)
    """
    x, v = y
    force = F0 * np.sin(omega * t)
    dxdt = v
    dvdt = (force - c * v - k * x) / m
    return [dxdt, dvdt]

# --------------------------------------------
# 2. Analytical Amplitude
# --------------------------------------------

def amplitude_closed_form(m: float, c: float, k: float, F0: float, omega: np.ndarray) -> np.ndarray:
    """Steady-State Amplitude A(ω) (absolute)."""
    denom = np.sqrt((k - m * omega ** 2) ** 2 + (c * omega) ** 2)
    return F0 / denom

# --------------------------------------------
# 3. Damping Ratio ζ and Quality Factor Q
# --------------------------------------------

def zeta(m: float, c: float, k: float) -> float:
    """ζ = c / (2 * sqrt(k m))"""
    return c / (2.0 * np.sqrt(k * m))

# --------------------------------------------
# 4. Simulation Utilities
# --------------------------------------------

def simulate_full(m: float, c: float, k: float, F0: float, omega: float,
                  y0=(0.0, 0.0), t_end: float = 60.0, num_points: int = 4000):
    """Returns full time series (t, x, v)."""
    t_eval = np.linspace(0.0, t_end, num_points)
    sol = solve_ivp(rhs, (0.0, t_end), y0, args=(m, c, k, F0, omega), t_eval=t_eval)
    return sol.t, sol.y[0], sol.y[1]


def simulate_amplitude(m: float, c: float, k: float, F0: float, omega: float,
                       t_end: float = 60.0, discard_ratio: float = 0.7) -> float:
    """Returns steady-state peak amplitude (|x|_max)."""
    t, x, _ = simulate_full(m, c, k, F0, omega, t_end=t_end)
    start_idx = int(len(t) * discard_ratio)
    x_ss = x[start_idx:]
    return np.max(np.abs(x_ss))
