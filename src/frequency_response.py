import os
import numpy as np
import matplotlib.pyplot as plt


def amplitude_closed_form(m: float, c: float, k: float, F0: float, omega: np.ndarray) -> np.ndarray:
    """Returns the amplitude of the steady state response analytically.

    A(ω) = F0 / sqrt( (k - m ω^2)^2 + (c ω)^2 )
    """
    denom = np.sqrt((k - m * omega**2) ** 2 + (c * omega) ** 2)
    return F0 / denom


def zeta(m: float, c: float, k: float) -> float:
    """Damping ratio ζ = c / (2 * sqrt(k m))"""
    return c / (2 * np.sqrt(k * m))


def main():
    # System parameters
    m = 1.0  # kg
    k = 20.0  # N/m
    F0 = 5.0  # N (same value used only scales the plot)

    omega_n = np.sqrt(k / m)  # natural angular frequency

    # Frequency sweep range: 0 .. 3 * omega_n
    omega = np.linspace(0.01, 3 * omega_n, 2000)  # start from 0.01 to avoid division by zero

    # Damping coefficients to analyze
    damping_values = {
        "c = 0.1 (ζ ≈ %.3f)" % zeta(m, 0.1, k): 0.1,
        "c = 0.5 (ζ ≈ %.3f)" % zeta(m, 0.5, k): 0.5,
        "c = 5   (ζ ≈ %.3f)" % zeta(m, 5.0, k): 5.0,
        "c = 15  (ζ ≈ %.3f)" % zeta(m, 15.0, k): 15.0,
    }

    # Color & style cycle
    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]
    linestyles = ["-", "--", "-.", ":"]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, c_val), color, ls in zip(damping_values.items(), colors, linestyles):
        A = amplitude_closed_form(m, c_val, k, F0, omega)
        ax.plot(omega / omega_n, A, linestyle=ls, color=color, label=label)

    ax.set_xlabel(r"Relative frequency $\omega/\omega_n$")
    ax.set_ylabel(r"Amplification factor $A/F_0$ (m/N)")
    ax.set_title("Single-DOF System Frequency Response — Analytical Solution")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize="small")

    # Output directory
    os.makedirs("figs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("figs/frequency_response.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
