import numpy as np

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

# 1. Define the System Model

# Physical system equation: m * d²x/dt² + c * dx/dt + k * x = F(t)

# Here, F(t) is assumed as F0 * sin(omega * t).

# This second-order equation is converted into a system of two first-order differential equations.

# y[0] = x (position)

# y[1] = v (velocity)

#

# dy[0]/dt = v = y[1]

# dy[1]/dt = (1/m) * (F(t) - c*v - k*x)

def model(t, y, m, c, k, F0, omega):
    """
    Defines the differential equations of a forced and damped mass-spring-damper system.

    Parameters:
    t: Time
    y: [position, velocity] state vector
    m, c, k: Mass, damping coefficient, spring constant
    F0, omega: Forcing amplitude and angular frequency

    Returns:
    A list containing [dx/dt, dv/dt] derivatives.
    """
    x, v = y
    force = F0 * np.sin(omega * t)
    dxdt = v
    dvdt = (force - c * v - k * x) / m
    return [dxdt, dvdt]

# 2. Set System Parameters

m = 1.0       # Mass (kg)
k = 20.0      # Spring constant (N/m)
F0 = 5.0      # Forcing amplitude (N)
# Resonance frequency: omega_n = sqrt(k/m) = sqrt(20) ~= 4.47 rad/s
omega = 4.47  # Forcing angular frequency set to resonance frequency

# Damping coefficients to compare
damping_cases = {
    "Low Damping (c=0.1)": 0.1,
    "Medium Damping (c=0.5)": 0.5,
    "High Damping (c=15.0)": 15.0
}

# 3. Define Simulation Conditions

# Initial conditions: System starts at rest and at equilibrium position.
y0 = [0.0, 0.0]  # Initial state vector [x0, v0]

# Time interval
t_span = [0, 50]  # Simulation start and end times (seconds)

# Time points for evaluation (for smoother plot)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Dictionary to store results
solutions = {}

# 4. Run Simulations for Different Damping Cases

for case_name, c_val in damping_cases.items():
    sol = solve_ivp(
        fun=model,
        t_span=t_span,
        y0=y0,
        args=(m, c_val, k, F0, omega),
        dense_output=True,
        t_eval=t_eval
    )
    solutions[case_name] = sol

# 5. Visualize Results

# Create a window with two vertical subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f'Effect of Different Damping Coefficients (Resonance Case ω={omega} rad/s)', fontsize=16)

# Colors and styles
colors = ['r', 'g', 'b']
styles = ['-', '--', ':']
case_names = list(damping_cases.keys())

# Top subplot: Position vs. Time
for i, case_name in enumerate(case_names):
    t = solutions[case_name].t
    x = solutions[case_name].y[0]
    ax1.plot(t, x, label=case_name, color=colors[i], linestyle=styles[i])
ax1.set_title('Position vs. Time')
ax1.set_ylabel('Position (m)')
ax1.legend()
ax1.grid(True)

# Bottom subplot: Velocity vs. Time
for i, case_name in enumerate(case_names):
    t = solutions[case_name].t
    v = solutions[case_name].y[1]
    ax2.plot(t, v, label=case_name, color=colors[i], linestyle=styles[i])
ax2.set_title('Velocity vs. Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.legend()
ax2.grid(True)

# Ensure the plot looks neat
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()
