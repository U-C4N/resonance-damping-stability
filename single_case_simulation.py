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
    Defines the differential equations of the forced and damped mass-spring-damper system.

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
c = 0.5       # Damping coefficient (N·s/m)
F0 = 5.0      # Forcing amplitude (N)
omega = 3.0   # Forcing angular frequency (rad/s)

# 3. Define Simulation Conditions

# Initial conditions: System starts at rest and at equilibrium position.

x0 = 0.0      # Initial position (m)
v0 = 0.0      # Initial velocity (m/s)
y0 = [x0, v0] # Initial state vector

# Time interval

t_span = [0, 50]  # Simulation start and end times (seconds)

# Time points for evaluation (for a smoother plot)

t_eval = np.linspace(t_span[0], t_span[1], 1000)

# 4. Solve the Differential Equations

# Solve the system using scipy.integrate.solve_ivp.

# The 'dense_output=True' argument allows us to obtain a continuous solution function.

solution = solve_ivp(
    fun=model,
    t_span=t_span,
    y0=y0,
    args=(m, c, k, F0, omega),
    dense_output=True,
    t_eval=t_eval
)

# Extract time and state variables (position and velocity) from the solution

t = solution.t
x, v = solution.y

# 5. Visualize the Results

# Create a window with two vertical subplots

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
fig.suptitle('Forced and Damped Mass-Spring-Damper System', fontsize=16)

# Top subplot: Position - Time

ax1.plot(t, x, 'b-')
ax1.set_title('Position vs. Time')
ax1.set_ylabel('Position (m)')
ax1.grid(True)

# Bottom subplot: Velocity - Time

ax2.plot(t, v, 'r-')
ax2.set_title('Velocity vs. Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.grid(True)

# Ensure the plot looks neat

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot

plt.show()
