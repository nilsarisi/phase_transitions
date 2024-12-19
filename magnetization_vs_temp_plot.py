import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(size):
    """Initialize the grid with random spins (+1 or -1)."""
    return np.random.choice([-1, 1], size=(size, size))

def calculate_energy(grid, J):
    """Calculate the total energy of the system."""
    energy = 0
    size = grid.shape[0]
    for i in range(size):
        for j in range(size):
            spin = grid[i, j]
            neighbors = (
                grid[(i + 1) % size, j]
                + grid[i, (j + 1) % size]
                + grid[(i - 1) % size, j]
                + grid[i, (j - 1) % size]
            )
            energy -= J * spin * neighbors
    return energy / 2
def metropolis_step(grid, T, J):
    """Perform one Metropolis step."""
    size = grid.shape[0]
    for _ in range(size**2): 
        i, j = np.random.randint(0, size, 2)
        spin = grid[i, j]
        neighbors = (
            grid[(i + 1) % size, j]
            + grid[i, (j + 1) % size]
            + grid[(i - 1) % size, j]
            + grid[i, (j - 1) % size]
        )
        delta_E = 2 * J * spin * neighbors
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
            grid[i, j] *= -1  # Flip spin
    return grid

def magnetization(grid):
    """Calculate the magnetization of the grid."""
    return np.abs(np.sum(grid)) / grid.size

def simulate(grid_size, J, temps, steps_per_temp):
    """Simulate the Ising model and return magnetization vs. temperature."""
    grid = initialize_grid(grid_size)
    magnetizations = []

    for T in temps:
        for _ in range(steps_per_temp):
            grid = metropolis_step(grid, T, J)
        magnetizations.append(magnetization(grid))
    return magnetizations

grid_size = 50
J = 1  
temps = np.linspace(1, 5, 50) 
steps_per_temp = 100

magnetizations = simulate(grid_size, J, temps, steps_per_temp)
plt.figure(figsize=(8, 6))
plt.plot(temps, magnetizations, marker="o")
plt.title("Magnetization vs Temperature")
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetization")
plt.axvline(x=2.27, color="red", linestyle="--", label="Critical Temp (~2.27 for 2D)")
plt.legend()
plt.grid()
plt.show()