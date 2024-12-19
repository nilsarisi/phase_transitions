import numpy as np
from scipy.signal import convolve2d
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython=True)
def metropolis_step(grid, T, J, h, neighbor_sum):
    size = grid.shape[0]
    for _ in range(size**2):
        i, j = np.random.randint(0, size, 2)
        spin = grid[i, j]
        delta_E = 2 * J * spin * neighbor_sum[i, j] + 2 * h * spin
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
            grid[i, j] = -spin
    return grid

def simulate_with_snapshots(grid_size, J, h, temps, steps_per_temp, average_last_n, snapshot_temps):
    grid = np.random.choice([-1, 1], size=(grid_size, grid_size))
    magnetizations = []
    snapshots = {}
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # Map each snapshot temperature to the closest temperature in the `temps` array
    snapshot_map = {T: temps[np.argmin(np.abs(temps - T))] for T in snapshot_temps}

    for T in temps:
        mags = []
        for step in range(steps_per_temp):
            neighbor_sum = convolve2d(grid, kernel, mode='same', boundary='wrap')
            grid = metropolis_step(grid, T, J, h, neighbor_sum)
            mags.append(np.sum(grid) / grid.size)

            # Save a snapshot if the temperature matches a snapshot-mapped temperature
            if T in snapshot_map.values() and step == steps_per_temp - 1:
                snapshots[T] = grid.copy()

        # Average over the last `average_last_n` steps
        avg_mag = np.mean(mags[-average_last_n:])
        magnetizations.append(avg_mag)

    return magnetizations, snapshots, snapshot_map

def calculate_critical_temperature(temps, magnetizations):
    magnetizations = np.array(magnetizations)
    temp_diff = np.diff(temps)
    mag_diff = np.diff(magnetizations)
    slopes = mag_diff / temp_diff
    critical_index = np.argmax(np.abs(slopes))
    critical_temp = temps[critical_index]
    return critical_temp

# Example Usage
grid_size = 50
J = 1
h = 0.1
temps = np.linspace(1, 5, 50)
steps_per_temp = 10000
average_last_n = 2000
snapshot_temps = [1.49, 1.98, 2.31, 3.53, 4.02]  # Temperatures for snapshots

# Run the simulation
magnetizations, snapshots, snapshot_map = simulate_with_snapshots(
    grid_size, J, h, temps, steps_per_temp, average_last_n, snapshot_temps
)

# Calculate the critical temperature
critical_temp = calculate_critical_temperature(temps, magnetizations)

# Plot Results
plt.figure(figsize=(8, 6))
plt.plot(temps, magnetizations, marker="o", label="Magnetization")
plt.axvline(critical_temp, color='red', linestyle='--', label=f"Critical T = {critical_temp:.3f}")
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetization")
plt.title("Magnetization vs Temperature with Critical Temperature")
plt.legend()
plt.grid()
plt.show()

# Plot Snapshots
fig, axes = plt.subplots(1, len(snapshot_temps), figsize=(15, 5))
for ax, T in zip(axes, snapshot_temps):
    closest_T = snapshot_map[T]
    if closest_T in snapshots:
        ax.imshow(snapshots[closest_T], cmap="coolwarm", interpolation="none")
        ax.set_title(f"T = {closest_T:.2f} (Closest to {T:.2f})")
        ax.axis("off")
    else:
        ax.set_title(f"No snapshot for T = {T:.2f}")
        ax.axis("off")
plt.suptitle("Snapshots of Spin Configurations at Closest Temperatures")
plt.tight_layout()
plt.show()
