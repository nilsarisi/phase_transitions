import numpy as np
import matplotlib.pyplot as plt

# Grid size (NxN)
N = 50  # You can adjust this size if needed

# Step 1: Generate a Random Grid (-1 or +1 values)
random_grid = np.random.choice([-1, 1], size=(N, N))

# Display the Random Grid
plt.figure(figsize=(6, 6))
plt.matshow(random_grid, cmap='bwr', vmin=-1, vmax=1, fignum=1)
plt.colorbar(label='Spin')
plt.title('Random Grid (Initial State)')
plt.show()

# Function to calculate the local energy at a given grid position (i, j)
def calculate_energy(grid, i, j):
    """Compute the local energy for site (i, j)."""
    neighbors_sum = (
        grid[(i - 1) % N, j] +  # Above neighbor
        grid[(i + 1) % N, j] +  # Below neighbor
        grid[i, (j - 1) % N] +  # Left neighbor
        grid[i, (j + 1) % N]    # Right neighbor
    )
    return -grid[i, j] * neighbors_sum

# Monte Carlo step to evolve the system
def monte_carlo_step(grid, temperature):
    """Perform a single Monte Carlo step."""
    i, j = np.random.randint(0, N, size=2)
    dE = -2 * calculate_energy(grid, i, j)
    if dE < 0 or np.random.rand() < np.exp(-dE / temperature):
        grid[i, j] *= -1  # Flip the spin

# Parameters for Monte Carlo simulation
temperature = 0.5  # Controls the randomness of spin flips
iterations = 50000  # Total number of Monte Carlo steps
interval = 1000  # Interval for recording magnetization

# Initialize the grid with a copy of the random grid
evolved_grid = random_grid.copy()
magnetizations = []  # To store magnetization values
steps = []  # To store iteration steps

# Perform the Monte Carlo simulation
for step in range(iterations):
    monte_carlo_step(evolved_grid, temperature)
    
    # Record magnetization at specified intervals
    if step % interval == 0:
        M = np.sum(evolved_grid) / (N * N)  # Compute magnetization
        magnetizations.append(M)
        steps.append(step)
        print(f"Step {step}: Magnetization = {M:.4f}")

# Plot the Magnetization vs. Steps
plt.figure(figsize=(8, 6))
plt.plot(steps, magnetizations, marker='o', linestyle='-', label='Magnetization')
plt.title('Magnetization vs. Monte Carlo Steps')
plt.xlabel('Monte Carlo Steps')
plt.ylabel('Magnetization')
plt.grid()
plt.legend()
plt.show()

# Display the Evolved Grid
plt.figure(figsize=(6, 6))
plt.matshow(evolved_grid, cmap='bwr', vmin=-1, vmax=1, fignum=2)
plt.colorbar(label='Spin')
plt.title('Evolved Grid (Final State)')
plt.show()
