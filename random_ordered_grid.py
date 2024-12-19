import numpy as np
import matplotlib.pyplot as plt

# Grid size (NxN)
N = 50  # You can adjust this size if needed

# Step 1: Generate a Random Grid (-1 or +1 values)
# The grid initializes with random values of either -1 or +1 at each position.
random_grid = np.random.choice([-1, 1], size=(N, N))

# Debug: Check a sample of the grid values to ensure proper initialization
print("Random Grid Sample:\n", random_grid[:5, :5])

# Display the Random Grid
# We use `matshow` to properly visualize the 2D grid with the `bwr` colormap.
plt.figure(figsize=(6, 6))
plt.matshow(random_grid, cmap='bwr', vmin=-1, vmax=1, fignum=1)
plt.colorbar(label='Spin')  # Adds a colorbar indicating spin values
plt.title('Random Grid')
plt.show()

# Step 2: Create an Ordered Grid (All +1)
# This grid represents the fully ordered state with all elements set to +1.
ordered_grid = np.ones((N, N))

# Display the Ordered Grid
# We ensure proper display by setting `vmin` and `vmax` for consistency.
plt.figure(figsize=(6, 6))
plt.matshow(ordered_grid, cmap='bwr', vmin=-1, vmax=1, fignum=2)
plt.colorbar(label='Spin')
plt.title('Ordered Grid')
plt.show()

# Function to calculate the local energy at a given grid position (i, j)
def calculate_energy(grid, i, j):
    """Compute the local energy for site (i, j)."""
    # Sum of the spins of the four neighbors (with periodic boundary conditions)
    neighbors_sum = (
        grid[(i - 1) % N, j] +  # Above neighbor
        grid[(i + 1) % N, j] +  # Below neighbor
        grid[i, (j - 1) % N] +  # Left neighbor
        grid[i, (j + 1) % N]    # Right neighbor
    )
    # Energy contribution: -s_ij * sum of neighbor spins
    return -grid[i, j] * neighbors_sum

# Monte Carlo step to evolve the system
def monte_carlo_step(grid, temperature):
    """Perform a single Monte Carlo step."""
    # Randomly select a site (i, j) in the grid
    i, j = np.random.randint(0, N, size=2)

    # Calculate the change in energy if the spin at (i, j) is flipped
    dE = -2 * calculate_energy(grid, i, j)

    # Metropolis criterion:
    # 1. If dE < 0, the flip is accepted (lowers energy).
    # 2. If dE >= 0, the flip is accepted with probability exp(-dE / T).
    if dE < 0 or np.random.rand() < np.exp(-dE / temperature):
        grid[i, j] *= -1  # Flip the spin

# Parameters for Monte Carlo simulation
temperature = 0.5  # Controls the randomness of spin flips
iterations = 50000  # Total number of Monte Carlo steps

# Initialize the grid with a copy of the random grid
evolved_grid = random_grid.copy()

# Perform the Monte Carlo simulation
for step in range(iterations):
    monte_carlo_step(evolved_grid, temperature)  # Apply a Monte Carlo step

    # Optional: Print progress every 10,000 steps for monitoring
    if step % 10000 == 0:
        print(f"Step {step}: Sum of spins = {np.sum(evolved_grid)}")

# Display the Evolved Grid after all Monte Carlo steps
plt.figure(figsize=(6, 6))
plt.matshow(evolved_grid, cmap='bwr', vmin=-1, vmax=1, fignum=3)
plt.colorbar(label='Spin')
plt.title('Evolved Grid (Monte Carlo Simulation)')
plt.show()
