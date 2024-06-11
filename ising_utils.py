import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import ipywidgets as widgets
import importlib
import sys
from matplotlib.collections import LineCollection


def initialize_lattice(N):
    """Initialize the lattice with random spins."""
    lattice = np.random.choice([-1, 1], size=(N, N))
    return lattice


def glauber_acceptance(delta_E, T):
    """Glauber acceptance function."""
    return np.exp(-delta_E / T) / (1 + np.exp(-delta_E / T))


def metropolis_acceptance(delta_E, T):
    """Metropolis acceptance function."""
    return np.where(delta_E <= 0, 1, np.exp(-delta_E / T))


def glauber_step(lattice, beta):
    """Perform one step of the Glauber dynamics."""
    N = lattice.shape[0]
    i, j = np.random.randint(0, N, size=2)
    spin = lattice[i, j]
    neighbors = lattice[(i+1)%N, j] + lattice[(i-1)%N, j] + lattice[i, (j+1)%N] + lattice[i, (j-1)%N]
    delta_E = 2 * spin * neighbors

    if np.random.rand() < glauber_acceptance(delta_E, 1/beta):
        lattice[i, j] *= -1


def metropolis_step(lattice, beta):
    """Perform one step of the Metropolis algorithm."""
    N = lattice.shape[0]
    i, j = np.random.randint(0, N, size=2)
    spin = lattice[i, j]
    neighbors = lattice[(i+1)%N, j] + lattice[(i-1)%N, j] + lattice[i, (j+1)%N] + lattice[i, (j-1)%N]
    delta_E = 2 * spin * neighbors

    if np.random.rand() < metropolis_acceptance(delta_E, 1/beta):
        lattice[i, j] *= -1


def simulate_metropolis_interactive(lattice, beta, steps):
    """Simulate the Metropolis dynamics and store snapshots."""
    snapshots = [lattice.copy()]
    for _ in range(steps):
        metropolis_step(lattice, beta)
        if _ % 1 == 0:  # Take a snapshot every 10 steps
            snapshots.append(lattice.copy())
    return snapshots


def simulate_glauber_interactive(lattice, beta, steps):
    """Simulate the Glauber dynamics and store snapshots."""
    snapshots = [lattice.copy()]
    for _ in range(steps):
        glauber_step(lattice, beta)
        if _ % 1 == 0:  # Take a snapshot every 10 steps
            snapshots.append(lattice.copy())
    return snapshots

### Plotting functions

def plot_lattice(lattice, step):
    """Plot the lattice for a given step."""
    plt.imshow(lattice[step], cmap='gray')
    plt.title(f"Step {step}")
    plt.show()


def plot_glauber_acceptance():
# Generate a range of Delta E values
    delta_E_values = np.linspace(-10, 10, 400)

    # Temperatures to plot
    temperatures = [0.5, 1.0, 2.0, 5.0]
    plt.figure(figsize=(10, 6))

    for T in temperatures:
        p_values = glauber_acceptance(delta_E_values, T)
        plt.plot(delta_E_values, p_values, label=f'T = {T}')

    plt.xlabel(r'$\Delta E$')
    plt.ylabel('Acceptance Probability $p(\Delta E, T)$')
    plt.title('Glauber Acceptance Function')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metroplis_acceptance():
    # Generate a range of Delta E values
    delta_E_values = np.linspace(-10, 10, 400)

    # Temperatures to plot
    temperatures = [0.5, 1.0, 2.0, 5.0]

    # Plot the Metropolis acceptance function for different temperatures
    plt.figure(figsize=(10, 6))

    for T in temperatures:
        p_values = metropolis_acceptance(delta_E_values, T)
        plt.plot(delta_E_values, p_values, label=f'T = {T}')

    plt.xlabel(r'$\Delta E$')
    plt.ylabel('Acceptance Probability $p(\Delta E, T)$')
    plt.title('Metropolis Acceptance Function')
    plt.legend()
    plt.grid(True)
    plt.show()


### Swendsen-Wang

# Union-Find Data Structure for cluster identification
class UnionFind:
    def __init__(self, N):
        self.parent = np.arange(N)
        self.rank = np.zeros(N)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootX
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


def update_bonds(lattice, beta):
    N = lattice.shape[0]
    uf = UnionFind(N * N)
    
    # Adjust shapes to accommodate bonds that extend outside the box
    horizontal_bonds = np.zeros((N, N + 1), dtype=bool)
    vertical_bonds = np.zeros((N + 1, N), dtype=bool)
    
    for i in range(N):
        for j in range(N):
            # Only create a vertical bond if the spins are the same
            if lattice[i, j] == lattice[(i + 1) % N, j] and np.random.rand() < 1 - np.exp(-2 * beta):
                uf.union(i * N + j, ((i + 1) % N) * N + j)
                vertical_bonds[i, j] = True
                if i == N - 1:  # Wrap-around bond at the top edge
                    vertical_bonds[N, j] = True
            # Only create a horizontal bond if the spins are the same
            if lattice[i, j] == lattice[i, (j + 1) % N] and np.random.rand() < 1 - np.exp(-2 * beta):
                uf.union(i * N + j, i * N + (j + 1) % N)
                horizontal_bonds[i, j] = True
                if j == N - 1:  # Wrap-around bond at the right edge
                    horizontal_bonds[i, N] = True

    return uf, horizontal_bonds, vertical_bonds


def update_spins(lattice, uf):
    N = lattice.shape[0]
    clusters = {}
    
    for i in range(N):
        for j in range(N):
            root = uf.find(i * N + j)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append((i, j))
    
    for cluster in clusters.values():
        if np.random.rand() < 0.5:
            for (i, j) in cluster:
                lattice[i, j] *= -1

def simulate_swendsen_wang_interactive(lattice, beta, steps):
    N = lattice.shape[0]

    # each of the snapshot lists should have the same number of elements before the for-loop
    #lattice_snapshots = [lattice.copy()]
    lattice_snapshots = []
    bond_snapshots = []
    
    # Initialize bonds before the loop
    horizontal_bonds = np.zeros((N, N + 1), dtype=bool)
    vertical_bonds = np.zeros((N + 1, N), dtype=bool)
    # bond_snapshots.append((horizontal_bonds.copy(), vertical_bonds.copy())) # Snapshot before bond update

    
    for step in range(steps):
        if step % 2 == 0:  # Bond update step
            uf, horizontal_bonds, vertical_bonds = update_bonds(lattice, beta)
            bond_snapshots.append((horizontal_bonds.copy(), vertical_bonds.copy())) # bond snapshot after bond update
            lattice_snapshots.append(lattice.copy())  # spin snapshot after bond update
        else: # Spin update step
            update_spins(lattice, uf)  # Pass the updated union-find object
            lattice_snapshots.append(lattice.copy())  # spin snapshot after spin update
            bond_snapshots.append((horizontal_bonds.copy(), vertical_bonds.copy()))  # bond snapshot after spin update 
        
    return lattice_snapshots, bond_snapshots


def plot_lattice_with_bonds(lattice_snapshots, bond_snapshots, step):
    lattice = lattice_snapshots[step]
    N = lattice.shape[0]
    
    fig, ax = plt.subplots()
    ax.set_title(f"Step {step}")

    bond_index = step
    horizontal_bonds, vertical_bonds = bond_snapshots[bond_index]
    
    if step % 2 == 0: 
        alpha = 1.0
    else:  
        alpha = 0.3
    
    # Plot bonds
    segments = []
    for i in range(N):
        for j in range(N):
            if horizontal_bonds[i, j]:
                segments.append(((j, i), ((j + 1), i)))
            if vertical_bonds[i, j]:
                segments.append(((j, i), (j, (i + 1))))
    bond_lines = LineCollection(segments, colors='red', alpha=alpha, zorder=1)
    ax.add_collection(bond_lines)
    
    # Plot lattice using discs on top of bonds
    x, y = np.meshgrid(range(N), range(N))
    x, y = x.flatten(), y.flatten()
    spins = lattice.flatten()
    
    colors = np.where(spins == 1, 'black', 'white')
    ax.scatter(x, y, c=colors, s=100, edgecolors='k', zorder=2)  # s is the size of the discs, zorder ensures it's on top
    
    ax.set_xlim(-0.5, N + .5)
    ax.set_ylim(-0.5, N + .5)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Invert y-axis to match the imshow orientation
    plt.show()