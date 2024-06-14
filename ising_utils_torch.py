import numpy as np
import torch
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import ipywidgets as widgets
import importlib
import sys
from matplotlib.collections import LineCollection


def initialize_lattice(N):
    """Initialize the lattice with random spins."""
    lattice = torch.randint(low=0, high=2, size=(N, N)) * 2 - 1
    return lattice

### Swendsen-Wang

# Union-Find Data Structure for cluster identification
class UnionFind:
    def __init__(self, N):
        self.parent = torch.arange(N, dtype=torch.int64)
        self.rank = torch.zeros(N, dtype=torch.int64)
    
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
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


def update_bonds(lattice, beta):
    N = lattice.shape[0]
    uf = UnionFind(N * N)
    
    # Adjust shapes to accommodate bonds
    horizontal_bonds = torch.zeros((N, N), dtype=torch.bool)
    vertical_bonds = torch.zeros((N, N), dtype=torch.bool)
    
    for i in range(N):
        for j in range(N):
            # Only create a vertical bond if the spins are the same and within bounds
            if i < N - 1 and lattice[i, j] == lattice[i + 1, j] and torch.rand(1).item() < 1 - torch.exp(torch.tensor(-2 * beta)).item():
                uf.union(i * N + j, (i + 1) * N + j)
                vertical_bonds[i, j] = True
            # Only create a horizontal bond if the spins are the same and within bounds
            if j < N - 1 and lattice[i, j] == lattice[i, j + 1] and torch.rand(1).item() < 1 - torch.exp(torch.tensor(-2 * beta)).item():
                uf.union(i * N + j, i * N + j + 1)
                horizontal_bonds[i, j] = True

    return uf, horizontal_bonds, vertical_bonds


def update_spins(lattice, uf):
    N = lattice.shape[0]
    clusters = {}
    
    for i in range(N):
        for j in range(N):
            root = uf.find(i * N + j).item()  # Convert tensor to Python integer
            if root not in clusters:
                clusters[root] = []
            clusters[root].append((i, j))
    
    for cluster in clusters.values():
        if torch.rand(1).item() < 0.5:
            for (i, j) in cluster:
                lattice[i, j] *= -1

    return lattice


def simulate_swendsen_wang_interactive(lattice, beta, steps):
    N = lattice.shape[0]

    # each of the snapshot lists should have the same number of elements before the for-loop
    lattice_snapshots = []
    bond_snapshots = []
    
    # Initialize bonds before the loop
    horizontal_bonds = torch.zeros((N, N + 1), dtype=torch.bool)
    vertical_bonds = torch.zeros((N + 1, N), dtype=torch.bool)
    
    for step in range(steps):
        if step % 2 == 0:  # Bond update step
            uf, horizontal_bonds, vertical_bonds = update_bonds(lattice, beta)
            bond_snapshots.append((horizontal_bonds.clone(), vertical_bonds.clone()))  # bond snapshot after bond update
            lattice_snapshots.append(lattice.clone())  # spin snapshot after bond update
        else:  # Spin update step
            update_spins(lattice, uf)  # Pass the updated union-find object
            lattice_snapshots.append(lattice.clone())  # spin snapshot after spin update
            bond_snapshots.append((horizontal_bonds.clone(), vertical_bonds.clone()))  # bond snapshot after spin update
        
    return lattice_snapshots, bond_snapshots

### Plotting

def plot_lattice(lattice, step):
    """Plot the lattice for a given step."""
    plt.imshow(lattice[step], cmap='gray')
    plt.title(f"Step {step}")
    plt.show()

def plot_lattice_with_bonds(lattice_snapshots, bond_snapshots, step, figsize=(10,10)):
    lattice = lattice_snapshots[step]
    N = lattice.shape[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"Step {step}")

    bond_index = step
    horizontal_bonds, vertical_bonds = bond_snapshots[bond_index]
    
    alpha = 1.0 if step % 2 == 0 else 0.3
    
    # Plot bonds
    segments = []
    for i in range(N):
        for j in range(N):
            if j < N - 1 and horizontal_bonds[i, j].item():  # Check within bounds for horizontal bonds
                segments.append(((j, i), (j + 1, i)))
            if i < N - 1 and vertical_bonds[i, j].item():  # Check within bounds for vertical bonds
                segments.append(((j, i), (j, i + 1)))
    bond_lines = LineCollection(segments, colors='red', alpha=alpha, zorder=1)
    ax.add_collection(bond_lines)
    
    # Plot lattice using discs on top of bonds
    x, y = np.meshgrid(range(N), range(N))
    x, y = x.flatten(), y.flatten()
    spins = lattice.flatten().numpy()
    
    colors = np.where(spins == 1, 'black', 'white')
    ax.scatter(x, y, c=colors, s=(400 / N), edgecolors='k', zorder=2)
    
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().invert_yaxis()  # Invert y-axis to match the imshow orientation
    plt.show()