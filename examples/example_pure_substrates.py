"""
Plot pure-substrate spectra from the `bioprocess_substrates` dataset.

A "pure" spectrum here is a measurement of a single substrate in isolation:
its concentration vector has exactly one non-zero entry (one substrate
present) and zeros for all other substrates.
"""

import matplotlib.pyplot as plt
import numpy as np

from raman_data import raman_data, RamanDataset

# Load the dataset. `dataset.spectra` has shape (n_samples, n_wavenumbers),
# `dataset.targets` has shape (n_samples, n_substrates) with concentrations,
# and `dataset.target_names` lists the substrate name per target column.
dataset: RamanDataset = raman_data("bioprocess_substrates")

# For each sample, count how many substrates have a non-zero concentration.
# Pure spectra are those with exactly one non-zero target.
nonzero_counts = np.sum(dataset.targets > 0, axis=1)
pure_idx = np.where(nonzero_counts == 1)[0]

# Plot every pure spectrum, labelled by the name of its single substrate.
plt.figure(figsize=(10, 6))
for i in pure_idx:
    # The index of the only non-zero target gives the substrate identity.
    substrate_idx = int(np.argmax(dataset.targets[i]))
    name = dataset.target_names[substrate_idx]
    plt.plot(dataset.raman_shifts, dataset.spectra[i], label=name)

plt.xlabel("Raman shift (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("Pure substrate spectra — bioprocess_substrates")
plt.legend()
plt.tight_layout()
plt.show()