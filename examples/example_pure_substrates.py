import matplotlib.pyplot as plt
import numpy as np

from raman_data import raman_data, RamanDataset

dataset: RamanDataset = raman_data("bioprocess_substrates")
dataset.targets = np.nan_to_num(dataset.targets, nan=0.0)

nonzero_counts = np.sum(dataset.targets > 0, axis=1)
pure_idx = np.where(nonzero_counts == 1)[0]

plt.figure(figsize=(10, 6))
for i in pure_idx:
    substrate_idx = int(np.argmax(dataset.targets[i]))
    name = dataset.target_names[substrate_idx]
    plt.plot(dataset.raman_shifts, dataset.spectra[i], label=name)

plt.xlabel("Raman shift (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("Pure substrate spectra — bioprocess_substrates")
plt.legend()
plt.tight_layout()
plt.show()