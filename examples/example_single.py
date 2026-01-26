import random

import numpy as np
from matplotlib import pyplot as plt
from raman_data import raman_data
from raman_data.datasets import pretty_name

random.seed(0)

dataset_name = "adenine_cAu"
dataset = raman_data(dataset_name)

if len(dataset) == 0:
    print(f"Dataset {dataset_name} is empty. Nothing to plot.")
    raise SystemExit(0)

print(f"Plotting up to 5 random spectra for: {dataset_name}")

# choose up to 5 random indices (handles datasets smaller than 5)
num_to_plot = min(5, len(dataset))
indices = random.sample(range(len(dataset)), k=num_to_plot)

for idx, i in enumerate(indices):
    if not isinstance(dataset.raman_shifts, list):
        if isinstance(dataset.targets[i], (int, np.integer)):
            plt.plot(dataset.raman_shifts, dataset.spectra[i], label=f"{i+1}: {dataset.target_names[dataset.targets[i]]}")
        else:
            if isinstance(dataset.targets[i], (float, np.floating)):
                plt.plot(dataset.raman_shifts, dataset.spectra[i], label=f"{i+1}: {dataset.target_names[0]}: {dataset.targets[i]:.2f}")
            else:
                plt.plot(dataset.raman_shifts, dataset.spectra[i], label=f"{i+1}: {dataset.target_names[0]}: {dataset.targets[i][0]:.2f}")
    else:
        plt.plot(dataset.raman_shifts[i], dataset.spectra[i], label=f"{i+1}: {dataset.target_names[dataset.targets[i]]}")

plt.grid()
plt.xlabel('Raman Shift')
plt.ylabel('Intensity')
plt.title(f"{pretty_name(dataset_name)} - Random {num_to_plot}/{len(dataset)} Spectra")
plt.legend()
plt.show()
