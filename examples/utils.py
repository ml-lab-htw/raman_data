import os
import random

import numpy as np
from matplotlib import pyplot as plt

from raman_data import RamanDataset

random.seed(0)


def plot_samples(dataset:RamanDataset, num_samples:int=5):
    if len(dataset) == 0:
        print(f"Dataset {dataset.name} is empty. Nothing to plot.")
        raise ValueError("Empty dataset.")

    print(f"Plotting up to {num_samples} random spectra for: {dataset.name}")

    # choose up to 5 random indices (handles datasets smaller than 5)
    num_to_plot = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), k=num_to_plot)
    fig = plt.figure(figsize=(8, 4))

    for idx, i in enumerate(indices):
        if not isinstance(dataset.raman_shifts, list):
            if isinstance(dataset.targets[i], (int, np.integer)):
                plt.plot(dataset.raman_shifts, dataset.spectra[i],
                         label=f"{dataset.target_names[dataset.targets[i]]}")
            else:
                if isinstance(dataset.targets[i], (float, np.floating)):
                    plt.plot(dataset.raman_shifts, dataset.spectra[i],
                             label=f"{dataset.target_names[0]}: {dataset.targets[i]:.2f}")
                else:
                    plt.plot(dataset.raman_shifts, dataset.spectra[i],
                             label=f"{dataset.target_names[0]}: {dataset.targets[i][0]:.2f}")
        else:
            plt.plot(dataset.raman_shifts[i], dataset.spectra[i],
                     label=f"{dataset.target_names[dataset.targets[i]]}")

    plt.grid()
    plt.xlabel('Raman Shift (cm$^{-1}$)')
    plt.ylabel('Intensity')
    plt.title(f"{dataset.name} - Random {num_to_plot}/{len(dataset)} Spectra")
    plt.legend()

    # set figure background to white
    fig.set_facecolor('white')

    # export as pdf

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{dataset.name}.pdf", pad_inches=0, bbox_inches='tight')
    plt.show()