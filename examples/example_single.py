import random
from matplotlib import pyplot as plt
from raman_data import raman_data
random.seed(0)

dataset_name = "knowitall_organics_preprocessed"
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
        plt.plot(dataset.raman_shifts, dataset.spectra[i], label=f"{i+1}: {dataset.target_names[dataset.targets[i]]}")
    else:
        plt.plot(dataset.raman_shifts[i], dataset.spectra[i], label=f"{i+1}: {dataset.target_names[dataset.targets[i]]}")

plt.grid()
plt.xlabel('Raman Shift')
plt.ylabel('Intensity')
plt.title(f"{dataset_name} - Random {num_to_plot}/{len(dataset)} Spectra")
plt.legend()
plt.show()
