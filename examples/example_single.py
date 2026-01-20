from matplotlib import pyplot as plt

from raman_data import raman_data

dataset_name = "Adenine"

print(f"Plotting first 5 spectra for: {dataset_name}")
dataset = raman_data(dataset_name)

for i in range(min(5, len(dataset))):
    plt.plot(dataset.raman_shifts, dataset.spectra[i], label=f"Spec {i+1}")

plt.xlabel('Raman Shift')
plt.ylabel('Intensity')
plt.title(f"{dataset_name} - First 5/{len(dataset)}  Spectra")
plt.legend()
plt.show()


