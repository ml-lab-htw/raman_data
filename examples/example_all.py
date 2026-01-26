from examples.utils import plot_samples
from raman_data import TASK_TYPE
from raman_data import raman_data


dataset_names = raman_data(task_type=None)
datasets = [raman_data(dataset_name) for dataset_name in dataset_names]
num_data = [len(dataset) for dataset in datasets]

# Filter by task type
regression_datasets = raman_data(task_type=TASK_TYPE.Regression)
classification_datasets = raman_data(task_type=TASK_TYPE.Classification)

# Summary
print("\nSummary:")
print(f"Total available datasets: {len(dataset_names)}")
print(f"Classification datasets: {len(classification_datasets)}")
print(f"Regression datasets: {len(regression_datasets)}")
print("\nNumber of samples per dataset:")
for idx in range(len(dataset_names)):
    num_samples = num_data[idx]
    name = dataset_names[idx]
    print(f"- {name}: {num_samples} samples")

# Iterate and plot first 5 spectra for all classification datasets
for dataset_name in classification_datasets:
    print(f"\n[Classification] Plotting first 5 spectra for: {dataset_name}")
    dataset = raman_data(dataset_name)
    plot_samples(dataset)


# Iterate and plot first 5 spectra for all regression datasets
for dataset_name in regression_datasets:
    print(f"\n[Regression] Plotting first 5 spectra for: {dataset_name}")
    dataset = raman_data(dataset_name)
    plot_samples(dataset)


