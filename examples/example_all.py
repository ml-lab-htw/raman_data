import numpy as np

from examples.utils import plot_samples
from raman_data import TASK_TYPE
from raman_data import raman_data
from raman_data.datasets import pretty_name

dataset_names = raman_data(task_type=None)
datasets = [raman_data(dataset_name) for dataset_name in dataset_names]
num_data = [len(dataset) for dataset in datasets]
f_mins = [np.min(dataset.raman_shifts) for dataset in datasets]
f_maxs = [np.max(dataset.raman_shifts) for dataset in datasets]
sr = [np.diff(dataset.raman_shifts).mean() for dataset in datasets]

# Filter by task type
classification_datasets = raman_data(task_type=TASK_TYPE.Classification)
regression_datasets = raman_data(task_type=TASK_TYPE.Regression)

# Summary
print("\nSummary:")
print(f"Total available datasets: {len(dataset_names)}")
print(f"Classification datasets: {len(classification_datasets)}")
print(f"Regression datasets: {len(regression_datasets)}")
print("\nDataset details:")
for idx in range(len(dataset_names)):
    num_samples = num_data[idx]
    f_min = f_mins[idx]
    f_max = f_maxs[idx]
    sr_ = sr[idx]
    name = pretty_name(dataset_names[idx])
    print(f"- {name}: {num_samples} samples (f_min={f_min}, f_max={f_max}, sr={sr_:.2f})")

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


