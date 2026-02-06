from examples.utils import plot_samples
from raman_data import raman_data, TASK_TYPE

# dataset_name = "sop_spectral_library_baseline_corrected"
# dataset = raman_data(dataset_name)
# plot_samples(dataset)


regression_datasets = raman_data(task_type=TASK_TYPE.Regression)
acid_species_datasets = [dataset for dataset in regression_datasets if "microgel_size" in dataset]

for dataset_name in acid_species_datasets:
    dataset = raman_data(dataset_name)
    plot_samples(dataset)
