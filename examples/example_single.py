from examples.utils import plot_samples
from raman_data import raman_data, TASK_TYPE

# dataset_name = "organic_compounds_raw"
# dataset = raman_data(dataset_name)
# plot_samples(dataset)


datasets = raman_data(task_type=None)
filtered_datasets = [dataset for dataset in datasets if "bacteria_identification" in dataset]

for dataset_name in filtered_datasets:
    dataset = raman_data(dataset_name)
    plot_samples(dataset)
