from examples.utils import plot_samples
from raman_data import raman_data, TASK_TYPE

dataset_name = "rwth_acid_species_succinic"

regression_datasets = raman_data(task_type=TASK_TYPE.Regression)
acid_species_datasets = [dataset for dataset in regression_datasets if "acid_species_succinic" in dataset]


for dataset_name in acid_species_datasets:
    dataset = raman_data(dataset_name)
    plot_samples(dataset)
