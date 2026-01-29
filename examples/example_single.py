from examples.utils import plot_samples
from raman_data import raman_data

dataset_name = "covid19_serum"
dataset = raman_data(dataset_name)

plot_samples(dataset)
