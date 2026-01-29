from examples.utils import plot_samples
from raman_data import raman_data

dataset_name = "mind_covid"
dataset = raman_data(dataset_name)

plot_samples(dataset)
