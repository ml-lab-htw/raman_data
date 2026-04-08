from collections import Counter
from examples.utils import plot_samples
from raman_data import raman_data

# dataset_name = "organic_compounds_raw"
# dataset = raman_data(dataset_name)
# plot_samples(dataset)


datasets = raman_data(task_type=None)
datasets = [dataset for dataset in datasets if "bioprocess_substrates" in dataset]

for dataset_name in datasets:
    dataset = raman_data(dataset_name)


    # class_counts = Counter(dataset.targets)
    #
    # filtered_classes = {class_name: count for class_name, count in class_counts.items() if count >= 9}
    # sorted_classes = sorted(filtered_classes.items(), key=lambda x: x[1])
    #
    # if len(class_counts) != len(filtered_classes):
    #     print(f"Dataset: {dataset_name}")
    #     print(f"  Number of classes before filtering: {len(class_counts)}")
    #     print(f"  Number of classes after filtering: {len(filtered_classes)}")


    plot_samples(dataset)
