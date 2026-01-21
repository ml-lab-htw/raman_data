"""
Originally from https://github.com/lyn1874/raman_spectra_matching_with_contrastive_learning
@author: bo
"""
import os.path

import numpy as np


def get_preprocessed_data(path):
    xy_train = np.load(os.path.join(path,"scale_xy_train.npy"))
    gt_train = np.load(os.path.join(path,"yclass_train.npy"))
    xy_val = np.load(os.path.join(path,"scale_xy_val.npy"))
    gt_val = np.load(os.path.join(path,"yclass_val.npy"))

    raman_shifts = np.arange(stop=xy_train.shape[1]+1, start=1)
    spectra = np.concatenate([xy_train, xy_val], axis=0)
    targets = np.concatenate([gt_train, gt_val], axis=0)

    return spectra, raman_shifts, targets


def get_raw_data(path):
    xy = np.load(os.path.join(path, "xy.npy"), allow_pickle=True)
    xx = np.load(os.path.join(path,"xx.npy"), allow_pickle=True)
    targets = np.load(os.path.join(path, "yclass.npy"))
    unique_class = np.unique(targets)

    raman_shifts = [np.array(x) for x in xx]
    spectra = [np.array(x) for x in xy]

    return spectra, raman_shifts, targets












