###############################################################################
# Code modified from image_folder.py to load spectrogram npy
###############################################################################

import torch.utils.data as data

import os
import os.path

def make_dataset(dir):
    specs = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.npy'):
                path = os.path.join(root, fname)
                specs.append(path)

    return specs
