import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.spectrogram_folder import make_dataset
import random
import numpy as np

class UnalignedTripletSpectrogramDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        # self.transform = get_transform(opt)
        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))

        # read the triplet from A and B --
        A_spec = np.load(A_path)
        B_spec = np.load(B_path)

        # get the triplet from A and B
        A0, A1, A2 = [self.transform(spec) for spec in A_spec[:]]
        B0, B1, B2 = [self.transform(spec) for spec in B_spec[:]]

        return {'A0': A0, 'A1': A1, 'A2': A2, 'B0': B0, 'B1': B1, 'B2': B2,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedTripletSpectrogramDataset'
