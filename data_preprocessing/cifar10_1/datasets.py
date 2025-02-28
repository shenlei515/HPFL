from torchvision.datasets.utils import download_url
import os
import numpy as np
from PIL import Image
import torch.utils.data as data

class CIFAR10Val1(object):
    """Borrowed from https://github.com/modestyachts/CIFAR-10.1"""

    stats = {
        "v4": {
            "data": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_data.npy",
            "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_labels.npy",
        },
        "v6": {
            "data": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy",
            "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy",
        },
    }

    def __init__(self, root, data_name=None, version=None, transform=None):
        version = "v6" if version is None else version
        assert version in ["v4", "v6"]

        self.data_name = data_name
        self.path_data = os.path.join(root, f"cifar10.1_{version}_data.npy")
        self.path_labels = os.path.join(root, f"cifar10.1_{version}_labels.npy")
        self._download(root, version)

        self.data = np.load(self.path_data)
        self.targets = np.load(self.path_labels).tolist()
        self.data_size = len(self.data)

        self.transform = transform

    def _download(self, root, version):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(url=self.stats[version]["data"], root=root)
        download_url(url=self.stats[version]["labels"], root=root)

    def _check_integrity(self) -> bool:
        if os.path.exists(self.path_data) and os.path.exists(self.path_labels):
            return True
        else:
            return False

    def __getitem__(self, index):
        img_array = self.data[index]
        target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img_array)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.data_size
    

class CIFAR10Val1_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        cifar_dataobj = CIFAR10Val1(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            targets = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            targets = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)




class CIFAR10Val1_truncated_WO_reload(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                full_dataset=None):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.full_dataset = full_dataset

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # print("download = " + str(self.download))
        # cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = self.full_dataset.data
            targets = np.array(self.full_dataset.targets)
        else:
            data = self.full_dataset.data
            targets = np.array(self.full_dataset.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)