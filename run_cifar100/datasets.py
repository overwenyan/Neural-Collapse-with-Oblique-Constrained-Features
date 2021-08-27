import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.utils import make_grid
import torch


class SampledCIFAR100(CIFAR100):
    def __init__(self, classes: list, **kwargs):
        super().__init__(**kwargs)
        self.cls_list = classes
        self.cls_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = np.array(self.targets)
        mask = np.isin(targets, classes)
        self.data = self.data[mask, ...]  # (B, H, W, C)
        targets = targets[mask].tolist()
        self.targets = [self.cls_to_idx[cls] for cls in targets]
        assert len(self.data) == len(self.targets)

    def _local_targets_to_global(self) -> list:
        return [self.cls_list[cls] for cls in self.targets]

    def plot_selected_cls(self, sample_per_cls=5):
        imgs = []
        for cls in self.cls_list:
            targets = np.array(self.targets)
            mask = (targets == cls)
            imgs.append(self.data[mask, ...][:sample_per_cls, ...])

        imgs_tensor = torch.Tensor(np.concatenate(imgs, axis=0))  # (N, H, W, C) -> (N, C, H, W)
        # print(f"{imgs_tensor.shape}, {imgs_tensor.dtype}")
        img_grid = make_grid(imgs_tensor.permute(0, 3, 1, 2),
                             imgs_tensor.shape[0] // len(self.cls_list), normalize=True)  # (C, H, W) -> (H, W, C)
        # self.classes is the whole list, and self.cls_list: [10, 28, ...]
        # self.targets: [0, 1, ...]
        labels = [self.classes[cls] for cls in self.cls_list]
        # print(labels)
        # print(self.cls_list)
        # print(self.targets)
        # print(self._local_targets_to_global())
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.show()


def select_k_cls(num_cls, batch_size, if_plot_batch=False, **kwargs):
    """
    Returns train loader and test loader.
    """
    # select input images and datasets
    print("start")
    if "transformer" not in kwargs:
        transformer = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    tmp_train_dataset = CIFAR100("./data", transform=None, download=True)
    classes = np.random.choice(len(tmp_train_dataset.classes), num_cls, replace=False)
    print("classes: {}".format(classes))
    train_dataset = SampledCIFAR100(classes, root="./data",
                                    transform=transformer, download=True)
    test_dataset = SampledCIFAR100(classes, root="./data",
                                   transform=transformer, download=True,
                                   train=False)

    num_workers = kwargs.get("num_workers", 2)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              num_workers=num_workers)
    # print(type(train_loader))
    test_loader = DataLoader(test_dataset, batch_size, num_workers)

    # plot the datasets
    if if_plot_batch:
        cnt_sample_train = 0
        cnt_classs_train = np.zeros(classes.shape[0])
        print("classes.shape[0]: {}".format(classes.shape[0]))
        for X, y in train_loader:
            # break
            if cnt_sample_train == 1:
                # pass
                # print(X, y)
                print("shape X: {}\tshape y: {}".format(X.shape, y.shape))
            
            for idx in range(classes.shape[0]):
                idx_0_list = np.where(y == idx)[0]
                cnt_classs_train[idx] += len(idx_0_list)
                # print("class: {}, cnt: {}".format(idx, len(idx_0_list)))
            cnt_sample_train += 1
        # print("cnt_sample_train = {}".format(cnt_sample_train))
        for cls_idx in range(classes.shape[0]):
            print("cls_idx: {}, cnt_sample_train = {}".format(cls_idx, cnt_classs_train[cls_idx]))

        nrows = min(4, int(np.floor(np.sqrt(batch_size))))
        print(nrows)
        num_samples = nrows ** 2
        img_grid = make_grid(X[:num_samples, ...], nrows, normalize=True)  # (B, C, H, W) -> (H, W, C)
        df = pd.DataFrame(np.array([train_dataset.classes[i] for i in y[:num_samples]]).reshape((nrows, nrows)))
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.show()
        print(df)
    return train_loader, test_loader


if __name__ == '__main__':
    # train_loader, test_loader = select_k_cls(num_cls=5, batch_size=128, if_plot_batch=True)
    train_loader, test_loader = select_k_cls(num_cls=20, batch_size=128, if_plot_batch=True)
    print()
