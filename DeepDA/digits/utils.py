import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm


# -------- Eval function --------


def model_eval(dataloader, model_g, model_f):
    """
    Model evaluation function
    args:
    - dataloader : considered dataset
    - model_g : feature exctrator (torch.nn)
    - model_f : classification layer (torch.nn)
    """
    model_g.eval()
    model_f.eval()
    total_samples = 0
    correct_prediction = 0
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img = img.cuda()
            label = label.long().cuda()
            gen_output = model_g(img)
            pred = F.softmax(model_f(gen_output), 1)
            correct_prediction += torch.sum(torch.argmax(pred, 1) == label)
            total_samples += pred.size(0)
        accuracy = correct_prediction.cpu().data.numpy() / total_samples
    return accuracy


# --------SAMPLER-------


class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    Taken from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/sampler.py
    """

    def __init__(self, labels, batch_size):
        self.classes = sorted(set(labels.numpy()))
        print(self.classes)

        n_classes = len(self.classes)
        self._n_samples = batch_size // n_classes
        self._n_remain = batch_size % n_classes
        if self._n_samples == 0:
            raise ValueError(f"batch_size should be bigger than the number of classes, got {batch_size}")

        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_) for class_ in self.classes
        ]

        # batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = int(np.round(self.n_dataset // batch_size))
        if self._n_batches == 0:
            raise ValueError(f"Dataset is not big enough to generate batches with size {batch_size}")
        print("K=", n_classes, "nk=", self._n_samples)
        print("Batch size = ", batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            add_class = set(np.random.choice(self.classes, self._n_remain, replace=False))
            for class_iter in self._class_iters:
                if class_iter.class_ in add_class:
                    add_samples = 1
                else:
                    add_samples = 0
                indices.extend(class_iter.get(self._n_samples + add_samples))

            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches


class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            print(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]


# --------Others-------


def save_acc(file_path, epoch, acc):
    if os.path.exists(file_path):
        header = False
    else:
        header = True
    with open(file_path, mode="a") as f:
        if header:
            f.write(f"epoch,acc\n{epoch},{acc}\n")
        else:
            f.write(f"{epoch},{acc}\n")
