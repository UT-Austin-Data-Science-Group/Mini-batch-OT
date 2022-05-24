import os

import numpy as np
import ot
import torch


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


def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections**2, dim=1, keepdim=True))
    return projections


def sliced_wasserstein_distance(first_samples, second_samples, num_projections=1000, device="cuda"):
    dim = second_samples.size(1)
    projections = rand_projections(dim, num_projections).to(device)
    first_projections = first_samples.matmul(projections.transpose(0, 1))
    second_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = torch.abs(
        (
            torch.sort(first_projections.transpose(0, 1), dim=1)[0]
            - torch.sort(second_projections.transpose(0, 1), dim=1)[0]
        )
    )
    wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, 2), dim=1)
    return wasserstein_distance.mean()


def compute_true_Wasserstein(X, Y, p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)


def save_dmodel(model, optimizer, dis, disoptimizer, tnet, optnet, epoch, folder):
    dictionary = {}
    dictionary["epoch"] = epoch
    dictionary["model"] = model.state_dict()
    dictionary["optimizer"] = optimizer.state_dict()
    if not (disoptimizer is None):
        dictionary["dis"] = dis.state_dict()
        dictionary["disoptimizer"] = disoptimizer.state_dict()
    else:
        dictionary["dis"] = None
        dictionary["disoptimizer"] = None
    if not (tnet is None):
        dictionary["tnet"] = tnet.state_dict()
        dictionary["optnet"] = optnet.state_dict()
    else:
        dictionary["tnet"] = None
        dictionary["optnet"] = None

    torch.save(dictionary, folder + "/model" + str(epoch) + ".pth")


def load_dmodel(folder, epoch=-1):
    if epoch == -1:
        dictionary = torch.load(folder + "/modelstop.pth")
    else:
        dictionary = torch.load(folder + "/model" + str(epoch) + ".pth")
    return (
        dictionary["epoch"],
        dictionary["model"],
        dictionary["optimizer"],
        dictionary["tnet"],
        dictionary["optnet"],
        dictionary["dis"],
        dictionary["disoptimizer"],
    )
