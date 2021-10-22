import os
import torch
import ot
import numpy as np
def save_acc(file_path, epoch, acc):
    if os.path.exists(file_path):
        header = False
    else:
        header = True
    with open(file_path, mode='a') as f:
        if header:
            f.write(f"epoch,acc\n{epoch},{acc}\n")
        else:
            f.write(f"{epoch},{acc}\n")
def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
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
def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)