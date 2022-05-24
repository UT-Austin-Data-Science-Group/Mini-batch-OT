import numpy as np
import ot
import torch
import torch.nn as nn
from utils import sliced_wasserstein_distance


class Generator(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size):
        super(Generator, self).__init__()
        self.image_size = image_size**2
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(True),
            nn.Linear(2 * hidden_size, 4 * hidden_size),
            nn.ReLU(True),
            nn.Linear(4 * hidden_size, self.image_size),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.main(input)


class MnistGenerator(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size, device):
        super(MnistGenerator, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device
        self.decoder = Generator(image_size, hidden_size, latent_size)

    def train_minibatch(
        self, model_op, data, k, m, method="OT", reg=0, breg=0, tau=1, mass=0.9, L=1000, bomb=False, ebomb=False
    ):
        # Sample latent vectors
        z = torch.randn((data.shape[0], self.latent_size))
        # Sample indices
        if (data.shape[0] % k) == 0:
            inds_data = np.split(np.array(range(data.shape[0])), k)
            inds_z = np.split(np.array(range(z.shape[0])), k)
        else:
            real_k = int(data.shape[0] / m)
            if real_k != 0:
                inds_data = np.split(np.array(range(real_k * m)), real_k)
                inds_z = np.split(np.array(range(real_k * m)), real_k)
                inds_data = list(inds_data)
                inds_z = list(inds_z)
                k = real_k
                if method != "sliced":
                    inds_data.append(np.array(range(real_k * m, data.shape[0])))
                    inds_z.append(np.array(range(real_k * m, data.shape[0])))
                    k = k + 1
                else:
                    k = k
            else:
                k = 1
                inds_data = [np.array(range(data.shape[0]))]
                inds_z = [np.array(range(data.shape[0]))]
        # Train generator
        gloss = []
        if bomb or ebomb:
            # Forward
            with torch.no_grad():
                self.eval()
                for i in range(k):
                    for j in range(k):
                        data_mb = data[inds_data[i]].to(self.device)
                        z_mb = z[inds_z[j]].cuda(self.device)
                        fake_mb = self.decoder(z_mb)
                        if method == "sliced":
                            gloss.append(
                                sliced_wasserstein_distance(
                                    data_mb.view(data_mb.shape[0], -1),
                                    fake_mb.view(fake_mb.shape[0], -1),
                                    num_projections=L,
                                    device=self.device,
                                )
                            )
                        else:
                            cost_matrix = (
                                torch.cdist(data_mb.view(data_mb.shape[0], -1), fake_mb.view(fake_mb.shape[0], -1))
                                ** 2
                            )
                            a, b = ot.unif(cost_matrix.size(0)), ot.unif(cost_matrix.size(1))
                            if method == "OT":
                                if reg == 0:
                                    pi = ot.emd(a, b, cost_matrix.detach().cpu().numpy())
                                else:
                                    pi = ot.sinkhorn(a, b, cost_matrix.detach().cpu().numpy(), reg=reg)
                            elif method == "UOT":
                                pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
                                    a, b, cost_matrix.detach().cpu().numpy(), reg=reg, reg_m=tau
                                )
                            elif method == "POT":
                                if reg == 0:
                                    pi = ot.partial.partial_wasserstein(
                                        a, b, cost_matrix.detach().cpu().numpy(), m=mass
                                    )
                                else:
                                    pi = ot.partial.entropic_partial_wasserstein(
                                        a, b, cost_matrix.detach().cpu().numpy(), m=mass, reg=reg
                                    )
                            pi = torch.from_numpy(pi).cuda(self.device)
                            gloss.append(torch.sum(pi * cost_matrix))
                # Solving kxk OT
                big_C = torch.stack(gloss).view(k, k)
                if bomb:
                    plan = ot.emd([], [], big_C.detach().cpu().numpy())
                elif ebomb:
                    plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=breg)
        # Reforward
        self.train()
        model_op.zero_grad()
        G_loss = 0
        for i in range(k):
            for j in range(k):
                if bomb or ebomb:
                    if plan[i, j] == 0:
                        continue
                data_mb = data[inds_data[i]].to(self.device)
                z_mb = z[inds_z[j]].cuda(self.device)
                fake_mb = self.decoder(z_mb)
                if method == "sliced":
                    loss = sliced_wasserstein_distance(
                        data_mb.view(data_mb.shape[0], -1),
                        fake_mb.view(fake_mb.shape[0], -1),
                        num_projections=L,
                        device=self.device,
                    )
                else:
                    cost_matrix = (
                        torch.cdist(data_mb.view(data_mb.shape[0], -1), fake_mb.view(fake_mb.shape[0], -1)) ** 2
                    )
                    a, b = ot.unif(cost_matrix.size(0)), ot.unif(cost_matrix.size(1))
                    if method == "OT":
                        if reg == 0:
                            pi = ot.emd(a, b, cost_matrix.detach().cpu().numpy())
                        else:
                            pi = ot.sinkhorn(a, b, cost_matrix.detach().cpu().numpy(), reg=reg)
                    elif method == "UOT":
                        pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
                            a, b, cost_matrix.detach().cpu().numpy(), reg=reg, reg_m=tau
                        )
                    elif method == "POT":
                        if reg == 0:
                            pi = ot.partial.partial_wasserstein(a, b, cost_matrix.detach().cpu().numpy(), m=mass)
                        else:
                            pi = ot.partial.entropic_partial_wasserstein(
                                a, b, cost_matrix.detach().cpu().numpy(), m=mass, reg=reg
                            )
                    pi = torch.from_numpy(pi).cuda(self.device)
                    loss = torch.sum(pi * cost_matrix)
                if bomb or ebomb:
                    mloss = plan[i, j] * loss
                else:
                    mloss = 1.0 / (k**2) * loss
                G_loss += mloss
                # Backward
                mloss.backward()
        # Gradient descent
        model_op.step()
        return G_loss
