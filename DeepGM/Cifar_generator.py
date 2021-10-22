import torch.nn as nn
import torch
from utils import *
import time
import numpy as np
import ot

class Discriminator(nn.Module):
    def __init__(self,image_size,latent_size,num_chanel,hidden_chanels=64):
        super(Discriminator,self).__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.num_chanel = num_chanel
        self.hidden_chanels = hidden_chanels
        self.main1 = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(self.num_chanel, self.hidden_chanels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.hidden_chanels, self.hidden_chanels *2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_chanels * 2, self.hidden_chanels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 4),
            nn.Tanh()
        )
        self.main2= nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.hidden_chanels * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.main1(x)
        y = self.main2(h).view(x.shape[0],-1)
        return y, h

class Generator(nn.Module):
    def __init__(self,image_size,latent_size,num_chanel,hidden_chanels=64):
        super(Generator,self).__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.num_chanel = num_chanel
        self.hidden_chanels = hidden_chanels
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.latent_size, self.hidden_chanels * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.hidden_chanels * 4, self.hidden_chanels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( self.hidden_chanels * 2, self.hidden_chanels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( self.hidden_chanels, self.num_chanel, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, z):
        x= self.main(z.view(z.shape[0],self.latent_size,1,1))
        return x


class Cifar_Generator(nn.Module):
    def __init__(self,image_size,latent_size,num_chanel,hidden_chanels,device):
        super(Cifar_Generator, self).__init__()
        self.image_size = image_size
        self.num_chanel = num_chanel
        self.latent_size = latent_size
        self.hidden_chanels=hidden_chanels
        self.device = device
        self.decoder = Generator(image_size,latent_size,num_chanel,hidden_chanels)

    def train_minibatch(self, model_op, discriminator, optimizer, data, k, m, method='OT', reg=0,breg=0, tau=1, mass=0.9,
                        L=1000,
                        bomb=False,ebomb=False):
        # Sample latent vectors
        z = torch.randn((data.shape[0], self.latent_size))
        # Sample indices
        if (data.shape[0] % k) == 0:
            inds_data = np.split(np.array(range(data.shape[0])), k)
            inds_z = np.split(np.array(range(z.shape[0])), k)
        else:
            real_k = int(data.shape[0] / m)
            if (real_k != 0):
                inds_data = np.split(np.array(range(real_k * m)), real_k)
                inds_z = np.split(np.array(range(real_k * m)), real_k)
                inds_data = list(inds_data)
                inds_z = list(inds_z)
                k = real_k
                if (method != 'sliced') and (data.shape[0] % m != 0):
                    inds_data.append(np.array(range(real_k * m, data.shape[0])))
                    inds_z.append(np.array(range(real_k * m, data.shape[0])))
                    k = k + 1
                else:
                    k = k
            else:
                k = 1
                inds_data = [np.array(range(data.shape[0]))]
                inds_z = [np.array(range(data.shape[0]))]
        # Train discriminator
        dloss = []
        if ((bomb or ebomb) and method != 'sliced'):
            # Forward
            self.eval()
            discriminator.eval()
            with torch.no_grad():
                for i in range(k):
                    for j in range(k):
                        data_mb = data[inds_data[i]].to(self.device)
                        z_mb = z[inds_z[j]].cuda(self.device)
                        fake_mb = self.decoder(z_mb)
                        y_data, feature_data_mb = discriminator(data_mb)
                        y_fake, feature_fake_mb = discriminator(fake_mb)
                        feature_data_mb = feature_data_mb.view(data_mb.size(0), -1)
                        feature_fake_mb = feature_fake_mb.view(z_mb.size(0), -1)
                        cost_matrix = torch.cdist(feature_data_mb, feature_fake_mb) ** 2
                        a, b = ot.unif(cost_matrix.size(0)), ot.unif(cost_matrix.size(1))
                        if method == 'OT':
                            if reg == 0:
                                pi = ot.emd(a, b, cost_matrix.detach().cpu().numpy())
                            else:
                                pi = ot.sinkhorn(a, b, cost_matrix.detach().cpu().numpy(), reg=reg)
                        elif method == 'UOT':
                            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, cost_matrix.detach().cpu().numpy(),
                                                                         reg=reg,
                                                                         reg_m=tau)
                        elif method == 'POT':
                            if reg == 0:
                                pi = ot.partial.partial_wasserstein(a, b, cost_matrix.detach().cpu().numpy(), m=mass)
                            else:
                                pi = ot.partial.entropic_partial_wasserstein(a, b, cost_matrix.detach().cpu().numpy(),
                                                                             m=mass, reg=reg)
                        pi = torch.from_numpy(pi).cuda(self.device)
                        dloss.append(torch.sum(pi * cost_matrix))
                # Solving kxk OT
                big_C = torch.stack(dloss).view(k, k)
                if (bomb):
                    plan = ot.emd([], [], big_C.detach().cpu().numpy())
                elif (ebomb):
                    plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=breg)
        # Refoward
        Dloss = 0
        self.train()
        discriminator.train()
        if (method == 'sliced'):
            optimizer.zero_grad()
            for i in range(k):
                data_mb = data[inds_data[i]].to(self.device)
                y_data, feature_data_mb = discriminator(data_mb)
                label = torch.full((data_mb.shape[0], 1), 1, dtype=torch.float32, device=self.device)
                criterion = nn.BCELoss(reduction='sum')
                errD_real = 1. / (k ** 2) * criterion(y_data, label)
                errD_real.backward()
            optimizer.step()
            optimizer.zero_grad()
            for j in range(k):
                z_mb = z[inds_z[j]].cuda(self.device)
                fake_mb = self.decoder(z_mb)
                y_fake, feature_fake_mb = discriminator(fake_mb)
                label = torch.full((data_mb.shape[0], 1), 0,dtype = torch.float32, device = self.device)
                criterion = nn.BCELoss(reduction='sum')
                errD_real = 1. / (k ** 2) * criterion(y_fake, label)
                errD_real.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            for i in range(k):
                for j in range(k):
                    if (bomb and method != 'sliced'):
                        if (plan[i, j] == 0):
                            continue
                    data_mb = data[inds_data[i]].to(self.device)
                    z_mb = z[inds_z[j]].cuda(self.device)
                    fake_mb = self.decoder(z_mb)
                    y_data, feature_data_mb = discriminator(data_mb)
                    y_fake, feature_fake_mb = discriminator(fake_mb)
                    feature_data_mb = feature_data_mb.view(data_mb.size(0), -1)
                    feature_fake_mb = feature_fake_mb.view(z_mb.size(0), -1)
                    cost_matrix = torch.cdist(feature_data_mb, feature_fake_mb) ** 2
                    a, b = ot.unif(cost_matrix.size(0)), ot.unif(cost_matrix.size(1))
                    if method == 'OT':
                        if reg == 0:
                            pi = ot.emd(a, b, cost_matrix.detach().cpu().numpy())
                        else:
                            pi = ot.sinkhorn(a, b, cost_matrix.detach().cpu().numpy(), reg=reg)
                    elif method == 'UOT':
                        pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, cost_matrix.detach().cpu().numpy(), reg=reg,
                                                                     reg_m=tau)
                    elif method == 'POT':
                        if reg == 0:
                            pi = ot.partial.partial_wasserstein(a, b, cost_matrix.detach().cpu().numpy(), m=mass)
                        else:
                            pi = ot.partial.entropic_partial_wasserstein(a, b, cost_matrix.detach().cpu().numpy(),
                                                                         m=mass, reg=reg)
                    pi = torch.from_numpy(pi).cuda(self.device)
                    if (bomb or ebomb):
                        mloss = -plan[i, j] * torch.sum(pi * cost_matrix)
                    else:
                        mloss = - 1. / (k ** 2) * torch.sum(pi * cost_matrix)
                    Dloss += mloss
                    # Estimate gradient
                    mloss.backward()
            # Gradient descent
            optimizer.step()
        # Train generator
        gloss = []
        if (bomb or ebomb):
            # Forward
            with torch.no_grad():
                self.eval()
                discriminator.eval()
                for i in range(k):
                    for j in range(k):
                        data_mb = data[inds_data[i]].to(self.device)
                        z_mb = z[inds_z[j]].cuda(self.device)
                        fake_mb = self.decoder(z_mb)
                        y_data, feature_data_mb = discriminator(data_mb)
                        y_real, feature_fake_mb = discriminator(fake_mb)
                        feature_data_mb = feature_data_mb.view(data_mb.size(0), -1)
                        feature_fake_mb = feature_fake_mb.view(z_mb.size(0), -1)
                        if (method == 'sliced'):
                            gloss.append(sliced_wasserstein_distance(feature_data_mb, feature_fake_mb,
                                                                     num_projections=L, device=self.device))
                        else:
                            cost_matrix = torch.cdist(feature_data_mb, feature_fake_mb) ** 2
                            a, b = ot.unif(cost_matrix.size(0)), ot.unif(cost_matrix.size(1))
                            if method == 'OT':
                                if reg == 0:
                                    pi = ot.emd(a, b, cost_matrix.detach().cpu().numpy())
                                else:
                                    pi = ot.sinkhorn(a, b, cost_matrix.detach().cpu().numpy(), reg=reg)
                            elif method == 'UOT':
                                pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, cost_matrix.detach().cpu().numpy(),
                                                                             reg=reg,
                                                                             reg_m=tau)
                            elif method == 'POT':
                                if reg == 0:
                                    pi = ot.partial.partial_wasserstein(a, b, cost_matrix.detach().cpu().numpy(),
                                                                        m=mass)
                                else:
                                    pi = ot.partial.entropic_partial_wasserstein(a, b,
                                                                                 cost_matrix.detach().cpu().numpy(),
                                                                                 m=mass, reg=reg)
                            pi = torch.from_numpy(pi).cuda(self.device)
                            gloss.append(torch.sum(pi * cost_matrix))
                # Solving kxk OT
                big_C = torch.stack(gloss).view(k, k)
                if (bomb):
                    plan = ot.emd([], [], big_C.detach().cpu().numpy())
                elif (ebomb):
                    plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=breg)
        # Reforward
        self.train()
        discriminator.train()
        model_op.zero_grad()
        G_loss = 0
        for i in range(k):
            for j in range(k):
                if (bomb or ebomb):
                    if (plan[i, j] == 0):
                        continue
                data_mb = data[inds_data[i]].to(self.device)
                z_mb = z[inds_z[j]].cuda(self.device)
                fake_mb = self.decoder(z_mb)
                y_data, feature_data_mb = discriminator(data_mb)
                y_real, feature_fake_mb = discriminator(fake_mb)
                feature_data_mb = feature_data_mb.view(data_mb.size(0), -1)
                feature_fake_mb = feature_fake_mb.view(z_mb.size(0), -1)
                if (method == 'sliced'):
                    loss = sliced_wasserstein_distance(feature_data_mb, feature_fake_mb,
                                                       num_projections=L, device=self.device)
                else:
                    cost_matrix = torch.cdist(feature_data_mb, feature_fake_mb) ** 2
                    a, b = ot.unif(cost_matrix.size(0)), ot.unif(cost_matrix.size(1))
                    if method == 'OT':
                        if reg == 0:
                            pi = ot.emd(a, b, cost_matrix.detach().cpu().numpy())
                        else:
                            pi = ot.sinkhorn(a, b, cost_matrix.detach().cpu().numpy(), reg=reg)
                    elif method == 'UOT':
                        pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, cost_matrix.detach().cpu().numpy(), reg=reg,
                                                                     reg_m=tau)
                    elif method == 'POT':
                        if reg == 0:
                            pi = ot.partial.partial_wasserstein(a, b, cost_matrix.detach().cpu().numpy(), m=mass)
                        else:
                            pi = ot.partial.entropic_partial_wasserstein(a, b, cost_matrix.detach().cpu().numpy(),
                                                                         m=mass, reg=reg)
                    pi = torch.from_numpy(pi).cuda(self.device)
                    loss = torch.sum(pi * cost_matrix)
                if (bomb or ebomb):
                    mloss = plan[i, j] * loss
                else:
                    mloss = 1. / (k ** 2) * loss
                G_loss += mloss
                # Backward
                mloss.backward()
        # Gradient descent
        model_op.step()
        return G_loss, Dloss




