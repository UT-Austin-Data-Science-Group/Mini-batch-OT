import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import ot
from tqdm import tqdm
from utils import model_eval, save_acc


class DigitsDA:
    def __init__(self, model_g, model_f, n_class, logger, out_dir, eta1=0.1, eta2=0.1, epsilon=0.1, batch_epsilon=0.0, mass=0.5, tau=1., test_interval=10):
        self.model_g = model_g   # target model
        self.model_f = model_f
        self.n_class = n_class
        self.logger = logger
        self.out_dir = out_dir
        self.out_file = os.path.join(self.out_dir, 'acc.csv')
        if os.path.exists(self.out_file):
            os.remove(self.out_file)
        self.eta1 = eta1 
        self.eta2 = eta2
        self.epsilon = epsilon
        self.batch_epsilon = batch_epsilon
        self.mass = mass
        self.tau = tau
        self.test_interval = test_interval
        self.logger.info('eta1, eta2, epsilon : {}, {}, {}'.format(self.eta1, self.eta2, self.epsilon))

    def fit(self, source_loader, target_loader, test_loader, n_epochs, criterion=nn.CrossEntropyLoss(), lr=2e-4, k=1, batch_size=25, method='jumbot'):
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            for i, data in tqdm(enumerate(source_loader)):
                # Load data
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, _ = next(target_loader_iter)
                    assert len(xt_mb_all) == batch_size
                except:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, _ = next(target_loader_iter)
                inds_xs = np.split(np.array(range(xs_mb_all.shape[0])), k)

                # Forward
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                    
                for i in range(k):
                    total_loss = 0
                    xs_mb = xs_mb_all[inds_xs[i]].cuda()
                    g_xs_mb = self.model_g(xs_mb)
                    f_g_xs_mb = self.model_f(g_xs_mb)
                    ys = ys_all[inds_xs[i]].cuda()
                    # Classifier loss
                    s_loss = 1./k * criterion(f_g_xs_mb, ys)
                    total_loss += s_loss
                    xt_mb = xt_mb_all[inds_xs[i]].cuda()
                    g_xt_mb = self.model_g(xt_mb)
                    f_g_xt_mb = self.model_f(g_xt_mb)
                    pred_xt = F.softmax(f_g_xt_mb, 1)
                    # Ground cost
                    embed_cost = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                    ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
                    t_cost = - torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))
                    total_cost = self.eta1 * embed_cost + self.eta2 * t_cost
                    # OT computation
                    a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                    if method == 'jumbot':
                        pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, total_cost.detach().cpu().numpy(), self.epsilon, self.tau)
                    elif method == 'jdot':
                        if self.epsilon == 0:
                            pi = ot.emd(a, b, total_cost.detach().cpu().numpy())
                        else:
                            pi = ot.sinkhorn(a, b, total_cost.detach().cpu().numpy(), reg=self.epsilon)
                    elif method == 'jpmbot':
                        if self.epsilon == 0:
                            pi = ot.partial.partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), self.mass)
                        else:
                            pi = ot.partial.entropic_partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), m=self.mass, reg=self.epsilon)
                    pi = torch.from_numpy(pi).float().cuda()
                    da_loss = 1./k * torch.sum(pi * total_cost)
                    mloss = da_loss
                    total_loss += mloss
                    total_loss.backward()
                        
                optimizer_g.step()
                optimizer_f.step()

            if id_epoch % self.test_interval == 0 or (id_epoch == n_epochs-1):
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info('At epoch {} source and test accuracies are {} and {}'.format(id_epoch, source_acc, target_acc))
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    checkpoint = {"model_g": self.model_g.state_dict(), 
                                  "model_f": self.model_f.state_dict(), 
                                  "epoch": id_epoch, "accuracy": target_acc}
                    torch.save(checkpoint, os.path.join(self.out_dir, "best_model.pth"))
        
        # Save final checkpoint
        checkpoint = {"model_g": self.model_g.state_dict(), 
                      "model_f": self.model_f.state_dict(), 
                      "epoch": n_epochs, "accuracy": target_acc}
        torch.save(checkpoint, os.path.join(self.out_dir, "final_model.pth"))
    
    def fit_bomb(self, source_loader, target_loader, test_loader, n_epochs, criterion=nn.CrossEntropyLoss(), lr=2e-4, k=1, batch_size=25, method='jumbot'):
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            for i, data in tqdm(enumerate(source_loader)):
                # Load data
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, _ = next(target_loader_iter)
                    assert len(xt_mb_all) == batch_size
                except:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, _ = next(target_loader_iter)
                inds_xs = np.split(np.array(range(xs_mb_all.shape[0])), k)
                inds_xt = np.split(np.array(range(xt_mb_all.shape[0])), k)

                list_da_loss = []
                # Forward
                with torch.no_grad():
                    for i in range(k):
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()

                        for j in range(k):
                            xt_mb = xt_mb_all[inds_xt[j]].cuda()
                            g_xt_mb = self.model_g(xt_mb)
                            f_g_xt_mb = self.model_f(g_xt_mb)
                            pred_xt = F.softmax(f_g_xt_mb, 1)
                            # Ground cost
                            embed_cost = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                            ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
                            t_cost = - torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))
                            total_cost = self.eta1 * embed_cost + self.eta2 * t_cost
                            # OT computation
                            a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                            if method == 'jumbot':
                                pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, total_cost.detach().cpu().numpy(), self.epsilon, self.tau)
                            elif method == 'jdot':
                                if self.epsilon == 0:
                                    pi = ot.emd(a, b, total_cost.detach().cpu().numpy())
                                else:
                                    pi = ot.sinkhorn(a, b, total_cost.detach().cpu().numpy(), reg=self.epsilon)
                            elif method == 'jpmbot':
                                if self.epsilon == 0:
                                    pi = ot.partial.partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), self.mass)
                                else:
                                    pi = ot.partial.entropic_partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), m=self.mass, reg=self.epsilon)
                            pi = torch.from_numpy(pi).float().cuda()
                            da_loss = torch.sum(pi * total_cost)
                            list_da_loss.append(da_loss)
                    # Solving kxk OT
                    big_C = torch.stack(list_da_loss).view(k, k)
                    if self.batch_epsilon == 0:
                        plan = ot.emd([], [], big_C.detach().cpu().numpy())
                    else:
                        plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=self.batch_epsilon)

                # Reforward
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                    
                for i in range(k):
                    for j in range(k):
                        total_loss = 0
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        f_g_xs_mb = self.model_f(g_xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()
                        # Classifier loss
                        s_loss = 1./(k**2) * criterion(f_g_xs_mb, ys)
                        total_loss += s_loss
                        if plan[i,j] == 0:
                            total_loss.backward()
                            continue
                        xt_mb = xt_mb_all[inds_xt[j]].cuda()
                        g_xt_mb = self.model_g(xt_mb)
                        f_g_xt_mb = self.model_f(g_xt_mb)
                        pred_xt = F.softmax(f_g_xt_mb, 1)
                        # Ground cost
                        embed_cost = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                        ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
                        t_cost = - torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))
                        total_cost = self.eta1 * embed_cost + self.eta2 * t_cost
                        # OT computation
                        a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                        if method == 'jumbot':
                            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, total_cost.detach().cpu().numpy(), self.epsilon, self.tau)
                        elif method == 'jdot':
                            if self.epsilon == 0:
                                pi = ot.emd(a, b, total_cost.detach().cpu().numpy())
                            else:
                                pi = ot.sinkhorn(a, b, total_cost.detach().cpu().numpy(), reg=self.epsilon)
                        elif method == 'jpmbot':
                            if self.epsilon == 0:
                                pi = ot.partial.partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), self.mass)
                            else:
                                pi = ot.partial.entropic_partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), m=self.mass, reg=self.epsilon)
                        pi = torch.from_numpy(pi).float().cuda()
                        da_loss = torch.sum(pi * total_cost)
                        mloss = plan[i,j] * da_loss
                        total_loss += mloss
                        total_loss.backward()
                        
                optimizer_g.step()
                optimizer_f.step()

            if id_epoch % self.test_interval == 0 or (id_epoch == n_epochs-1):
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info('At epoch {} source and test accuracies are {} and {}'.format(id_epoch, source_acc, target_acc))
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    checkpoint = {"model_g": self.model_g.state_dict(), 
                                  "model_f": self.model_f.state_dict(), 
                                  "epoch": id_epoch, "accuracy": target_acc}
                    torch.save(checkpoint, os.path.join(self.out_dir, "best_model.pth"))
        
        # Save final checkpoint
        checkpoint = {"model_g": self.model_g.state_dict(), 
                      "model_f": self.model_f.state_dict(), 
                      "epoch": n_epochs, "accuracy": target_acc}
        torch.save(checkpoint, os.path.join(self.out_dir, "final_model.pth"))

    def fit_bomb2(self, source_loader, target_loader, test_loader, n_epochs, criterion=nn.CrossEntropyLoss(), lr=2e-4, k=1, batch_size=25, method='jumbot'):
        """
        A more stable version when the training loss is small.
        However, it does not work well with the entropic version of BoMb.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            for i, data in tqdm(enumerate(source_loader)):
                # Load data
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, _ = next(target_loader_iter)
                    assert len(xt_mb_all) == batch_size
                except:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, _ = next(target_loader_iter)
                inds_xs = np.split(np.array(range(xs_mb_all.shape[0])), k)
                inds_xt = np.split(np.array(range(xt_mb_all.shape[0])), k)

                list_da_loss = []
                # Forward
                with torch.no_grad():
                    for i in range(k):
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()

                        for j in range(k):
                            xt_mb = xt_mb_all[inds_xt[j]].cuda()
                            g_xt_mb = self.model_g(xt_mb)
                            f_g_xt_mb = self.model_f(g_xt_mb)
                            pred_xt = F.softmax(f_g_xt_mb, 1)
                            # Ground cost
                            embed_cost = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                            ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
                            t_cost = - torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))
                            total_cost = self.eta1 * embed_cost + self.eta2 * t_cost
                            # OT computation
                            a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                            if method == 'jumbot':
                                pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, total_cost.detach().cpu().numpy(), self.epsilon, self.tau)
                            elif method == 'jdot':
                                if self.epsilon == 0:
                                    pi = ot.emd(a, b, total_cost.detach().cpu().numpy())
                                else:
                                    pi = ot.sinkhorn(a, b, total_cost.detach().cpu().numpy(), reg=self.epsilon)
                            elif method == 'jpmbot':
                                if self.epsilon == 0:
                                    pi = ot.partial.partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), self.mass)
                                else:
                                    pi = ot.partial.entropic_partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), m=self.mass, reg=self.epsilon)
                            pi = torch.from_numpy(pi).float().cuda()
                            da_loss = torch.sum(pi * total_cost)
                            list_da_loss.append(da_loss)
                    # Solving kxk OT
                    big_C = torch.stack(list_da_loss).view(k, k)
                    if self.batch_epsilon == 0:
                        plan = ot.emd([], [], big_C.detach().cpu().numpy())
                    else:
                        plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=self.batch_epsilon)
                    mapping = np.argmax(plan, axis=1)

                # Reforward
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                    
                for i in range(k):
                    j = mapping[i]
                    total_loss = 0
                    xs_mb = xs_mb_all[inds_xs[i]].cuda()
                    g_xs_mb = self.model_g(xs_mb)
                    f_g_xs_mb = self.model_f(g_xs_mb)
                    ys = ys_all[inds_xs[i]].cuda()
                    # Classifier loss
                    s_loss = 1./k * criterion(f_g_xs_mb, ys)
                    total_loss += s_loss
                    xt_mb = xt_mb_all[inds_xt[j]].cuda()
                    g_xt_mb = self.model_g(xt_mb)
                    f_g_xt_mb = self.model_f(g_xt_mb)
                    pred_xt = F.softmax(f_g_xt_mb, 1)
                    # Ground cost
                    embed_cost = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                    ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
                    t_cost = - torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))
                    total_cost = self.eta1 * embed_cost + self.eta2 * t_cost
                    # OT computation
                    a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                    if method == 'jumbot':
                        pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, total_cost.detach().cpu().numpy(), self.epsilon, self.tau)
                    elif method == 'jdot':
                        if self.epsilon == 0:
                            pi = ot.emd(a, b, total_cost.detach().cpu().numpy())
                        else:
                            pi = ot.sinkhorn(a, b, total_cost.detach().cpu().numpy(), reg=self.epsilon)
                    elif method == 'jpmbot':
                        if self.epsilon == 0:
                            pi = ot.partial.partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), self.mass)
                        else:
                            pi = ot.partial.entropic_partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), m=self.mass, reg=self.epsilon)
                    pi = torch.from_numpy(pi).float().cuda()
                    da_loss = torch.sum(pi * total_cost)
                    mloss = plan[i,j] * da_loss
                    total_loss += mloss
                    total_loss.backward()
                        
                optimizer_g.step()
                optimizer_f.step()

            if id_epoch % self.test_interval == 0 or (id_epoch == n_epochs-1):
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info('At epoch {} source and test accuracies are {} and {}'.format(id_epoch, source_acc, target_acc))
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    checkpoint = {"model_g": self.model_g.state_dict(), 
                                  "model_f": self.model_f.state_dict(), 
                                  "epoch": id_epoch, "accuracy": target_acc}
                    torch.save(checkpoint, os.path.join(self.out_dir, "best_model.pth"))
        
        # Save final checkpoint
        checkpoint = {"model_g": self.model_g.state_dict(), 
                      "model_f": self.model_f.state_dict(), 
                      "epoch": n_epochs, "accuracy": target_acc}
        torch.save(checkpoint, os.path.join(self.out_dir, "final_model.pth"))

    def source_only(self, source_loader, criterion=nn.CrossEntropyLoss(), lr=2e-4):
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)

        for _ in tqdm(range(10)):
            self.model_g.train()
            self.model_f.train()
            for _, data in enumerate(source_loader):
                # Load data
                xs_mb, ys = data
                xs_mb, ys = xs_mb.cuda(), ys.cuda()
                
                g_xs_mb = self.model_g(xs_mb)
                f_g_xs_mb = self.model_f(g_xs_mb)

                # loss
                s_loss = criterion(f_g_xs_mb, ys)

                # train the model 
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                tot_loss = s_loss
                tot_loss.backward()

                optimizer_g.step()
                optimizer_f.step()
                
        source_acc = self.evaluate(source_loader)
        self.logger.info("Source accuracy is {}".format(source_acc))
    
    def evaluate(self, data_loader):
        score = model_eval(data_loader, self.model_g, self.model_f)
        return score
