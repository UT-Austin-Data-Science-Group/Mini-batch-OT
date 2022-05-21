import argparse
import os
import random
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import network
import lr_schedule, data_list
from utils import *
import ot 
from tqdm import tqdm


def image_train(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_test(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    return accuracy

def train(args):
    eta1 = args.eta1
    eta2 = args.eta2
    eta3 = args.eta3
    tau = args.tau
    epsilon = args.epsilon
    mass = args.mass
    k = args.k
    ot_type = args.ot_type
    print(eta1, eta2, tau, epsilon, mass)
    log_str = "-"*50 + "\n eta1 = {:.3f}, eta2 = {:.3f} \n".format(eta1, eta2)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)
    
    ## prepare data
    train_bs, test_bs = args.batch_size, args.batch_size * 2

    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train())
    dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train())
    dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test())

    dset_loaders = {}
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    
    source_labels = torch.zeros((len(dsets["source"])))

    for i, data in tqdm(enumerate(open(args.s_dset_path).readlines())):
        source_labels[i] = int(data.split()[1])

    train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
    dset_loaders["source"] = torch.utils.data.DataLoader(dsets["source"],
                                                         batch_sampler=train_batch_sampler, 
                                                         num_workers=args.worker)
    
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=True)
    dset_loaders["test"]   = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False,
                                        num_workers=args.worker)

    if "ResNet" in args.net:
        params = {"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, 'class_num': args.class_num}
        base_network = network.ResNetFc(**params)
    
    if "VGG" in args.net:
        params = {"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, 'class_num': args.class_num}
        base_network = network.VGGFc(**params)

    base_network = base_network.cuda()

    parameter_list = base_network.get_parameters()
    base_network = torch.nn.DataParallel(base_network).cuda() 

    ## set optimizer
    optimizer_config = {"type":torch.optim.SGD, "optim_params":
                        {'lr':args.lr, "momentum":0.9, "weight_decay":5e-4, "nesterov":True}, 
                        "lr_type":"inv", "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}
                    }
    optimizer = optimizer_config["type"](parameter_list,**(optimizer_config["optim_params"]))

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    best_acc = 0
    best_iter = 0

    for i in tqdm(range(args.max_iterations + 1)):

        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            # obtain the class-level weight and evalute the current model
            base_network.train(False)
            temp_acc = image_classification(dset_loaders, base_network)
            
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            args.out_file.write(log_str+"\n")
            args.out_file.flush()
            print(log_str)
            
            if best_acc < temp_acc:
                best_acc = temp_acc
                best_iter = i
                best_model = base_network.state_dict()

        if i % args.test_interval == 0:
            log_str = "\n{}, iter: {:05d}, source/ target: {:02d} / {:02d}\n".format(args.name, i, train_bs, train_bs)
            args.out_file.write(log_str)
            args.out_file.flush()
            print(log_str)
            
        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # train one iter
        # if i % len(dset_loaders["source"]) == 0:
        #     iter_source = iter(dset_loaders["source"])
        # if i % len(dset_loaders["target"]) == 0:
        #     iter_target = iter(dset_loaders["target"])

        for _ in range(k):
            try:
                xs, ys = next(iter_source)
                xt, _ = next(iter_target)
            except:
                iter_source = iter(dset_loaders["source"])
                iter_target = iter(dset_loaders["target"])
                xs, ys = next(iter_source)
                xt, _ = next(iter_target)
            xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()
            g_xs, f_g_xs = base_network(xs)
            g_xt, f_g_xt = base_network(xt)
            
            pred_xt = F.softmax(f_g_xt, 1)

            classifier_loss = torch.nn.CrossEntropyLoss()(f_g_xs, ys) / k

            ys = F.one_hot(ys, num_classes=args.class_num).float()

            M_embed = torch.cdist(g_xs, g_xt)**2
            M_sce = - torch.mm(ys, torch.transpose(torch.log(pred_xt), 0, 1))
            M = eta1 * M_embed + eta2 * M_sce  # Ground cost

            #OT computation
            a, b = ot.unif(g_xs.size()[0]), ot.unif(g_xt.size()[0])
            M_cpu = M.detach().cpu().numpy()
            if ot_type == 'ot':
                if epsilon == 0:
                    pi = ot.emd(a, b, M_cpu)
                else:
                    pi = ot.sinkhorn(a, b, M_cpu, reg=epsilon)
            elif ot_type == 'uot':
                pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M_cpu, epsilon, tau)
            elif ot_type == 'pot':
                if i <= (args.max_iterations / 2):
                    adap_mass = mass/args.max_iterations * i + 0.01
                else:
                    adap_mass = mass / 2
                # adap_mass = int(adap_mass * train_bs + 1) / train_bs
                if epsilon == 0:
                    pi = ot.partial.partial_wasserstein(a, b, M_cpu, adap_mass)
                else:
                    pi = ot.partial.entropic_partial_wasserstein(a, b, M_cpu, m=adap_mass, reg=epsilon)
            pi = torch.from_numpy(pi).float().cuda()
            transfer_loss = eta3 * torch.sum(pi * M) / k

            if i%100==0:
                log_str = "sum(pi) = {}, trasfer loss = {}, min(M) = {}, min(M_embed) = {}, min(M_sce) = {}\n".format(
                    torch.sum(pi).item(), transfer_loss.item(), torch.min(M).item(), torch.min(M_embed).item(), torch.min(M_sce).item())
                args.out_file.write(log_str)
                args.out_file.flush()
                print(log_str)
            
            total_loss = classifier_loss + transfer_loss    
            total_loss.backward()
        
        optimizer.step()

    torch.save(best_model, os.path.join(args.output_dir, "best_model.pt"))
    torch.save(base_network.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    
    log_str = 'Acc: ' + str(np.round(best_acc*100, 2)) + '\n'
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)
    
    with open(f'result_m{ot_type}.txt', 'a') as f:
        log_str = "method {}, iter: {:05d}, precision: {:.5f}\n".format(args.name + '_' + args.output, best_iter, best_acc)
        f.write(log_str)

    return best_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BA3US for Partial Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='run')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=65, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers") 
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet50", "VGG16"])
    
    parser.add_argument('--dset', type=str, default='office_home', choices=["office", "office_home", "imagenet_caltech"])
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--ot_type', type=str, default='ot', choices=['ot', 'uot', 'pot'], help='Type of optimal transport')
    parser.add_argument('--eta1', type=float, default=0.003, help="weight of embedding loss")
    parser.add_argument('--eta2', type=float, default=0.75, help="weight of transportation loss")
    parser.add_argument('--eta3', type=float, default=10., help="weight of transfer loss")
    parser.add_argument('--epsilon', type=float, default=0.01, help="OT regularization coefficient")
    parser.add_argument('--tau', type=float, default=0.06, help="marginal penalization coeffidient")
    parser.add_argument('--mass', type=float, default=0.5, help="ratio of masses to be transported")
    parser.add_argument('--k', type=int, default=1, help='number of minibatches')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        k = 25
        args.class_num = 65
        args.max_iterations = 5000
        args.test_interval = 500
        args.lr=1e-3

    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        k = 10
        args.class_num = 31
        args.max_iterations = 2000
        args.test_interval = 200
        args.lr=1e-4

    if args.dset == 'imagenet_caltech':
        names = ['imagenet', 'caltech']
        k = 84
        args.class_num = 1000
        if args.s == 1:
            args.class_num = 256

        args.max_iterations = 40000
        args.test_interval = 4000
        args.lr=1e-3

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_folder = './data/'
    args.s_dset_path = data_folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = data_folder + args.dset + '/' + names[args.t] + '_' + str(k) + '_list.txt'

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    # args.output_dir = os.path.join('ckp/partial', args.net, args.dset, args.name, args.output)
    args.output_dir = os.path.join('snapshot', args.name, args.output)

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    args.out_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    args.out_file.write(str(args)+'\n')
    args.out_file.flush()
    
    train(args)
