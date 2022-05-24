import argparse
import os
import os.path as osp
import random

import lr_schedule
import network
import numpy as np
import ot
import pre_process as prep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_list import BalancedBatchSampler, ImageList, ImageList_label
from torch.utils.data import DataLoader
from tqdm import tqdm


def image_classification_test(loader, model, test_10crop=True):
    all_output = []
    all_label = []
    dataset = loader["test"]

    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(dataset[i]) for i in range(10)]
            for _ in tqdm(range(len(dataset[0]))):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    predict_out = nn.Softmax(dim=1)(predict_out)
                    outputs.append(predict_out)
                outputs = sum(outputs) / 10
                all_output.append(outputs.float().cpu())
                all_label.append(labels.float())
        else:
            iter_test = iter(dataset)
            for _ in tqdm(range(len(dataset))):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                _, outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)
                all_output.append(outputs.float().cpu())
                all_label.append(labels.float())

    all_output = torch.cat(all_output, 0)
    all_label = torch.cat(all_label, 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def train(config):
    # set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]["params"])
    prep_dict["target"] = prep.image_train(**config["prep"]["params"])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]["params"])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]["params"])

    # prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    source_list = ["." + i for i in open(data_config["source"]["list_path"]).readlines()]
    target_list = ["." + i for i in open(data_config["target"]["list_path"]).readlines()]

    dsets["source"] = ImageList(source_list, transform=prep_dict["source"])
    if config["args"].stratify_source:
        source_labels = torch.zeros((len(dsets["source"])))

        for i, data in tqdm(enumerate(source_list)):
            source_labels[i] = int(data.split()[1])

        source_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
        dset_loaders["source"] = DataLoader(
            dsets["source"], batch_sampler=source_sampler, num_workers=config["args"].num_worker
        )
    else:
        dset_loaders["source"] = DataLoader(
            dsets["source"], batch_size=train_bs, shuffle=True, num_workers=config["args"].num_worker, drop_last=True
        )
    dsets["target"] = ImageList(target_list, transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(
        dsets["target"], batch_size=train_bs, shuffle=True, num_workers=config["args"].num_worker, drop_last=True
    )
    print("source dataset len:", len(dsets["source"]))
    print("target dataset len:", len(dsets["target"]))

    if prep_config["test_10crop"]:
        for i in range(10):
            test_list = ["." + i for i in open(data_config["test"]["list_path"]).readlines()]
            dsets["test"] = [ImageList(test_list, transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [
                DataLoader(dset, batch_size=test_bs, shuffle=False, num_workers=config["args"].num_worker)
                for dset in dsets["test"]
            ]
    else:
        test_list = ["." + i for i in open(data_config["test"]["list_path"]).readlines()]
        dsets["test"] = ImageList(test_list, transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(
            dsets["test"], batch_size=test_bs, shuffle=False, num_workers=config["args"].num_worker
        )

    dsets["target_label"] = ImageList_label(target_list, transform=prep_dict["target"])
    dset_loaders["target_label"] = DataLoader(
        dsets["target_label"],
        batch_size=test_bs,
        shuffle=False,
        num_workers=config["args"].num_worker,
        drop_last=False,
    )

    class_num = config["network"]["params"]["class_num"]

    # set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    if config["restore_path"]:
        checkpoint = torch.load(osp.join(config["restore_path"], "best_model.pth"))["base_network"]
        ckp = {}
        for k, v in checkpoint.items():
            if "module" in k:
                ckp[k.split("module.")[-1]] = v
            else:
                ckp[k] = v
        base_network.load_state_dict(ckp)
        log_str = "successfully restore from {}".format(osp.join(config["restore_path"], "best_model.pth"))
        config["out_file"].write(log_str + "\n")
        config["out_file"].flush()
        print(log_str)

    parameter_list = base_network.get_parameters()

    # set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config["gpu"].split(",")
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in range(len(gpus))])

    # train
    ot_type = config["ot_type"]
    k = config["k"]
    eta1 = config["eta1"]
    eta2 = config["eta2"]
    epsilon = config["epsilon"]
    tau = config["tau"]
    mass = config["mass"]
    best_step = 0
    best_acc = 0.0
    iter_source = iter(dset_loaders["source"])
    iter_target = iter(dset_loaders["target"])
    batch_size = train_bs
    large_batch_size = batch_size * k
    latent_size = base_network.output_num()
    n_channels = 3
    if prep_config["test_10crop"]:
        img_size = 224
    else:
        img_size = 256
    total_s_loss = 0
    total_da_loss = 0

    for i in tqdm(range(config["num_iterations"]), total=config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.eval()
            temp_acc = image_classification_test(dset_loaders, base_network, test_10crop=prep_config["test_10crop"])
            temp_model = base_network  # nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_step = i
                best_acc = temp_acc
                best_model = temp_model
                checkpoint = {"base_network": best_model.state_dict()}
                torch.save(checkpoint, osp.join(config["output_path"], "best_model.pth"))
                print("\n##########     save the best model.    #############\n")
            log_str = "iter: {:05d}, s_loss: {:.5f}, da_loss: {:.5f}, precision: {:.5f}".format(
                i, total_s_loss, total_da_loss, temp_acc
            )
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)

        if i >= config["stop_step"]:
            log_str = "method {}, iter: {:05d}, precision: {:.5f}".format(config["output_path"], best_step, best_acc)
            config["final_log"].write(log_str + "\n")
            config["final_log"].flush()
            break

        # train one iter
        base_network.train()
        all_xs = torch.rand(large_batch_size, n_channels, img_size, img_size)
        all_ys = torch.zeros(large_batch_size, dtype=torch.long)
        all_xt = torch.rand(large_batch_size, 3, img_size, img_size)
        all_g_xs = torch.rand(large_batch_size, latent_size)
        all_g_xt = torch.rand(large_batch_size, latent_size)
        all_f_g_xt = torch.rand(large_batch_size, class_num)

        # STEP 1. Forward without grad in GPU
        with torch.no_grad():
            for idx in range(k):
                try:
                    xs_mb, ys_mb = next(iter_source)
                    xt_mb, _ = next(iter_target)
                except StopIteration:
                    iter_source = iter(dset_loaders["source"])
                    iter_target = iter(dset_loaders["target"])
                    xs_mb, ys_mb = next(iter_source)
                    xt_mb, _ = next(iter_target)

                xs_mb, xt_mb, ys_mb = xs_mb.cuda(), xt_mb.cuda(), ys_mb.cuda()
                g_xs_mb, f_g_xs_mb = base_network(xs_mb)
                g_xt_mb, f_g_xt_mb = base_network(xt_mb)
                all_xs[idx * batch_size : (idx + 1) * batch_size, :] = xs_mb.detach().cpu()
                all_ys[idx * batch_size : (idx + 1) * batch_size] = ys_mb.detach().cpu()
                all_xt[idx * batch_size : (idx + 1) * batch_size, :] = xt_mb.detach().cpu()
                all_g_xs[idx * batch_size : (idx + 1) * batch_size, :] = g_xs_mb.detach().cpu()
                all_g_xt[idx * batch_size : (idx + 1) * batch_size, :] = g_xt_mb.detach().cpu()
                all_f_g_xt[idx * batch_size : (idx + 1) * batch_size, :] = f_g_xt_mb.detach().cpu()

        # STEP 2. Calculate OT map in CPU
        with torch.no_grad():
            embed_cost = torch.cdist(all_g_xs, all_g_xt) ** 2
            all_ys_oh = F.one_hot(all_ys, num_classes=class_num).float()
            all_pred_xt = torch.log(F.softmax(all_f_g_xt, 1))
            t_cost = -torch.mm(all_ys_oh, torch.transpose(all_pred_xt, 0, 1))
            total_cost = eta1 * embed_cost + eta2 * t_cost
            a, b = ot.unif(total_cost.size(0)), ot.unif(total_cost.size(1))
            if ot_type == "ot":
                if epsilon == 0:
                    plan = ot.emd(a, b, total_cost.detach().cpu().numpy())
                else:
                    plan = ot.sinkhorn(a, b, total_cost.detach().cpu().numpy(), reg=epsilon)
            elif ot_type == "uot":
                plan = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, total_cost.detach().cpu().numpy(), epsilon, tau)
            elif ot_type == "pot":
                nb_dummies = int(np.ceil(large_batch_size / 100))
                if epsilon == 0:
                    plan = ot.partial.partial_wasserstein(
                        a, b, total_cost.detach().cpu().numpy(), m=mass, nb_dummies=nb_dummies
                    )
                else:
                    plan = ot.partial.entropic_partial_wasserstein(
                        a, b, total_cost.detach().cpu().numpy(), m=mass, reg=epsilon
                    )
            mapping = np.argmax(plan, axis=1)

            # if using ot or pot loss, replace the transportation plan
            # by a coefficient vector for efficient implementation
            # calculate coefficient, 1/k -> 1 and 0 -> 0
            if (ot_type != "uot") and (epsilon == 0):
                coef = torch.zeros(large_batch_size)
                for idx in range(large_batch_size):
                    if np.abs(plan[idx, mapping[idx]] - 1 / large_batch_size) <= 1e-6:
                        coef[idx] = 1.0
                coef = coef.unsqueeze(1)

        # STEP 3. Reforward with grad in GPU
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        total_s_loss = 0
        total_da_loss = 0

        for idx in range(0, large_batch_size, batch_size):
            row_inds = np.s_[idx : idx + batch_size]
            col_inds = mapping[row_inds]
            xs_mb = all_xs[row_inds, :]
            ys_mb = all_ys[row_inds]
            xt_mb = all_xt[col_inds, :]
            xs_mb, xt_mb, ys_mb = xs_mb.cuda(), xt_mb.cuda(), ys_mb.cuda()
            g_xs_mb, f_g_xs_mb = base_network(xs_mb)
            g_xt_mb, f_g_xt_mb = base_network(xt_mb)
            pred_xt = torch.log(F.softmax(f_g_xt_mb, 1))
            ys_oh = F.one_hot(ys_mb, num_classes=class_num).float()
            if (ot_type != "uot") and (epsilon == 0):
                coef_m = coef[row_inds].cuda()
                embed_loss = torch.sum((g_xs_mb - g_xt_mb) ** 2 * coef_m, 1)
                t_loss = -torch.sum(ys_oh * pred_xt * coef_m, 1)
                da_loss = eta1 * embed_loss + eta2 * t_loss
                da_loss = torch.mean(da_loss) / k
            else:
                coef_m = torch.from_numpy(plan[row_inds, col_inds]).cuda()
                embed_loss = torch.cdist(g_xs_mb, g_xt_mb) ** 2
                t_loss = -torch.mm(ys_oh, torch.transpose(pred_xt, 0, 1))
                da_loss = eta1 * embed_loss + eta2 * t_loss
                da_loss = torch.sum(coef_m * da_loss)
            s_loss = nn.CrossEntropyLoss()(f_g_xs_mb, ys_mb) / k
            total_loss = s_loss + da_loss
            total_loss.backward()
            total_s_loss += s_loss
            total_da_loss += da_loss

        optimizer.step()
    checkpoint = {"base_network": base_network.state_dict()}
    torch.save(checkpoint, osp.join(config["output_path"], "final_model.pth"))
    return best_acc


if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser(description="Conditional Domain Adversarial Network")
    parser.add_argument("--gpu_id", type=str, nargs="?", default="0", help="device id to run")
    parser.add_argument(
        "--net",
        type=str,
        default="ResNet50",
        choices=[
            "ResNet18",
            "ResNet34",
            "ResNet50",
            "ResNet101",
            "ResNet152",
            "VGG11",
            "VGG13",
            "VGG16",
            "VGG19",
            "VGG11BN",
            "VGG13BN",
            "VGG16BN",
            "VGG19BN",
            "AlexNet",
        ],
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="office",
        choices=["office", "image-clef", "visda", "office-home"],
        help="The dataset or source dataset used",
    )
    parser.add_argument(
        "--s_dset_path", type=str, default="./data/office/amazon_31_list.txt", help="The source dataset path list"
    )
    parser.add_argument(
        "--stratify_source", action="store_true", help="whether to use a stratified sampling on minibatches"
    )
    parser.add_argument(
        "--t_dset_path", type=str, default="./data/office/webcam_10_list.txt", help="The target dataset path list"
    )
    parser.add_argument("--test_interval", type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument("--snapshot_interval", type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument(
        "--output_dir", type=str, default="san", help="output directory of our model (in ../snapshot directory)"
    )
    parser.add_argument(
        "--restore_dir", type=str, default=None, help="restore directory of our model (in ../snapshot directory)"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=36, help="training batch size")
    parser.add_argument("--cos_dist", type=str2bool, default=False, help="the classifier uses cosine similarity.")
    parser.add_argument("--stop_step", type=int, default=0, help="stop steps")
    parser.add_argument("--final_log", type=str, default=None, help="final_log file")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--test_10crop", type=str2bool, default=True)
    # --- OT parameters ---
    parser.add_argument(
        "--ot_type", type=str, default="ot", choices=["ot", "uot", "pot"], help="Type of optimal transport"
    )
    parser.add_argument("--eta1", type=float, default=0.1, help="weight of embedding loss")
    parser.add_argument("--eta2", type=float, default=0.1, help="weight of transportation loss")
    parser.add_argument("--epsilon", type=float, default=0.0, help="OT regularization coefficient")
    parser.add_argument("--tau", type=float, default=1.0, help="marginal penalization coeffidient")
    parser.add_argument("--mass", type=float, default=0.5, help="ratio of masses to be transported")
    parser.add_argument("--k", type=int, default=1, help="number of minibatches")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # train config
    config = {}
    config["args"] = args
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.stop_step + 1
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["ot_type"] = args.ot_type
    config["eta1"] = args.eta1
    config["eta2"] = args.eta2
    config["epsilon"] = args.epsilon
    config["tau"] = args.tau
    config["mass"] = args.mass
    config["k"] = args.k
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    config["restore_path"] = "snapshot/" + args.restore_dir if args.restore_dir else None
    if os.path.exists(config["output_path"]):
        print("checkpoint dir exists, which will be removed")
        import shutil

        shutil.rmtree(config["output_path"], ignore_errors=True)
    if not os.path.isdir("snapshot/"):
        os.mkdir("snapshot/")
    os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    config["prep"] = {"test_10crop": args.test_10crop, "params": {"resize_size": 256, "crop_size": 224}}

    if "ResNet" in args.net:
        net = network.ResNetFc
        config["network"] = {
            "name": net,
            "params": {
                "resnet_name": args.net,
                "use_bottleneck": True,
                "bottleneck_dim": 512,
                "new_cls": True,
                "cos_dist": args.cos_dist,
            },
        }
    elif "VGG" in args.net:
        config["network"] = {
            "name": network.VGGFc,
            "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True},
        }

    config["optimizer"] = {
        "type": optim.SGD,
        "optim_params": {"lr": args.lr, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True},
        "lr_type": "inv",
        "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75},
    }

    config["dataset"] = args.dset
    test_bs = 4
    if config["dataset"] == "office":
        if (
            ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path)
            or ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path)
            or ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path)
            or ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path)
        ):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or (
            "dslr" in args.s_dset_path and "webcam" in args.t_dset_path
        ):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
            args.stop_step = 20000
        else:
            config["optimizer"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 31
        args.stop_step = 20000
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 65
        test_bs = 10
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
        test_bs = 61
    else:
        raise ValueError("Dataset has not been implemented.")

    config["data"] = {
        "source": {"list_path": args.s_dset_path, "batch_size": args.batch_size},
        "target": {"list_path": args.t_dset_path, "batch_size": args.batch_size},
        "test": {"list_path": args.t_dset_path, "batch_size": test_bs},
    }

    if args.lr != 0.001:
        config["optimizer"]["lr_param"]["lr"] = args.lr
        config["optimizer"]["lr_param"]["gamma"] = 0.001
    config["out_file"].write(str(config) + "\n")
    config["out_file"].flush()
    if args.stop_step == 0:
        config["stop_step"] = 10000
    else:
        config["stop_step"] = args.stop_step
    if args.final_log is None:
        config["final_log"] = open("log.txt", "a")
    else:
        config["final_log"] = open(args.final_log, "a")
    train(config)
