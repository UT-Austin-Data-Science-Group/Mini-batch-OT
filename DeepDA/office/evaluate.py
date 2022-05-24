import argparse
import math
import os
import os.path as osp

import network
import numpy as np
import pre_process as prep
import torch
import torch.nn as nn
from data_list import ImageList
from torch.utils.data import DataLoader
from tqdm import tqdm


class Inspector:
    def __init__(self, num_class):
        self.num_class = num_class
        self.preds = []
        self.acc = []

    def add_batch(self, pred, label):
        maxpred, argpred = torch.max(pred.data.cpu(), dim=1)
        sample = np.concatenate(
            [maxpred.numpy().reshape(-1, 1), (argpred == label).float().numpy().reshape(-1, 1)], axis=1
        )
        self.preds.append(sample)

    def report(self):
        preds = np.concatenate(self.preds, axis=0)
        preds = np.array(sorted(preds, key=lambda x: x[0], reverse=True))
        num = len(preds)
        n_ = 0

        for i in range(1, 21):
            n = int(math.floor(num * 0.05 * i))
            acc_top = sum(preds[:n, 1]) / n
            acc = sum(preds[n_:n, 1]) / (n - n_)
            n_ = n
            print(
                "{}%: maxprob {:.3f}, acc_top {:.3f} acc {:.3f};".format(5 * i, preds[n - 1][0], acc_top, acc),
                end="  ",
            )
            if i % 2 == 0:
                print(" ")


class Eval:
    def __init__(self, num_class, log_file):
        self.num_class = num_class
        self.log_file = log_file
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.ignore_index = None

    def Pixel_Accuracy(self):
        if np.sum(self.confusion_matrix) == 0:
            self.log_file.write("Attention: pixel_total is zero!!!\n")
            self.log_file.flush()
            PA = 0
        else:
            PA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        return PA

    def Print_Every_class_Eval(self):
        MPA = np.diag(self.confusion_matrix) / (1 + self.confusion_matrix.sum(axis=1))
        pas = []

        for ind_class in tqdm(range(len(MPA))):
            pa = str(round(MPA[ind_class] * 100, 2)) if not np.isnan(MPA[ind_class]) else "nan"
            pas.append(pa)

        self.log_file.write(str(pas) + "\n")
        self.log_file.write(str(np.nanmean(MPA)) + "\n")
        self.log_file.flush()

    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)

        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


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
    return accuracy, predict.numpy().astype(int), all_label.numpy().astype(int)


def evaluate(config):
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

    source_list = open(data_config["source"]["list_path"]).readlines()
    target_list = open(data_config["target"]["list_path"]).readlines()

    dsets["source"] = ImageList(source_list, transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(
        dsets["source"], batch_size=train_bs, shuffle=True, num_workers=4, drop_last=True
    )
    dsets["target"] = ImageList(target_list, transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(
        dsets["target"], batch_size=train_bs, shuffle=True, num_workers=4, drop_last=True
    )

    if prep_config["test_10crop"]:
        for i in range(10):
            test_list = ["." + i for i in open(data_config["test"]["list_path"]).readlines()]
            dsets["test"] = [ImageList(test_list, transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [
                DataLoader(dset, batch_size=test_bs, shuffle=False, num_workers=4) for dset in dsets["test"]
            ]
    else:
        test_list = ["." + i for i in open(data_config["test"]["list_path"]).readlines()]
        dsets["test"] = ImageList(test_list, transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=4)

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

    gpus = config["gpu"].split(",")
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in range(len(gpus))])

    evaluater = Eval(class_num, log_file=config["out_file"])

    base_network.train(False)
    temp_acc, predict, all_label = image_classification_test(
        dset_loaders, base_network, test_10crop=prep_config["test_10crop"]
    )
    log_str = "arg acc: {}".format(temp_acc)
    config["out_file"].write(log_str + "\n")
    config["out_file"].flush()
    evaluater.add_batch(predict, all_label)
    evaluater.Print_Every_class_Eval()
    evaluater.reset()
    return


if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser(description="Conditional Domain Adversarial Network")
    parser.add_argument("--method", type=str, default="ALDA", choices=["DANN", "ALDA", "OT"])
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
        choices=["office", "visda", "office-home"],
        help="The dataset or source dataset used",
    )
    parser.add_argument(
        "--s_dset_path", type=str, default="./data/office/amazon_31_list.txt", help="The source dataset path list"
    )
    parser.add_argument(
        "--t_dset_path", type=str, default="./data/office/webcam_10_list.txt", help="The target dataset path list"
    )
    parser.add_argument(
        "--output_dir", type=str, default="san", help="output directory of our model (in ../snapshot directory)"
    )
    parser.add_argument(
        "--restore_dir", type=str, default=None, help="restore directory of our model (in ../snapshot directory)"
    )
    parser.add_argument("--batch_size", type=int, default=36, help="batch_size")
    parser.add_argument("--cos_dist", default=False, type=str2bool, help="cos_dist")
    parser.add_argument("--final_log", type=str, default=None, help="final_log file")
    parser.add_argument("--loss_type", type=str, default="all", help="whether add reg_loss or correct_loss.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["method"] = args.method
    config["gpu"] = args.gpu_id
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    config["restore_path"] = "snapshot/" + args.restore_dir if args.restore_dir else None
    if os.path.exists(config["output_path"]):
        print("checkpoint dir exists, which will be removed")
        import shutil

        shutil.rmtree(config["output_path"], ignore_errors=True)
    os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    config["prep"] = {"test_10crop": True, "params": {"resize_size": 256, "crop_size": 224, "alexnet": False}}

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

    config["dataset"] = args.dset
    test_bs = 4
    if config["dataset"] == "office":
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "office-home":
        config["network"]["params"]["class_num"] = 65
        test_bs = 10
    elif config["dataset"] == "visda":
        config["network"]["params"]["class_num"] = 12
        test_bs = 61
    else:
        raise ValueError("Dataset has not been implemented.")

    config["data"] = {
        "source": {"list_path": args.s_dset_path, "batch_size": args.batch_size},
        "target": {"list_path": args.t_dset_path, "batch_size": args.batch_size},
        "test": {"list_path": args.t_dset_path, "batch_size": test_bs},
    }
    config["out_file"].write(str(config) + "\n")
    config["out_file"].flush()

    if args.final_log is None:
        config["final_log"] = open("log.txt", "a")
    else:
        config["final_log"] = open(args.final_log, "a")
    evaluate(config)
