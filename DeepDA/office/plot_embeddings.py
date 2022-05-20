import os
import argparse
import random
import shutil
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import network
import pre_process as prep
from data_list import ImageList

TICK_SIZE = 14
TITLE_SIZE = 20
MARKER_SIZE = 10
NUM_SAMPLES = 2000


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if (torch.cuda.is_available()):
        torch.cuda.manual_seed(seed)


@torch.no_grad()
def extract_feature(loader, model, test_10crop=True):
    model.train(False)
    start_test = True
    if test_10crop:
        iter_test = [iter(loader[i]) for i in range(10)]
        for i in tqdm(range(len(loader[0]))):
            data = [iter_test[j].next() for j in range(10)]

            inputs = [data[j][0].cuda() for j in range(10)]
            labels = data[0][1]

            outputs = []
            for j in range(10):
                _, predict_out = model(inputs[j])
                outputs.append(predict_out)
            outputs = outputs.mean(0, keepdim=True)

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat(
                    (all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    else:
        iter_test = iter(loader)
        for i in tqdm(range(len(loader))):
            data = iter_test.next()

            inputs = data[0].cuda()
            labels = data[1]

            _, outputs = model(inputs)

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = torch.cat(
                    (all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels), 0)
    return all_output, all_label


def plot_embeddings(config, ax, title):
    # set pre-process
    transforms = None
    if config["prep"]["test_10crop"]:
        transforms = prep.image_test_10crop(**config["prep"]['params'])
    else:
        transforms = prep.image_test(**config["prep"]['params'])

    # prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for split in ['source', 'target']:
        bs = data_config[split]["batch_size"]

        t = open(data_config[split]["list_path"]).readlines()
        d = {}
        for tt in t:
            tt = tt.strip().split()
            d.setdefault(tt[-1], [])
            d[tt[-1]].append(tt[0])

        for k in d:
            n = int(len(d[k]) * config['ratio'])
            d[k] = np.random.choice(d[k], size=n, replace=False)
            # d[k] = d[k][:n]

        ds_list = []
        for k, v in d.items():
            for vv in v:
                ds_list.append(f'./{vv} {k}')

        if config["prep"]["test_10crop"]:
            for _ in range(10):
                dsets[split] = [
                    ImageList(ds_list, transform=transforms[i])
                    for i in range(10)
                ]
                dset_loaders[split] = [
                    DataLoader(dset,
                               batch_size=bs,
                               shuffle=False,
                               num_workers=4)
                    for dset in dsets[split]
                ]
        else:
            dsets[split] = ImageList(ds_list, transform=transforms)
            dset_loaders[split] = DataLoader(dsets[split],
                                             batch_size=bs,
                                             shuffle=False,
                                             num_workers=4)

    # set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    if config["restore_path"]:
        chk_path = os.path.join(config["restore_path"], "best_model.pth")
        checkpoint = torch.load(chk_path)["base_network"]
        ckp = {}
        for k, v in checkpoint.items():
            if "module" in k:
                ckp[k.split("module.")[-1]] = v
            else:
                ckp[k] = v
        base_network.load_state_dict(ckp)

    base_network.fc = nn.Identity()

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[
                                       int(i) for i in range(len(gpus))])

    embeddings = {}
    labels = {}
    for split in ['source', 'target']:
        embeddings[split], labels[split] = extract_feature(dset_loaders[split],
                                                           base_network,
                                                           test_10crop=config["prep"]["test_10crop"])

    tsne = TSNE(perplexity=30, n_components=2, init='pca',
                n_iter=3000, random_state=config['seed'])

    ds_labels = torch.cat([
        torch.zeros(len(labels['source'])),
        torch.ones(len(labels['target']))
    ])
    embeddings = torch.cat([
        embeddings['source'],
        embeddings['target']
    ])
    emb_tsne = tsne.fit_transform(embeddings)

    ax.scatter(emb_tsne[ds_labels == 0, 0], emb_tsne[ds_labels == 0, 1], c=labels['source'],
               s=MARKER_SIZE, alpha=0.5, marker='o', cmap=cm.nipy_spectral, label='Source')
    ax.scatter(emb_tsne[ds_labels == 1, 0], emb_tsne[ds_labels == 1, 1], c=labels['target'],
               s=MARKER_SIZE*5, alpha=0.5, marker='+', cmap=cm.nipy_spectral, label='Target')

    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(loc='upper right')
    ax.set_xlim(-125, 125)
    ax.set_ylim(-125, 125)


def parse_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu_id', type=str, nargs='?', default='0',
        help="device id to run"
    )
    parser.add_argument(
        '--net', type=str, default='ResNet50',
        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101",
                 "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19",
                 "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"]
    )
    parser.add_argument(
        '--dset', type=str, default='office',
        choices=['office', 'visda', 'office-home'],
        help="The dataset or source dataset used"
    )
    parser.add_argument(
        '--s_dset_path', type=str,
        default='./data/office/amazon_31_list.txt',
        help="The source dataset path list"
    )
    parser.add_argument(
        '--t_dset_path', type=str,
        default='./data/office/webcam_10_list.txt',
        help="The target dataset path list"
    )
    parser.add_argument(
        '--output_dir', type=str,
        help="output directory of our model (in ../snapshot directory)"
    )
    parser.add_argument(
        '--restore_dir', nargs=3,
        help="restore directory of our model (in ../snapshot directory)"
    )
    parser.add_argument(
        '--titles', nargs=3,
        help="subplot titles"
    )
    parser.add_argument(
        '--batch_size', type=int,
        default=100,
        help="batch_size"
    )
    parser.add_argument(
        '--cos_dist', type=str2bool,
        default=False,
        help="cos_dist"
    )
    parser.add_argument(
        '--ratio', type=float,
        default=0.01,
        help='ratio of instances per class'
    )
    parser.add_argument(
        '--seed', type=int,
        default=2020,
        help="random seed"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config['seed'] = args.seed
    config["gpu"] = args.gpu_id

    config["output_path"] = "snapshot/" + args.output_dir

    if os.path.exists(config["output_path"]):
        print("checkpoint dir exists, which will be removed")
        shutil.rmtree(config["output_path"], ignore_errors=True)
    os.makedirs(config["output_path"], exist_ok=True)

    config["prep"] = {
        "test_10crop": False,
        'params': {
            "resize_size": 256,
            "crop_size": 224,
            'alexnet': False,
        }
    }

    if "ResNet" in args.net:
        net = network.ResNetFc
        config["network"] = {
            "name": net,
            "params": {
                "resnet_name": args.net,
                "use_bottleneck": True,
                "bottleneck_dim": 512,
                "new_cls": True,
                "cos_dist": args.cos_dist
            }
        }
    elif "VGG" in args.net:
        config["network"] = {
            "name": network.VGGFc,
            "params": {
                "vgg_name": args.net,
                "use_bottleneck": True,
                "bottleneck_dim": 256,
                "new_cls": True
            }
        }

    config["dataset"] = args.dset
    config["data"] = {
        "source": {
            "list_path": args.s_dset_path,
            "batch_size": args.batch_size
        },
        "target": {
            "list_path": args.t_dset_path,
            "batch_size": args.batch_size
        },
    }
    config['ratio'] = args.ratio

    if config["dataset"] == "office":
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "office-home":
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "visda":
        config["network"]["params"]["class_num"] = 12
    else:
        raise ValueError('Dataset has not been implemented.')

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    for i in range(3):
        seed_everything(config['seed'])
        config["restore_path"] = "snapshot/" + args.restore_dir[i]
        plot_embeddings(config, ax[i], args.titles[i])
    plt.savefig(config['output_path'] + '/plot.png', bbox_inches='tight')
    plt.savefig(config['output_path'] + '/plot.pdf', bbox_inches='tight')