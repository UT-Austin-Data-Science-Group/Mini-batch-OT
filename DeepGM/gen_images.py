from __future__ import print_function

import argparse
import os

import torch
from MnistAutoencoder import MnistAutoencoder
from torchvision.utils import save_image
from utils import load_dmodel


torch.backends.cudnn.enabled = False


def fid_score_images(
    foldername, batch_size, num_images, drand, latent_size, decoder, num_chanel, image_size, epoch_cont, device
):
    if not (os.path.isdir(foldername + "/genimages" + str(epoch_cont))):
        os.makedirs(foldername + "/genimages" + str(epoch_cont))
    with torch.no_grad():
        for i in range(int(num_images / batch_size)):
            fixednoise = drand((batch_size, latent_size)).to(device)
            sample = decoder(fixednoise)
            for _ in range(sample.shape[0]):
                save_image(
                    sample[_].view(1, num_chanel, image_size, image_size),
                    foldername + "/genimages" + str(epoch_cont) + "/image" + str(batch_size * i + _) + ".png",
                    scale_each=True,
                    normalize=True,
                )


def main():
    # train args
    parser = argparse.ArgumentParser(description="AE")
    parser.add_argument("--datadir", default="./", help="path to dataset")
    parser.add_argument("--outdir", default="./result", help="directory to output images")
    parser.add_argument(
        "--batch-size", type=int, default=5000, metavar="N", help="input batch size for training (default: 512)"
    )
    parser.add_argument(
        "--num-batch", type=int, default=2, metavar="N", help="input batch size for training (default: 512)"
    )
    parser.add_argument(
        "--epochs", type=str, default="100", metavar="N", help="number of epochs to train (default: 200)"
    )
    parser.add_argument("--lr", type=float, default=0.0005, metavar="LR", help="learning rate (default: 0.0005)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        metavar="N",
        help="number of dataloader workers if device is CPU (default: 16)",
    )
    parser.add_argument("--seed", type=int, default=16, metavar="S", help="random seed (default: 16)")
    parser.add_argument("--g", type=str, default="circular", help="g")
    parser.add_argument("--e", type=float, default=1, help="e")
    parser.add_argument("--num-projection", type=int, default=1000, help="number projection")
    parser.add_argument("--lam", type=float, default=1, help="Regularization strength")
    parser.add_argument("--p", type=int, default=2, help="Norm p")
    parser.add_argument("--niter", type=int, default=10, help="number of iterations")
    parser.add_argument("--r", type=float, default=1000, help="R")
    parser.add_argument("--dim", type=int, default=100, help="Latent size")
    parser.add_argument("--latent-size", type=int, default=32, help="Latent size")
    parser.add_argument("--dataset", type=str, default="MNIST", help="(CELEBA|CIFAR)")
    parser.add_argument("--model-type", type=str, required=True, help="(mWD|bombWD|bombTWD|mS|bombS|bombTS|)")
    parser.add_argument("--hsize", type=int, default=100, help="Latent size")
    args = parser.parse_args()
    torch.random.manual_seed(args.seed)
    dataset = args.dataset
    model_type = args.model_type
    latent_size = args.latent_size
    num_projection = args.num_projection
    if model_type == "mWD":
        model_dir = os.path.join(
            args.outdir,
            model_type
            + "_n"
            + str(num_projection)
            + "_m"
            + str(args.num_batch)
            + "_size"
            + str(args.batch_size)
            + "_s"
            + str(args.seed),
        )
    elif model_type == "bombWD":
        model_dir = os.path.join(
            args.outdir,
            model_type
            + "_n"
            + str(num_projection)
            + "_m"
            + str(args.num_batch)
            + "_size"
            + str(args.batch_size)
            + "_e"
            + str(args.e)
            + "_iter"
            + str(args.niter)
            + "_s"
            + str(args.seed),
        )
    elif model_type == "bombTWD":
        model_dir = os.path.join(
            args.outdir,
            model_type
            + "_n"
            + str(num_projection)
            + "_m"
            + str(args.num_batch)
            + "_size"
            + str(args.batch_size)
            + "_s"
            + str(args.seed),
        )
    elif model_type == "mWWD":
        model_dir = os.path.join(
            args.outdir,
            model_type + "_m" + str(args.num_batch) + "_size" + str(args.batch_size) + "_s" + str(args.seed),
        )
    elif model_type == "bombWWD":
        model_dir = os.path.join(
            args.outdir,
            model_type
            + "_m"
            + str(args.num_batch)
            + "_size"
            + str(args.batch_size)
            + "_e"
            + str(args.e)
            + "_iter"
            + str(args.niter)
            + "_s"
            + str(args.seed),
        )
    elif model_type == "bombTWWD":
        model_dir = os.path.join(
            args.outdir,
            model_type + "_m" + str(args.num_batch) + "_size" + str(args.batch_size) + "_s" + str(args.seed),
        )
    elif model_type == "mS":
        model_dir = os.path.join(
            args.outdir,
            model_type
            + "_m"
            + str(args.num_batch)
            + "_size"
            + str(args.batch_size)
            + "_be"
            + str(args.be)
            + "_biter"
            + str(args.bniter)
            + "_s"
            + str(args.seed),
        )
    elif model_type == "bombS":
        model_dir = os.path.join(
            args.outdir,
            model_type
            + "_m"
            + str(args.num_batch)
            + "_size"
            + str(args.batch_size)
            + "_e"
            + str(args.e)
            + "_iter"
            + str(args.niter)
            + "_be"
            + str(args.be)
            + "_biter"
            + str(args.bniter)
            + "_s"
            + str(args.seed),
        )
    elif model_type == "bombTS":
        model_dir = os.path.join(
            args.outdir,
            model_type
            + "_m"
            + str(args.num_batch)
            + "_size"
            + str(args.batch_size)
            + "_be"
            + str(args.be)
            + "_biter"
            + str(args.bniter)
            + "_s"
            + str(args.seed),
        )
    if not (os.path.isdir(args.outdir)):
        os.makedirs(args.outdir)
    if not (os.path.isdir(model_dir)):
        os.makedirs(model_dir)
    # determine device and device dep. args
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # set random seed
    torch.manual_seed(args.seed)
    print(
        "batch size {}\nepochs {}\nAdam lr {} \n using device {}\nseed set to {}".format(
            args.batch_size, args.epochs, args.lr, device.type, args.seed
        )
    )
    # build train and test set data loaders
    if dataset == "MNIST":
        image_size = 28
        num_chanel = 1
        model = MnistAutoencoder(image_size=28, latent_size=args.latent_size, hidden_size=100, device=device).to(
            device
        )
    #            train_net(args.latent_size, 1000, transform_net, op_trannet)
    elif dataset == "CIFAR":
        from Cifar_generator import DCGANAE

        image_size = 64
        num_chanel = 3

        model = DCGANAE(image_size=64, latent_size=latent_size, num_chanel=3, hidden_chanels=64, device=device).to(
            device
        )
    elif dataset == "CELEBA" or dataset == "LSUN":
        from Cifar_generator import DCGANAE

        image_size = 64
        num_chanel = 3
        model = DCGANAE(image_size=64, latent_size=latent_size, num_chanel=3, hidden_chanels=64, device=device).to(
            device
        )
    epoch_cont, model_state, optimizer_state, d, s, d2, s2 = load_dmodel(model_dir, args.epochs)
    model.load_state_dict(model_state)
    model.eval()
    print("Continue from epoch " + str(epoch_cont))

    fid_score_images(
        model_dir,
        args.batch_size,
        10000,
        torch.randn,
        latent_size,
        model.decoder,
        num_chanel,
        image_size,
        epoch_cont,
        device,
    )


if __name__ == "__main__":
    main()
