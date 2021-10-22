from __future__ import print_function
import argparse
import os
import numpy as np
import torch
from torch import optim
from torchvision import transforms
from experiments import sampling, sampling_eps
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from tqdm import tqdm
import imageio
from PIL import Image
import random
import logging
from Celeba_generator import Celeba_Generator, Discriminator
from fid_score import calculate_fid_given_paths
from utils import *


# torch.backends.cudnn.enabled = False


def main():
    # train args
    parser = argparse.ArgumentParser(description='AE')
    parser.add_argument('--datadir', default='.data', help='path to dataset')
    parser.add_argument('--outdir', default='./result',
                        help='directory to output images')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='GPU id to use')
    parser.add_argument('--m', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--k', type=int, default=8, metavar='N',
                        help='input num batch for training (default: 200)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=16, metavar='S',
                        help='random seed (default: 16)')
    parser.add_argument('--latent-size', type=int, default=128,
                        help='Latent size')
    parser.add_argument('--fid-each', type=int, default=5,
                        help='Latent size')
    parser.add_argument('--method', type=str, default='OT',
                        help='OT')
    parser.add_argument('--bomb', action='store_true',
                        help='whether to use Bomb version')
    parser.add_argument('--reg', type=float, default=1,
                        help='sinkhorn reg')
    parser.add_argument('--ebomb', action='store_true',
                        help='whether to use eBomb version')
    parser.add_argument('--breg', type=float, default=1,
                        help='sinkhorn breg')
    parser.add_argument('--tau', type=float, default=1,
                        help='tau UOT')
    parser.add_argument('--mass', type=float, default=0.9,
                        help='mass POT')
    parser.add_argument('--L', type=int, default=1000,
                        help='L')
    args = parser.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    method = args.method
    latent_size = args.latent_size
    args.epochs = args.epochs * args.k
    description = 'CelebA_' + method + '_k' + str(args.k) + '_m' + str(args.m) \
                  + '_reg' + str(args.reg) + '_tau' + str(args.tau) + '_mass' \
                  + str(args.mass) + '_'+str(args.L)+'_seed' + str(args.seed) + '_' \
                  + str(args.epochs) + 'epochs'

    if (args.bomb or args.ebomb):
        if (args.bomb):
            bomb = True
            ebomb = False
            description = 'BoMb-' + description
        else:
            bomb = False
            ebomb = True
            description = 'eBoMb' + str(args.breg) + '-' + description
        model_dir = os.path.join(args.outdir, description)
    else:
        bomb = False
        ebomb=False
        model_dir = os.path.join(args.outdir, description)

    # create output directories
    LOG_DIR = 'logs/celeba'
    CSV_DIR = 'csv/celeba'
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{description}.log")
    csv_file = os.path.join(CSV_DIR, f"{description}.csv")
    if os.path.exists(log_file):
        os.remove(log_file)
    if os.path.exists(csv_file):
        os.remove(csv_file)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # config logger
    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.info(f"Parameters are: {args}")
    logger.info('batch size {}\nepochs {}\nAdam lr {} \n using device {}\n'.format(
        args.m, args.epochs, args.lr, device.type
    ))

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        datasets.CelebA(args.datadir, split='train', download=True,
                        transform=transforms.Compose([
                            transforms.Resize(64),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])),
        batch_size=args.m * args.k, shuffle=True, num_workers=args.num_workers)

    # model
    model = Celeba_Generator(image_size=64, latent_size=latent_size, num_chanel=3, hidden_chanels=64, device=device).to(
        device)
    dis = Discriminator(64, args.latent_size, 3, 64).to(device)
    disoptimizer = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    fixednoise = torch.randn((64, latent_size)).to(device)

    for epoch in range(0, args.epochs):
        total_g_loss = 0.0
        total_d_loss = 0.0
        logger.info(f'Epoch: {epoch}')
        print(f'Epoch: {epoch}')

        for batch_idx, (data, y) in tqdm(enumerate(train_loader, start=0)):
            g_loss, d_loss = model.train_minibatch(optimizer, dis, disoptimizer, data, args.k,args.m, method, args.reg,args.breg, args.tau,
                                                          args.mass,args.L, bomb,ebomb)
            total_g_loss += g_loss
            total_d_loss += d_loss

        total_g_loss /= (batch_idx + 1)
        total_d_loss /= (batch_idx + 1)

        if (bomb):
            logger.info("BoMb-{} Epoch: {}, G Loss: {}, D Loss: {}".format(method, epoch, total_g_loss, total_d_loss))
        elif(ebomb):
            logger.info("eBoMb-{} Epoch: {}, G Loss: {}, D Loss: {}".format(method, epoch, total_g_loss, total_d_loss))
        else:
            logger.info("{} Epoch: {}, G Loss: {}, D Loss: {}".format(method, epoch, total_g_loss, total_d_loss))

        if (epoch % args.fid_each == 0) or (epoch == args.epochs - 1):
            save_m_dir = model_dir + '/models'
            if not (os.path.isdir(save_m_dir)):
                os.makedirs(save_m_dir)
            torch.save(model.state_dict(), '%s/G_%06i.pth' % (save_m_dir, epoch))

            count_imgs = 0
            model.eval()
            with torch.no_grad():
                sampling(model_dir + '/sample_epoch_' + str(epoch) + ".png", fixednoise, model.decoder, 64, 64, 3)
                outdir_images = model_dir + '/images'
                if not (os.path.isdir(outdir_images)):
                    os.makedirs(outdir_images)
                Nb = 11000 // args.m  # write 10K sampled images for test FID computation
                for i in tqdm(range(Nb)):
                    z = torch.randn(args.m, latent_size).cuda(device=device)
                    fake_images = model.decoder(z)
                    fake_images_np = fake_images.cpu().detach().numpy()
                    fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 3, 64, 64)
                    fake_images_np = ((fake_images_np.transpose((0, 2, 3, 1)) / 2.0 + .5) * 255).astype(np.uint8)

                    for i in range(args.m):
                        imageio.imwrite('%s/img_%06i.png' % (outdir_images, count_imgs), fake_images_np[i])
                        count_imgs += 1

            model.train()
            logger.info('wrote images to %s', outdir_images)
            torch.cuda.empty_cache()

            # Compute FID score
            dataset_paths = [outdir_images, 'fid_stats_celeba_test.npz']
            fid_score = calculate_fid_given_paths(dataset_paths, 4, device, 2048, args.num_workers)
            logger.info(f"FID score: {fid_score}")
            save_acc(csv_file, epoch, fid_score)


if __name__ == '__main__':
    main()
