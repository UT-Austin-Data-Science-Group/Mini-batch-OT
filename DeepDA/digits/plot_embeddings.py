import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from train_digits import *
import torch
from models import weights_init, USPS_generator, SVHN_generator
from tqdm import tqdm
from sklearn.manifold import TSNE

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if (torch.cuda.is_available()):
        torch.cuda.manual_seed(seed)

def feature_extraction(model, dataloader):
    embed_list = []
    label_list = []
    
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img = img.to(args.device)
            embed = model(img)
            label_list.append(label)
            embed_list.append(embed)
    
    return torch.cat(embed_list).cpu().numpy(), torch.cat(label_list).cpu().numpy()

# For reproduction
SEED = 1980
seed_everything(SEED)

args = {'source_ds': 'svhn', 
        'target_ds': 'mnist', 
        'k': 1,
        'lr': 4e-4,
        'mass': 0.85,
        'tau': 1.0,
        'data_dir': './data', 
        'mbsize': 500, 
        'epoch': 100,
        'num_workers': 8, 
        'device': 'cuda:0'}

args = dotdict(args)
batch_size = args.k * args.mbsize

# Load dataset
source_train_dl, target_train_dl, source_test_dl, target_test_dl = get_dataloader(
    args.source_ds, args.target_ds, args.data_dir, batch_size, args.num_workers)

# Create generator
if args.source_ds == 'svhn' or args.target_ds == 'svhn':
    model_g = SVHN_generator().to(args.device).apply(weights_init)
elif args.source_ds == 'usps' or args.target_ds == 'usps':
    model_g = USPS_generator().to(args.device).apply(weights_init)

fig = plt.figure(figsize=(20, 5))
TICK_SIZE = 14
TITLE_SIZE = 20
MARKER_SIZE = 50
NUM_SAMPLES = 2000
title_list = ['m-OT', 'm-UOT', 'm-POT']
method_list = ['jdot', 'jumbot', 'jpmbot']
epsilon_list = [0.0, 0.1, 0.1]
subplot_idx = 131

for idx in range(3):
    ax = fig.add_subplot(subplot_idx)
    args.method = method_list[idx]
    args.epsilon = epsilon_list[idx]
    title = title_list[idx]
    
    # Load checkpoint 
    checkpoint_path = 'snapshot/{}_{}_to_{}_k{}_mbsize{}_lr{}_epsilon{}_mass{}_tau{}/best_model.pth'.format(
        args.method, args.source_ds, args.target_ds, args.k, 
        args.mbsize, args.lr, args.epsilon, args.mass, args.tau)
    state_dict = torch.load(checkpoint_path, map_location=args.device)
    print(f"Accuracy of {args.method} is {state_dict['accuracy']}")
    model_g.load_state_dict(state_dict['model_g'])

    # Extract latent feature
    source_embed, source_label = feature_extraction(model_g, source_test_dl)
    target_embed, target_label = feature_extraction(model_g, target_test_dl)

    # T-SNE plot
    combined_imgs = np.vstack([source_embed[0:NUM_SAMPLES, :], target_embed[0:NUM_SAMPLES, :]])
    combined_labels = np.concatenate([source_label[0:NUM_SAMPLES], target_label[0:NUM_SAMPLES]])
    combined_labels = combined_labels.astype('int')
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    source_only_tsne = tsne.fit_transform(combined_imgs)
    ax.scatter(source_only_tsne[:NUM_SAMPLES,0], source_only_tsne[:NUM_SAMPLES,1], c=combined_labels[:NUM_SAMPLES],
                s=MARKER_SIZE, alpha=0.5, marker='o', cmap=cm.jet, label='source')
    ax.scatter(source_only_tsne[NUM_SAMPLES:,0], source_only_tsne[NUM_SAMPLES:,1], c=combined_labels[NUM_SAMPLES:],
                s=MARKER_SIZE, alpha=0.5, marker='+', cmap=cm.jet, label='target')
    ax.set_xlim(-125, 125)
    ax.set_ylim(-125, 125)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(loc='upper right')
    subplot_idx += 1

plt.savefig(f'{args.source_ds.upper()}2{args.target_ds.upper()}.png', bbox_inches='tight')
plt.savefig(f'{args.source_ds.upper()}2{args.target_ds.upper()}.pdf', bbox_inches='tight')
plt.close()