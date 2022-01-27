# BoMb-OT
Python3 implementation of the papers [On Transportation of Mini-batches: A Hierarchical Approach](https://arxiv.org/abs/2102.05912) and [Improving Mini-batch Optimal Transport via Partial Transportation](https://arxiv.org/abs/2108.09645).

Please CITE our papers whenever this repository is used to help produce published results or incorporated into other software.
```
@article{nguyen2021transportation,
      title={On Transportation of Mini-batches: A Hierarchical Approach}, 
      author={Khai Nguyen and Dang Nguyen and Quoc Nguyen and Tung Pham and Hung Bui and Dinh Phung and Trung Le and Nhat Ho},
      journal={arXiv preprint arXiv:2102.05912},
      year={2021},
}
```
```
@article{nguyen2021improving,
      title={Improving Mini-batch Optimal Transport via Partial Transportation}, 
      author={Khai Nguyen and Dang Nguyen and Tung Pham and Nhat Ho},
      journal={arXiv preprint arXiv:2108.09645},
      year={2021},
}
```
This implementation is made by Khai Nguyen and Dang Nguyen. README is on updating process.
## Requirement

* python 3.6
* pytorch 1.7.1
* torchvision
* numpy
* tqdm
* geomloss
* POT
* matplotlib
* cvxpy

## What is included?
The scalable implementation of the **batch of mini-batches** scheme and the conventional **averaging scheme** of mini-batch transportation types: optimal transport (OT), partial optimal transport (POT), unbalanced optimal transport (UOT), sliced optimal transport for:

* Deep Generative Models 
* Deep Domain Adaptation
* Approximate Bayesian Computation
* Color Transfer
* Gradient Flow


## Deep Adaptation on digits datasets (DeepDA/digits)

### Code organization
cfg.py : this file contains arguments for training.

methods.py : this file implements the training process of the deep DA.

models.py : this file contains the architecture of the generator and the classifier. 

train_digits.py: running file for deep DA.

utils.py : this file contains implementation of utility functions.

### Terminologies
--method : type of mini-batch deep DA method (jdot, jumbot, jpmbot)

--source_ds : the source dataset 

--target_ds : the target dataset

--epsilon : OT regularization coefficient for Sinkhorn algorithm

--tau : marginal penalization coefficient in UOT

--mass : fraction of masses in POT

--eta1 : weight of embedding loss 

--eta2 : weight of transportation loss 

--k : number of mini-batches

--mbsize : mini-batch size

--n_epochs : number of running epochs

--test_interval : interval of two continuous test phase

--lr : initial learning rate

--data_dir : path to the dataset

--reg : OT regularization coefficient for Sinkhorn algorithm

--bomb : Using Batch of Mini-batches

--ebomb : Using entropic Batch of Mini-batches

--breg : OT regularization coefficient for entropic Batch of Mini-batches

### Change the number of mini-batches $k$
```
bash sh/exp_mOT_change_k.sh
bash sh/exp_BoMbOT_change_k.sh
```

### Change the mini-batch size $m$
```
bash sh/exp_mOT_change_m.sh
bash sh/exp_BoMbOT_change_m.sh
```

## Deep Adaptation on Office-Home and VisDA datasets (DeepDA/office)

### Code organization
data_list.py : this file contains functions to create the dataset.

evaluate.py : this file is used to evaluate model trained on the VisDA dataset.

lr_schedule.py : this file implements the learning rate scheduler.

network.py : this file contains the architecture of the generator and the classifier. 

pre_process.py : this file implements preprocessing techniques. 

train.py : this file implements the training process for both datasets.

### Terminologies
--net : architecture type of the generator

--dset : name of the dataset

--test_interval : interval of two continuous test phase

--s_dset_path : path to the source dataset

--stratify_source : use stratify sampling

--s_dset_path : path to the target dataset

--batch_size : mini-batch size

--stop_step : number of iterations

--ot_type : type of OT loss (balanced, unbalanced, partial)

--eta1 : weight of embedding loss ($\alpha$ in equation 10)

--eta2 : weight of transportation loss ($\lambda_t$ in equation 10)

--epsilon : OT regularization coefficient for Sinkhorn algorithm

--tau : marginal penalization coefficient in UOT

--mass : fraction of masses in POT

--bomb : Using Batch of Mini-batches

--ebomb : Using entropic Batch of Mini-batches

--breg : OT regularization coefficient for entropic Batch of Mini-batches

### Train on Office-Home
```
bash sh/train_home.sh
```

### Train on VisDA
```
bash sh/train_visda.sh
```

## Deep Generative model (DeepGM)

### Code organization
Celeba_generator.py, Cifar_generator.py : these files contain the architecture of the generator on CelebA and CIFAR10 datasets, and include some self-function to compute losses of corresponding baselines. 

experiments.py : this file contains some functions for generating images.

fid_score.py: this file is used to compute the FID score.

gen_images.py: read saved models to produce 10000 images to calculate FID.

inception.py: this file contains the architecture of Inception Net V3.

main_celeba.py, main_cifar.py : running files on the corresponding datasets.

utils.py : this file contains implementation of utility functions.

### Terminologies
--method : type of OT loss (OT, UOT, POT, sliced)

--reg : OT regularization coefficient for Sinkhorn algorithm

--tau : marginal penalization coefficient in UOT

--mass : fraction of masses in POT

--k : number of mini-batches

--m : mini-batch size

--epochs : number of epochs at k = 1. The actual running epochs are calculated by multiplying this value by the value of k.

--lr : initial learning rate

--latent-size : the latent size of the generator

--datadir : path to the dataset

--L : number of projections when using slicing approach

--bomb : Using Batch of Mini-batches

--ebomb : Using entropic Batch of Mini-batches

--breg : OT regularization coefficient for entropic Batch of Mini-batches

### Train on CIFAR10 
``` 
CUDA_VISIBLE_DEVICES=0 python main_cifar.py --method POT --reg 0 --tau 1 \
    --mass 0.7 --k 2 --m 100 --epochs 100 --lr 5e-4 --latent-size 32 --datadir ./data
```

### Train on CELEBA
``` 
CUDA_VISIBLE_DEVICES=0 python main_celeba.py --method POT --reg 0 --tau 1 \
    --mass 0.7 --k 2 --m 200 --epochs 100 --lr 5e-4 --latent-size 32 --datadir ./data
```

## Gradient Flow (GradientFlow)
```
python main.py
```

## Color Transfer (Color Transfer)
```
python main.py  --m=100 --T=10000 --source images/s1.bmp --target images/t1.bmp --cluster

```
### Terminologies

--k : number of mini-batches

--m : the size of mini-batches

--T : the number of steps

--cluster: K-means clustering to compress images

--palette: show the color palette

--source: Path to the source image

## Acknowledgment
The structure of DeepDA is largely based on [JUMBOT](https://github.com/kilianFatras/JUMBOT) and [ALDA](https://github.com/ZJULearning/ALDA).  The structure of ABC is largely based on [SlicedABC](https://github.com/kimiandj/slicedwass_abc). We are very grateful for their open sources.
