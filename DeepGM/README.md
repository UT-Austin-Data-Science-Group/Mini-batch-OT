## Deep Generative model (DeepGM)

### Code organization
Celeba_generator.py, Cifar_generator.py, Mnist_generator.py : these files contain the architecture of the generator on CelebA, CIFAR10, and MNIST datasets, including some self-function to compute losses of corresponding baselines. 

experiments.py : this file contains some functions for generating images.

fid_score.py: this file is used to compute the FID score.

gen_images.py: read saved models to produce 10000 images to calculate FID.

inception.py: this file contains the architecture of Inception Net V3.

main_celeba.py, main_cifar.py, main_mnist.py : running files on the corresponding datasets.

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
CUDA_VISIBLE_DEVICES=0 python main_cifar.py --method POT --reg 0 --tau 1 --mass 0.7 --k 2 --m 100 --epochs 100 --lr 5e-4 --latent-size 32 --datadir ./data
```

### Train on CELEBA
``` 
CUDA_VISIBLE_DEVICES=0 python main_celeba.py --method POT --reg 0 --tau 1 --mass 0.7 --k 2 --m 200 --epochs 100 --lr 5e-4 --latent-size 32 --datadir ./data
```

For more detailed settings, please check them in our papers.