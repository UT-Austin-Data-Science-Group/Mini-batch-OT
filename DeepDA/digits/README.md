## Deep Domain Adaptation on digits datasets

### Code organization
cfg.py : this file contains arguments for training.

methods.py : this file implements the training process of the deep DA.

models.py : this file contains the architecture of the generator and the classifier. 

plot_embeddings.py: this file visualizes TSNE embeddings.

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
bash sh/exp_mUOT_change_k.sh
bash sh/exp_BoMbUOT_change_k.sh
```

### Change the mini-batch size $m$
```
bash sh/exp_mOT_change_m.sh
bash sh/exp_BoMbOT_change_m.sh
bash sh/exp_mUOT_change_m.sh
bash sh/exp_BoMbUOT_change_m.sh
```

### Compare between mini-batch methods
```
bash sh/exp_mOT.sh
bash sh/exp_mUOT.sh
bash sh/exp_mPOT.sh
```
For more detailed settings, please check them in our papers.