## Deep Domain Adaptation on Office-Home and VisDA datasets

### Code organization
data_list.py : this file contains functions to create the dataset.

evaluate.py : this file is used to evaluate model trained on the VisDA dataset.

lr_schedule.py : this file implements the learning rate scheduler.

network.py : this file contains the architecture of the generator and the classifier. 

plot_embeddings.py: this file visualizes TSNE embeddings.

pre_process.py : this file implements preprocessing techniques.

train.py : this file implements the conventional DA training process for both datasets.

train_visda.py : this file improves the conventional DA scheme on VisDA by handling the floating-point underflow problem.

train_ts.py : this file contains the two-stage implementation.

### Terminologies
--net : architecture type of the generator

--dset : name of the dataset

--test_interval : interval of two continuous test phase

--s_dset_path : path to the source dataset

--stratify_source : use stratify sampling

--t_dset_path : path to the target dataset

--batch_size : mini-batch size

--k : number of mini-batches

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

### Run two-stage implementation
```
bash sh/train_ts.sh
```

### Visualize TSNE embeddings on VisDA
```
bash sh/plot_embeddings_visda.sh
```

For more detailed settings, please check them in our papers.