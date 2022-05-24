## Partial Domain Adaptation on Office-Home

### Code organization
data_list.py : this file contains functions to create the dataset.

lr_schedule.py : this file implements the learning rate scheduler.

my_loss.py: this file implements loss functions for baselimes.

network.py : this file contains the architecture of the generator and the classifier. 

run_BA3US.py : this file implements the BA3US method.

run_mOT.py : this file implements mini-batch methods.

utils.py : this file contains implementation of utility functions.

### Terminologies
--net : architecture type of the generator

--dset : name of the dataset

--test_interval : interval of two continuous test phase

--s : source

--t : target

--batch_size : mini-batch size

--k : number of mini-batches

--ot_type : type of OT loss (balanced, unbalanced, partial)

--eta1 : weight of embedding loss ($\alpha$ in equation 10)

--eta2 : weight of transportation loss ($\lambda_t$ in equation 10)

--eta3 : weight of transfor loss (only applied for UOT)

--epsilon : OT regularization coefficient for Sinkhorn algorithm

--tau : marginal penalization coefficient in UOT

--mass : fraction of masses in POT

### Run the BA3US method
```
bash sh/exp_BA3US.sh
```

### Run mini-batch methods
```
bash sh/exp_mOT.sh
bash sh/exp_mUOT.sh
bash sh/exp_mPOT.sh
```

For more detailed settings, please check them in our papers.