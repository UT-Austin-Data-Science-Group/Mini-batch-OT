# BoMb-OT
Official Python3 implementations of ICML papers [On Transportation of Mini-batches: A Hierarchical Approach](https://arxiv.org/abs/2102.05912) and [Improving Mini-batch Optimal Transport via Partial Transportation](https://arxiv.org/abs/2108.09645).

Please CITE our papers whenever this repository is used to help produce published results or incorporated into other software.
```
@InProceedings{nguyen2021transportation,
    title={On Transportation of Mini-batches: A Hierarchical Approach}, 
    author={Khai Nguyen and Dang Nguyen and Quoc Nguyen and Tung Pham and Hung Bui and Dinh Phung and Trung Le and Nhat Ho},
    booktitle={Proceedings of the 39th International Conference on Machine Learning},
    year={2022},
}
```
```
@InProceedings{nguyen2021improving,
    title={Improving Mini-batch Optimal Transport via Partial Transportation}, 
    author={Khai Nguyen and Dang Nguyen and Tung Pham and Nhat Ho},
    booktitle = {Proceedings of the 39th International Conference on Machine Learning},
    year={2022},
}
```
This implementation is made by Khai Nguyen and Dang Nguyen. README is on updating process.

## Requirements
The code is implemented with Python (3.9.7) and Pytorch (1.10.1).

To install the required python packages, run
```
pip install -r requirements.txt
```

## What is included?
The scalable implementation of the **batch of mini-batches** scheme and the conventional **averaging scheme** of mini-batch transportation types: optimal transport (OT), partial optimal transport (POT), unbalanced optimal transport (UOT), and sliced optimal transport for:

* Deep Generative Models 
* Deep Domain Adaptation
* Approximate Bayesian Computation
* Color Transfer
* Gradient Flow

## Acknowledgment
The structure of DeepDA is largely based on [JUMBOT](https://github.com/kilianFatras/JUMBOT) and [ALDA](https://github.com/ZJULearning/ALDA).  The structure of ABC is largely based on [SlicedABC](https://github.com/kimiandj/slicedwass_abc). We are very grateful for their open sources.
