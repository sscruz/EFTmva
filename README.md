# EFTmva: multivariate discriminants for EFT

## Set-up instructions

```
conda create -n eft_mva python=python=3.10.8
conda activate eft_mva

# get pytorch https://pytorch.org/get-started/locally/

pip install tqdm
pip install matplotlib
pip install pyyaml
pip install uproot awkward

```

## Instructions to run

`train.py` allows the user to train a neural network to discriminate between the SM and either (a) a specific BSM point (specified with the option `--bsm-point`) or (b) events reweighed by a given term of the quadratic expansion (specified with the option `--term`)

An example of usage would be 

```
python train.py --term ctg_ctg --files /path/to/files/*.root
```

The data is provided through `rootfiles` that are converted to `pytorch` tensors. The tensors are stored in separate files for the input features, the SM weights and the BSM weights, the latter with a name specific to the BSM point / term chosen. If any of these need to be reproduced, the associated file can be erased or pass the argument `--forceRebuild`
