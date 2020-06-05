# agwe-recipe

This recipe trains acoustic word embeddings (AWEs) and acoustically grounded word embeddings (AGWEs) on paired data
consisting of word labels (given by their character sequences) and spoken word segments.

The training objective is based on the multiview triplet loss functions
of [Wanjia et al., 2016](https://arxiv.org/pdf/1611.04496.pdf).
Hard negative sampling was added in [Settle et al., 2019](https://arxiv.org/pdf/1903.12306.pdf) to improve
training speed (similar to `src/multiview_triplet_loss_old.py`). The current version (see `src/multiview_triplet_loss.py`) uses semi-hard negative sampling [Schroff et al.](https://arxiv.org/pdf/1503.03832.pdf) (instead of hard negative sampling) and includes `obj1` from Wanjia et al. in the loss.

### Dependencies
python 3, pytorch 1.4, h5py, numpy, scipy

### Dataset (for the purpose of TTIC 31110)
Use [this link](https://forms.gle/EGuaYYW72bzs4KbK8) to download the dataset.

### Training

Edit `train_config.json` and run `train.sh`
```
./train.sh
```

### Evaluate
Edit `eval_config.json` and run `eval.sh`
```
./eval.sh
```

### Results
With the default train_config.json you should obtain the following results:

acoustic_ap= 0.79

crossview_ap= 0.75

### Acknowledgement

This repo is forked from [Shane Settle's agwe-recipe repo](https://github.com/shane-settle/agwe-recipe).

