# agwe-recipe-transformer

This recipe trains acoustic word embeddings (AWEs) and acoustically grounded word embeddings (AGWEs) on paired data
consisting of word labels (given by their character sequences) and spoken word segments, where the original implementation is based on RNNs and 
the one we implemented is based on transformer encoders.

The training objective is based on the multiview triplet loss functions
of [Wanjia et al., 2016](https://arxiv.org/pdf/1611.04496.pdf).
Hard negative sampling was added in [Settle et al., 2019](https://arxiv.org/pdf/1903.12306.pdf) to improve
training speed (similar to `src/multiview_triplet_loss_old.py`). The current version (see `src/multiview_triplet_loss.py`) uses semi-hard negative sampling [Schroff et al.](https://arxiv.org/pdf/1503.03832.pdf) (instead of hard negative sampling) and includes `obj1` from Wanjia et al. in the loss.

### Dependencies
python 3, pytorch 1.4, h5py, numpy, scipy

### Dataset (for the purpose of TTIC 31110)
Use [this link](https://forms.gle/EGuaYYW72bzs4KbK8) to download the dataset. (Caveat: you must have the appropriate permissions.)

### Training

In general: Edit a `train_config.json` file and run a `train.sh` file.
```
./train.sh
```

For RNNs: Edit `train_config_RNN.json` file and run `trainRNN.sh` file.

For transformer encoders: Edit `train2_config.json` and run `train2.sh` file.

#### Difference between `train.json` and `train2.json`

There are two versions of the training file for the transformer encoder: this is because I tried a couple of implementations
during the project. The first file `train.json` uses an implementation where a normalizing constant for each utterance that is the length
of the longest utterance in the batch. This is presumably not what we want, so I changed the implementation to correct for this 
variable normalizing constant, which is the implementation you will find in `train2.json`. I will clean this up later, but
here is a note for now.

I decided to keep both in the repository so people are aware this may be a problem and that they could try both.

TO DO: Change the nomenclature to make this more apparent.

### Evaluate
Edit `eval_config.json` and run `eval.sh`
```
./eval.sh
```

TO DO: add a `eval_config.json` file for RNN and transformer for ease.

### Results

#### RNN
With the default `train_config_RNN.json` you should obtain the following results:

acoustic_ap= 0.79

crossview_ap= 0.75

#### Transformer encoder
With the `train2_config.json` you should obtain the following results with the following parameters:

phoneme view: 4 layers, 16 heads, d_{model} 256, d_{feed forward} 1024

acoustic view: 6 layers, 16 heads, d_{model} 256, d_{feed forward} 1024



acoustic_ap= 0.71

crossview_ap= 0.55

### Acknowledgement

This repo is forked from [Shane Settle's agwe-recipe repo](https://github.com/shane-settle/agwe-recipe). Thank you so much for lying down the foundation
for us to build upon!

