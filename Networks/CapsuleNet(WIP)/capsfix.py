#%%
import os

import netron
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.optim as optim
from fastai import *
from fastai.vision import *
from torch.autograd import Variable

os.environ["TORCH_HOME"] = "/media/subhaditya/DATA/COSMO/Datasets-Useful"

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
#%%

path = untar_data(URLs.CIFAR_100)

data = (
    (
        ImageList.from_folder(path)
        .split_by_rand_pct()
        .label_from_folder()
        .transform(get_transforms(), size=28)
    )
    .databunch(bs=64)
    .normalize(imagenet_stats)
)
#%%
data.show_batch(4)
#%%
data.c
#%%
data
# %%

#%%
learn = None
gc.collect()
#%%
learn = Learner(
    data,
    Model(),
    metrics=[accuracy, error_rate],
    opt_func=AdamW,
    callback_fns=ShowGraph,
)
#%%

learn.lr_find()
learn.recorder.plot()
#%%
learn.fit_one_cycle(10, 1e-2)
#%%
