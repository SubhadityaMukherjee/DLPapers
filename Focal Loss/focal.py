# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch.nn.functional as F
import torch.nn as nn
import torch
from fastai import *
from fastai.vision import *
import torch.onnx
from torch.autograd import Variable
import scipy.stats as stats


# %%
import os

os.environ["TORCH_HOME"] = "/media/subhaditya/DATA/COSMO/Datasets-Useful"


# %%
path = untar_data(URLs.CIFAR_100)
# path = Path('data/CCSN/')


# %%
data = (
    (
        ImageList.from_folder(path)
        .split_by_rand_pct()
        .label_from_folder()
        .transform(get_transforms(), size=128)
    )
    .databunch(bs=64)
    .normalize(imagenet_stats)
)


# %%
data.show_batch(4)


# %%
data.c


# %%
data

# %% [markdown]
# ![foc](focal.png)

# %%


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N, C, H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt*Variable(at)

        loss = -1*(1-pt)**self.gamma*logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum


# %%
learn = cnn_learner(
    data,
    models.resnet34,
    metrics=[accuracy, error_rate],
    opt_func=AdamW,
    callback_fns=ShowGraph,
    loss_func=FocalLoss()).to_fp16()


# %%
learn.lr_find()
learn.recorder.plot()


# %%
learn.fit_one_cycle(10, 3e-2,wd = 10e-4)


# %%
learn.fit_one_cycle(10, 1e-4)
