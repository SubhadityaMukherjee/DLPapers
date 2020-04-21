# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os

from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from fastai.vision import *
from IPython import get_ipython

# %%
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


os.environ["TORCH_HOME"] = "/media/subhaditya/DATA/COSMO/Datasets-Useful"


# %%
path = untar_data(URLs.CIFAR)


# %%
data = (
    (
        ImageList.from_folder(path)
        .split_by_rand_pct()
        .label_from_folder()
        .transform(get_transforms(), size=32)
    )
    .databunch(bs=64)
    .normalize(imagenet_stats)
)


# %%
data.show_batch(4)


# %%
data.c


# %%
learn = None
gc.collect()

# %%

class UpSample(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(UpSample, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.ReLU(True)(x)
        return x
# %%
class DownSample(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.ReLU(True)(x)
        return x
# %%
class SameRes(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(SameRes, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.ReLU(True)(x)
        return x
# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.channels = 128
        self.kernel_size = 3

        self.layers = 8
        self.scales = 5

        self.node_ops = nn.ModuleList()

        self.start_node = SameRes(3, self.channels)

        self.fc = nn.Linear(self.channels,10)

        for layer in range(self.layers):
            self.node_ops.append(nn.ModuleList()) # add list for each layer
            self.node_ops[layer] = nn.ModuleList() # list for each scale

            if layer == 0:
                # print(self.node_ops)
                for i in range(self.scales):
                    self.node_ops[layer].append(nn.ModuleList())

                    node = DownSample(self.channels,self.channels)
                    self.node_ops[layer][i].append(node)
            else:
                for i in range(self.scales):
                    self.node_ops[layer].append(nn.ModuleList())

                    node = SameRes(self.channels,self.channels)
                    self.node_ops[layer][i].append(node)
                    if i == 0:
                        self.node_ops[layer][i].append(
                            UpSample(self.channels,self.channels))
                    elif i == self.scales -1:
                        self.node_ops[layer][i].append(
                            DownSample(self.channels,self.channels))
                        if layer == self.layers-1:
                            self.node_ops[layer][i].append(
                                DownSample(self.channels,self.channels))
                    else:
                        self.node_ops[layer][i].append(
                            DownSample(self.channels,self.channels))
                        self.node_ops[layer][i].append(
                            UpSample(self.channels,self.channels))
                        if layer == self.layers-1:
                            self.node_ops[layer][i].append(
                                DownSample(self.channels,self.channels))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)
                torch.nn.init.constant(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0,0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        node_activ = [[[] for i in range(self.scales)] for j in range(self.layers)]
        out = self.start_node(x)
        for layer in range(self.layers):
            if layer == 0:
                for i in range(self.scales):
                    if i == 0:
                        node_activ[layer][i] = self.node_ops[layer][i][0](out)
                    else:
                        node_activ[layer][i] = self.node_ops[layer][i][0](node_activ[layer][i-1])
            else:
                for i in range(self.scales):
                    if i == 0:
                        t1 = (node_activ[layer-1][i])
                        t2 = self.node_ops[layer][i][1](node_activ[layer-1][i+1])
                        t = self.node_ops[layer][i][0](t1 + t2)
                        node_activ[layer][i] = t
                    elif i == self.scales-1:
                        t1 = (node_activ[layer-1][i])
                        t2 = self.node_ops[layer][i][1](node_activ[layer-1][i-1])

                        if layer == self.layers-1:
                            t3 = self.node_ops[layer][i][2](node_activ[layer][i-1])
                            t = self.node_ops[layer][i][0](t1 + t2  + t3)
                        else:
                            t = self.node_ops[layer][i][0](t1 + t2)
                        node_activ[layer][i] = t
                    else:
                        t1 = (node_activ[layer-1][i])
                        t2 = self.node_ops[layer][i][2](node_activ[layer-1][i+1])
                        t3 = self.node_ops[layer][i][1](node_activ[layer-1][i-1])
                        if layer == self.layers-1:
                            t4 = self.node_ops[layer][i][3](node_activ[layer][i-1])
                            t = self.node_ops[layer][i][0](t1 + t2 + t3 + t4)
                        else:
                            t = self.node_ops[layer][i][0](t1 + t2 + t3)
                        node_activ[layer][i] = t

        out = node_activ[-1][-1]
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


# %%

learn = Learner(
    data, Net(), metrics=[accuracy], opt_func=AdamW, callback_fns=ShowGraph
).to_fp16()

# %%
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# %%
learn.unfreeze()
learn.fit_one_cycle(10, slice(4e-4),wd= 10e-4)


# %%
learn.show_results()
