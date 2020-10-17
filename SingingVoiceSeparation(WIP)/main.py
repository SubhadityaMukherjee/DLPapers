import argparse
from datetime import datetime as dt
import gc
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

import librosa
import soundfile as sf

# Split into train val
def train_val_split(mix_dir, inst_dir, val_rate, val_filelist_json):
    input_exts = [".wav", ".m4a", ".3gp", ".oma", ".mp3", ".mp4"]
    # Get a list of mixed songs
    X_list = sorted(
        [
            os.path.join(mix_dir, fname)
            for fname in os.listdir(mix_dir)
            if os.path.splitext(fname)[1] in input_exts
        ]
    )
    # Get a list of non mixed songs
    y_list = sorted(
        [
            os.path.join(inst_dir, fname)
            for fname in os.listdir(inst_dir)
            if os.path.splitext(fname)[1] in input_exts
        ]
    )

    filelist = list(zip(X_list, y_list))
    random.shuffle(filelist)

    # Save to json

    val_filelist = []
    if val_filelist_json is not None:
        with open(val_filelist_json, "r", encoding="utf8") as f:
            val_filelist = json.load(f)

    if len(val_filelist) == 0:
        val_size = int(len(filelist) * val_rate)
        train_filelist = filelist[:-val_size]
        val_filelist = filelist[-val_size:]
    else:
        train_filelist = [pair for pair in filelist if list(pair) not in val_filelist]

    return train_filelist, val_filelist


# Define util functions for spectrogram
def crop_center(h1, h2, concat=True):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h2_shape[3] < h1_shape[3]:
        raise ValueError('h2_shape[3] must be greater than h1_shape[3]')
    s_time = (h2_shape[3] - h1_shape[3]) // 2
    # Identify the time of the waves to find the half of them
    e_time = s_time + h1_shape[3]
    h2 = h2[:, :, :, s_time:e_time]
    # Concat the results of the split
    if concat:
        return torch.cat([h1, h2], dim=1)
    else:
        return h2

# Loop for train set
def train_inner_epoch(X_train, y_train, model, optimizer, batchsize):
    sum_loss = 0
    model.train()
    aux_crit = nn.L1Loss()
    criterion = nn.L1Loss(reduction="none")
    perm = np.random.permutation(len(X_train))
    # Set up instance loss function
    instance_loss = np.zeros(len(X_train), dtype=np.float32)
    for i in range(0, len(X_train), batchsize):
        local_perm = perm[i : i + batchsize]
        X_batch = torch.from_numpy(X_train[local_perm]).cuda()
        y_batch = torch.from_numpy(y_train[local_perm]).cuda()

        model.zero_grad()
        mask, aux = model(X_batch).cuda()
        # Crop to the center using utils
        X_batch = crop_center(mask, X_batch, False)
        y_batch = crop_center(mask, y_batch, False)
        base_loss = criterion(X_batch * mask, y_batch)
        aux_loss = aux_crit(X_batch * aux, y_batch

        # Calculate loss and gradient step

        loss = base_loss.mean() * 0.9 + aux_loss * 0.1
        loss.backward()
        optimizer.step()

        abs_diff_np = base_loss.detach().cpu().numpy()
        instance_loss[local_perm] = abs_diff_np.mean(axis=(1, 2, 3))
        sum_loss += float(loss.detach().cpu().numpy()) * len(X_batch)

    return sum_loss / len(X_train), instance_loss
