from optparse import OptionParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define all the required options
parser = OptionParser()

parser.add_option(
    "-e",
    "--epochs",
    dest="epochs",
    default=200,
    type="int",
    help="number of epochs (default: 80)",
)

parser.add_option(
    "-b",
    "--batch-size",
    dest="batch_size",
    default=100,
    type="int",
    help="batch size (default: 16)",
)
parser.add_option(
    "--df",
    "--disp_freq",
    dest="disp_freq",
    default=100,
    type="int",
    help="frequency of displaying the training results (default: 100)",
)
parser.add_option(
    "--vf",
    "--val_freq",
    dest="val_freq",
    default=600,
    type="int",
    help="run validation for each <val_freq> iterations (default: 2000)",
)
parser.add_option(
    "-j",
    "--workers",
    dest="workers",
    default=0,
    type="int",
    help="number of data loading workers (default: 16)",
)
# For data
parser.add_option(
    "--dn",
    "--data_name",
    dest="data_name",
    default="fashion_mnist",
    help="mnist, fashion_mnist, t_mnist, c_mnist, cub (default: mnist)",
)

parser.add_option(
    "--ih",
    "--img_h",
    dest="img_h",
    default=28,
    type="int",
    help="input image height (default: 28)",
)
parser.add_option(
    "--iw",
    "--img_w",
    dest="img_w",
    default=28,
    type="int",
    help="input image width (default: 28)",
)
parser.add_option(
    "--ic",
    "--img_c",
    dest="img_c",
    default=1,
    type="int",
    help="number of input channels (default: 1)",
)

parser.add_option(
    "--ni",
    "--num_iterations",
    dest="num_iterations",
    default=3,
    type="int",
    help="number of routing iterations (default: 3)",
)
parser.add_option(
    "--nc",
    "--num_classes",
    dest="num_classes",
    default=10,
    type="int",
    help="number of classes (default: 10)",
)

# For loss
parser.add_option(
    "--mp",
    "--m_plus",
    dest="m_plus",
    default=0.9,
    type="float",
    help="m+ parameter (default: 0.9)",
)
parser.add_option(
    "--mm",
    "--m_minus",
    dest="m_minus",
    default=0.1,
    type="float",
    help="m- parameter (default: 0.1)",
)
parser.add_option(
    "--la",
    "--lambda_val",
    dest="lambda_val",
    default=0.5,
    type="float",
    help="Down-weighting parameter for the absent class (default: 0.5)",
)
parser.add_option(
    "--al",
    "--alpha",
    dest="alpha",
    default=0.0005,
    type="float",
    help="Regularization coefficient to scale down the reconstruction loss (default: 0.0005)",
)

parser.add_option(
    "--sd",
    "--save-dir",
    dest="save_dir",
    default="./save",
    help="saving directory of .ckpt models (default: ./save)",
)

# For optimizer
parser.add_option(
    "--lr",
    "--lr",
    dest="lr",
    default=0.001,
    type="float",
    help="learning rate (default: 0.001)",
)
parser.add_option(
    "--beta1",
    "--beta1",
    dest="beta1",
    default=0.9,
    type="float",
    help="beta 1 for Adam optimizer (default: 0.9)",
)

# For a Deeper CapsNet
parser.add_option(
    "--fe",
    "--feature_extractor",
    dest="feature_extractor",
    default="inception",
    help="densenet, resnet, inception (default: resnet)",
)

# # Initial Convolution layer
parser.add_option(
    "--f_conv1",
    "--f_conv1",
    dest="f_conv1",
    default=256,
    type="int",
    help="number of filters for the conv1 layer (default: 256)",
)
parser.add_option(
    "--k_conv1",
    "--k_conv1",
    dest="k_conv1",
    default=9,
    type="int",
    help="filter size of the conv1 layer (default: 9)",
)
parser.add_option(
    "--s_conv1",
    "--s_conv1",
    dest="s_conv1",
    default=2,
    type="int",
    help="filter size of the conv1 layer (default: 1)",
)

# # Primary capsule layer
parser.add_option(
    "--f_prim",
    "--f_prim",
    dest="f_prim",
    default=256,
    type="int",
    help="number of filters for the primary capsule layer (default: 256)",
)
parser.add_option(
    "--k_prim",
    "--k_prim",
    dest="k_prim",
    default=9,
    type="int",
    help="filter size of the primary capsule layer (default: 9)",
)
parser.add_option(
    "--s_prim",
    "--s_prim",
    dest="s_prim",
    default=2,
    type="int",
    help="stride of the Primary capsule layer (default: 2)",
)

parser.add_option(
    "--pcd",
    "--primary_cap_dim",
    dest="primary_cap_dim",
    default=8,
    type="int",
    help="dimension of each primary capsule (default: 8)",
)
parser.add_option(
    "--dcd",
    "--digit_cap_dim",
    dest="digit_cap_dim",
    default=16,
    type="int",
    help="dimension of each digit capsule (default: 16)",
)

parser.add_option(
    "--ws",
    "--weight_share",
    dest="share_weight",
    default=True,
    help="whether to share W among child capsules of the same type (default: True)",
)

# For decoder
parser.add_option(
    "--ad",
    "--add_decoder",
    dest="add_decoder",
    default=True,
    help="whether to use decoder or not",
)
parser.add_option(
    "--h1",
    "--h1",
    dest="h1",
    default=512,
    type="int",
    help="number of hidden units of the first hidden layer (default: 512)",
)
parser.add_option(
    "--h2",
    "--h2",
    dest="h2",
    default=1024,
    type="int",
    help="number of hidden units of the first hidden layer (default: 1024)",
)

# For loading
parser.add_option(
    "--lp",
    "--load_model_path",
    dest="load_model_path",
    default="./models",
    help="path to load a .ckpt model",
)

options, _ = parser.parse_args()
