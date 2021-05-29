import argparse
import os

import syft as sy  # PySyft
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from tester import *
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm
from trainer import *

os.environ["TORCH_HOME"] = "~/Documents/datasets/"

# Arguments for PySyft
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")  # person1
alice = sy.VirtualWorker(hook, id="alice")  # person2

# Allowing arguments for direct execution from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="folder for custom training", default="")
parser.add_argument(
    "--arch",
    default="resnet18",
    help="""Choose any model from pytorch. Or input "my" for taking a model from model.py """,
)
parser.add_argument("--weight-decay", default=1e-4, help="weight decay coefficient")
parser.add_argument("--resume", default=False, help="Resume training from a checkpoint")
parser.add_argument(
    "--pretrained",
    default=True,
    help="If part of the standard datasets, downloaded pretrained weights",
)
parser.add_argument("--batch-size", type=int, default=128, help="input batch size")
parser.add_argument("--test-batch-size", type=int, default=1000)

parser.add_argument("--epochs", type=int, default=20, help="no of epochs to train for")

parser.add_argument("--lr", type=float, default=0.01, help="Base learning rate")

parser.add_argument(
    "--max_lr", type=float, default=0.1, help="Max learning rate for OneCycleLR"
)


parser.add_argument(
    "--dry-run", action="store_true", default=False, help="quickly check a single pass"
)

parser.add_argument("--seed", type=int, default=100, help="torch random seed")

parser.add_argument(
    "--log-interval", type=int, default=20, help="interval to show results"
)

parser.add_argument(
    "--save-model",
    action="store_true",
    default=True,
    help="Choose if model to be saved or not",
)

parser.add_argument(
    "--save_path", default="models/model.pt", help="Choose model saved filepath"
)

args = parser.parse_args()

# Setting params

torch.manual_seed(args.seed)
device = torch.device("cuda")
kwargs = {"batch_size": args.batch_size}
kwargs.update(
    {
        "num_workers": 1,
        "pin_memory": True,
    }
)

# Defining batch transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Loading dataset
train_loader = sy.FederatedDataLoader(  # <-- this is now a FederatedDataLoader
    datasets.MNIST(
        "/home/eragon/Documents/datasets",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ).federate(
        (bob, alice)
    ),  # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
    shuffle=True,
    **kwargs,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "/home/eragon/Documents/datasets",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    shuffle=True,
    **kwargs,
)

# Loading model

if args.arch == "my":
    from Nets import *

    model = Net().to(device)
    print("Using custom architecture")
else:
    if args.pretrained:
        print(f"Using pretrained {args.arch}")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"Not using pretrained {args.arch}")
        model = models.__dict__[args.arch]()

model = model.to(device)
print(model)
start_epoch = 1
if args.resume:
    loc = "cuda"
    checkpoint = torch.load(args.save_path, map_location=loc)
    model.load_state_dict(checkpoint["state_dict"])
    mode.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]

    print(f"Done loading pretrained, Start epoch: {checkpoint['epoch']}")

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)

for epoch in tqdm(range(start_epoch, args.epochs + 1)):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

if args.save_model:
    torch.save(model.state_dict(), args.save_path)
