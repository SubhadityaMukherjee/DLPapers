import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from trainer import *
from tester import *
import torchvision.models as models
import torch.nn.utils.prune as prune
import os
os.environ["TORCH_HOME"] = "~/Desktop/Datasets/"

# Allowing arguments for direct execution from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--data', help = "folder for custom training", default = "")
parser.add_argument('--arch', default = 'resnet18', help= '''Choose any model from pytorch. Or input "my" for taking a model from model.py ''')
parser.add_argument("--weight-decay", default = 1e-4, help = "weight decay coefficient")
parser.add_argument("--resume", default = False, help = "Resume training from a checkpoint")
parser.add_argument("--pretrained", default = True, help = "If part of the standard datasets, downloaded pretrained weights")
parser.add_argument('--batch-size', type = int, default = 128, help = 'input batch size')
parser.add_argument(
    "--test-batch-size", type = int, default = 1000
)

parser.add_argument(
    "--epochs", type = int, default = 20, help = "no of epochs to train for"
)

parser.add_argument(
    "--lr", type = float, default = 0.01, help = "Base learning rate"
)

parser.add_argument(
    "--max_lr", type = float, default = 0.1, help = "Max learning rate for OneCycleLR"
)


parser.add_argument(
    "--dry-run", action = 'store_true', default = False, help = 'quickly check a single pass'
)

parser.add_argument(
    "--seed", type = int, default = 100, help = "torch random seed"
)

parser.add_argument(
    "--log-interval", type = int, default = 20, help = "interval to show results"
)

parser.add_argument(
    "--save-model", action = 'store_true', default = True, help = "Choose if model to be saved or not"
)

parser.add_argument("--save_path", default = "models/model.pt", help = "Choose model saved filepath")

args = parser.parse_args()

# Setting params

torch.manual_seed(args.seed)
device = torch.device("cuda")
kwargs = {'batch_size':args.batch_size}
kwargs.update(
    {'num_workers':8,
     'pin_memory':True,
     'shuffle': True

    }
)

# Defining batch transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
    ]
)

# Loading dataset
"""
train_data = datasets.CIFAR10("~/Desktop/Datasets/",train = True, transform = transform)

test_data = datasets.CIFAR10("~/Desktop/Datasets/",train = False, transform = transform)

train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
"""
# Loading model

if args.arch == "my":
    from Nets import *
    model = Net().to(device)
    print("Using custom architecture")
else:
    if args.pretrained:
        print(f"Using pretrained {args.arch}")
        model = models.__dict__[args.arch](pretrained = True)
    else:
        print(f"Not using pretrained {args.arch}")
        model = models.__dict__[args.arch]()

model = model.to(device)
print(model)

# Unpruned model
print("\n\# Unpruned module")
module = model.conv1
print("\# unpruned params")
print(list(module.named_parameters()))
print("\# Unpruned buffers")
print(list(module.named_buffers()))

# Prune a single module (here conv1)

print("# Prune a single module (here conv1) by 30%")
prune.random_unstructured(module, name = "weight", amount=0.3)
print("# conv1 pruned params")
print(list(module.named_parameters()))
print("# conv1 pruned buffers")
print(list(module.named_buffers()))

# Prune weight using L1 norm and 3 smallest entries

prune.l1_unstructured(module, name = "bias", amount=3)
print("# conv1 pruned bias params")
print(list(module.named_parameters()))
print("# conv1 pruned bias buffers")
print(list(module.named_buffers()))
print("# Forward pre hooks")
print(module._forward_pre_hooks)

# Iterative pruning (Prune multiple times in series, zeros out 50%)

prune.ln_structured(module, name = "weight", amount = 0.5, n = 2, dim = 0)

# weights pruned 
print(module.weight)
for hook in module._forward_pre_hooks.values():
    if hook._tensor_name == "weight":  # select out the correct hook
        break

# pruning history
print(list(hook))

# Remove reparamatrization
prune.remove(module, 'weight')
print(list(module.named_parameters()))

# Prune multiple based on type (20% in conv and 40% in linear)

model = Net().to(device)
for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers 
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)
    # prune 40% of connections in all linear layers 
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)

print(dict(model.named_buffers()).keys())

# Global pruning

model = Net().to(device)

parameters_to_prune  = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
)

# Check global sparsity

print("global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
            + torch.sum(model.fc2.weight == 0)
            + torch.sum(model.fc3.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
            + model.fc2.weight.nelement()
            + model.fc3.weight.nelement()
        )
    )
)

"""
start_epoch = 1
if args.resume:
    loc = "cuda:0"
    checkpoint = torch.load(args.save_path, map_location = loc)
    model.load_state_dict(checkpoint['state_dict'])
    mode.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

    print(f"Done loading pretrained, Start epoch: {checkpoint['epoch']}")

optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay =
                        args.weight_decay)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr =
                                       args.max_lr,steps_per_epoch =
                                          len(train_loader), epochs = 10)

for epoch in tqdm(range(start_epoch, args.epochs+1)):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

if args.save_model:
    torch.save(model.state_dict(), args.save_path)
"""

