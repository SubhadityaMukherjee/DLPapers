import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # Setting model to train
    device = torch.device("cuda")  # Sending to GPU
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        model.send(data.location)  # Send to location (worker)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Reset grads
        output = model(data)  # Passing batch through model

        # Will need to change everytime. Loss
        loss = F.nll_loss(output, target )

        loss.backward()  # Backprop
        optimizer.step()  # Pass through optimizer
        model.get()  # Get it back

        if batch_idx % args.log_interval == 0:
            loss = loss.get()
            # print(loss.item())
            if args.dry_run:
                break
