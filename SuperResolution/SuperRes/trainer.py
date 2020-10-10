import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils
import torch.optim as optim


def train(args, device, train_loader,model,  epoch, optimizer, criterion):
    epoch_loss = 0
    device = torch.device("cuda") # Sending to GPU
    for batch_idx, batch in tqdm(enumerate(train_loader, 1)):
        input, target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad() # zero gradients
        loss = criterion(model(input), target) # calc loss
        epoch_loss += loss.item()
        loss.backward() #backprop
        optimizer.step()

        print(f"Iteration: {batch_idx}, Loss: {loss} ")
    print(f"Avg epoch_loss: {epoch/len(train_loader)}")


