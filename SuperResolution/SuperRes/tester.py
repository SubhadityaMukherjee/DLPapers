import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from math import log10

criterion = nn.MSELoss()
def test(model, device, test_loader):
    avg_psnr = 0 #psnr is a metric that shows how well the image compares
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            mse = criterion(output, target)
            psnr = 10*log10(1/mse.item())
            avg_psnr += psnr

    avg_psnr /= len(test_loader.dataset)
    print(f"Avg psnr: {avg_psnr} dB")
