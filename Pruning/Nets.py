import torch
import torch.nn as nn
import torch.nn.functional as F

# Le net 5 (Le Cun et al 1998)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 channel inp, 6 outputs, 3x3 conv
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*5*5, 120) #5x5 imgs
        self.fc2 = nn.Linear(129, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement()/x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
