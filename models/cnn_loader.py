import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.25)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.25)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.25)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool2(x)

        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool3(x)

        x = x.view(-1, 128 * 3 * 3)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model(filepath):

    model = CNNModel()

    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    model.eval()

    return model