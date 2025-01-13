import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPShallow(nn.Module):
    def __init__(self):
        super(MLPShallow, self).__init__()
        self.input_layer = nn.Linear(28 * 28, 512)
        self.hidden_layer_1 = nn.Linear(512, 256)
        self.hidden_layer_2 = nn.Linear(256, 128)
        self.hidden_layer_3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        x = F.relu(self.input_layer(x))
        x = self.dropout1(x)

        x = F.relu(self.hidden_layer_1(x))
        x = self.dropout1(x)

        x = F.relu(self.hidden_layer_2(x))
        x = self.dropout2(x)

        x = F.relu(self.hidden_layer_3(x))
        x = self.dropout2(x)

        x = self.output_layer(x)
        return x

def load_model(filepath):

    model = MLPShallow()
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    model.eval()
    return model
