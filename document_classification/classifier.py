import torch.nn as nn
import torch.nn.functional as F

class RNNClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_classes=42):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, y=None):
        out, hn = self.rnn(x)
        logits = self.fc(out)

        if y is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, y)

        return loss, logits