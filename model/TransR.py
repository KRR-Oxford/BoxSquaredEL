import torch.nn as nn

class TransR(nn.Module):
    def __init__(self, embedding_dim):
        super(TransR, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )
    def forward(self, input):
        return self.mlp(input)