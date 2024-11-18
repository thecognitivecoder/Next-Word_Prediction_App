# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size=1024, activation_func="relu"):
        super(NextWord, self).__init__()
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        
        # Store the activation function based on the input parameter
        if activation_func == "relu":
            self.activation = F.relu
        elif activation_func == "tanh":
            self.activation = torch.tanh
        elif activation_func == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation function: {activation_func}")

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)  # Flatten the embedding output
        x = self.activation(self.lin1(x))  # Apply selected activation
        x = self.lin2(x)  # Output logits for vocabulary size
        return x
