import torch
import torch.nn as nn
import torch.autograd as autograd

def get_act(name):
    name = name.lower()
    if name == "tanh": return nn.Tanh()
    if name == "relu": return nn.ReLU()
    if name == "sine":
        class Sine(nn.Module):
            def forward(self, x): return torch.sin(x)
        return Sine()
    raise ValueError(f"Unknown activation {name}")

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation, layers_per_block):
        super().__init__()
        if layers_per_block == 2:
            self.main = nn.Sequential(                 # <-- central path
                nn.Linear(in_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, out_dim),
            )
        if layers_per_block == 3:
            self.main = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, out_dim),
            )
        self.shortcut = nn.Linear(in_dim, out_dim)  # Projection shortcut

    def forward(self, x):
        return self.main(x) + self.shortcut(x)

class ResidualNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim, hidden_dim, output_dim = cfg["in_features"], cfg["hidden_dim"], cfg["out_features"]
        layers_per_block = cfg["layers_per_block"]
        num_blocks = cfg.get("num_blocks", 4)
        act = get_act(cfg.get("activation", "tanh"))
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualBlock(input_dim, hidden_dim, hidden_dim, act, layers_per_block))
        for _ in range(num_blocks - 2):
            self.blocks.append(ResidualBlock(hidden_dim, hidden_dim, hidden_dim, act, layers_per_block))
        self.blocks.append(ResidualBlock(hidden_dim, hidden_dim, output_dim, act, layers_per_block))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x