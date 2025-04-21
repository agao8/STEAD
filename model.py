import torch
from torch import nn
from utils import FeedForward, DECOUPLED
from performer_pytorch import Performer

class AttnBlock(nn.Module):
    def __init__(self, dim, depth, dropout, attn_dropout, heads = 16, ff_mult = 2):
        super().__init__()
        self.performer = Performer(dim = dim, 
                                   depth = depth, 
                                   heads = heads, 
                                   dim_head = dim // heads, 
                                   causal = False,
                                   ff_mult = ff_mult,
                                   local_attn_heads = 8,
                                   local_window_size = dim // 8,
                                   ff_dropout = dropout,
                                   attn_dropout = attn_dropout,
                                   )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B, -1, C)
        x = self.performer(x)
        x = x.view(B, T, H, W, C)
        return x

class ConvBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        dropout = 0.,
        heads = 16,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.conv = DECOUPLED(dim, heads)
        self.ff = FeedForward(dim, ff_mult, dropout)

    def forward(self, x):
        x = x + self.conv(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# main class

class Model(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.2,
        attn_dropout = 0.1,
        ff_mult = 4,
        dims = (192, 128),
        depths = (3, 3),
        block_types = ('c', 'a')
    ):
        dims = dims
        depths = depths
        block_types = block_types
        super().__init__()
        self.init_dim, *_, last_dim = dims

        self.stages = nn.ModuleList([])

        for ind, (depth, block_types) in enumerate(zip(depths, block_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            
            if block_types == "c":
                for _ in range(depth):
                    self.stages.append(
                        ConvBlock(
                            dim = stage_dim,
                            ff_mult=ff_mult,
                            dropout = dropout,
                        )
                    )
            elif block_types == "a":
                for _ in range(depth):
                    self.stages.append(AttnBlock(stage_dim, 1, dropout, attn_dropout, ff_mult=ff_mult))
                
            if not is_last:
                self.stages.append(
                    nn.Sequential(
                        nn.LayerNorm(stage_dim),
                        nn.Linear(stage_dim, dims[ind + 1])
                    )
                )

        self.norm0 = nn.LayerNorm(192)
        self.linear = nn.Linear(192, dims[0])
        self.norm = nn.LayerNorm(last_dim)
        self.fc = nn.Linear(last_dim, 1)
        self.drop_out = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        
    def forward(self, x):

        x = x.permute(0, 2, 3, 4, 1)

        if x.shape[4] != self.init_dim:
            x = self.linear(self.norm0(x))

        for stage in self.stages:
            x = stage(x)

        
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pooling(x).squeeze()

        
        x = self.drop_out(x)
        x = self.norm(x)
        logits = self.fc(x)
        return logits, x
