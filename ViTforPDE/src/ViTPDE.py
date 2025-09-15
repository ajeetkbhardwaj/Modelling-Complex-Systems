import torch
import torch.nn as nn
from einops import rearrange 
from einops.layers.torch import Rearrange

"""
FeedForwardNetwork:
1. Layer Normalization
2. Linear Layer : dim -> hidden_dim
3. GELU 
4. Dropout
5. Linear Layer : hidden_dim -> dim
6. Dropout

"""
class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.ffnet = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffnet(x)
    
"""
It will take a sequence of embedding of dimension equal to dim
"""
class AttensionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.2):
        super().__init__()
        inner_dim = dim_head * heads
        proj_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if proj_out else nn.Identity()
    
    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

"""
Transformer Block : 
1. Applied depth times :
   a) Attension block : dim -> dim (least)
   b) MLP block : dim-> dim(least)
2. Applied Layer Normalization
"""
class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(  
         nn.ModuleList([
        AttensionBlock(dim, heads=heads, dim_head=dim_head),
        FeedForwardNetwork(dim, mlp_dim)
    ])
)

            """self.layers.append(nn.ModuleList)([
                AttensionBlock(dim, heads=heads, dim_head = dim_head),
                FeedForwardNetwork(dim, mlp_dim)
            ])"""

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x 
        return self.norm(x)


def pairs(n):
    return n if isinstance(n, tuple) else (n, n)

"""
image_size : (n, c, h, w)
patch sizes : (p_h, p_w)
number of patches : (n_h, n_w)
Remark : Hold true h%p_h == 0

1. Applied to patch embedding :
   a) (n, c, p_h * p1, p_w * p2) -> (n, n_h*n_w, p_h*p_w*c)
   b) Normalization Layer
   c) Linear Embedding : p_h * p_w * c -> dim
   d) Layer Normization
2. Positional Embedding
3. Transformer Block
4. Depatchify
"""
class ViTPDE(nn.Module):
    def __init__(self,
                image_size,
                patch_size,
                dim,
                depth,
                heads,
                mlp_dim = 256,
                channels = 1,
                dim_head = 32,
                emb_dropout = 0.2):
        super().__init__()
        image_height, image_width = pairs(image_size)
        patch_height, patch_width = pairs(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.patch_to_image = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height // patch_height)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerBlock(dim, depth, heads, dim_head, mlp_dim)

        self.conv_last = torch.nn.Conv2d(in_channels = channels,
                                          out_channels= channels,
                                          kernel_size = 3,
                                          padding     = 1)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        _, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.patch_to_image(x)
        x = self.conv_last(x)
        return x
    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams})')

        return nparams