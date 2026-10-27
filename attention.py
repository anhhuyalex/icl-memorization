import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, einsum
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1) # for head: Q = W_q @ x, K = W_k @ x, V = W_v @ x
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # split into heads

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
        
        attn = self.attend(dots) # attn = softmax(Q @ K^T / sqrt(d))
        out = torch.matmul(attn, v) # out = attn @ V
        out = rearrange(out, 'b h n d -> b n (h d)') # concat heads
        return self.to_out(out)
    
class TransformerEncoder(nn.Module):
    def __init__(self, model_size, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(model_size)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(model_size, heads = heads, dim_head = dim_head),
                FeedForward(model_size, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

class Transformer(nn.Module):
    def __init__(self, x_dim, model_size, len_context, num_classes, depth, heads, mlp_dim, dim_head = 64, positional_encoding=False,
                 transformer_encoder = "pytorch"):
        """
        Adapting implementation from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
        """
        super().__init__()   
        self.to_x_embedding = nn.Sequential(
            nn.LayerNorm(x_dim),
            nn.Linear(x_dim, model_size),
            nn.LayerNorm(model_size),
        )
        # self.to_y_embedding = nn.Embedding(num_classes, model_size)
        # self.len_context = 2 * len_context + 1 # x,y from support set and x from query set

        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
 
        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.len_context, model_size)) # learnable positional embedding
        
        self.transformer_encoder = TransformerEncoder(model_size, depth, heads, dim_head, mlp_dim)
        self.linear_head = nn.Linear(model_size, num_classes)

    def forward(self, x):
        """
        :param x: (B, N, D) tensor of support set -- sequences of randomly projected images,
            where N is the sequence length, and D is the dimensionality of each token
        """
        seq = self.to_x_embedding(x)
        if self.positional_encoding: seq += self.pos_embedding

        output = self.transformer_encoder(seq)
        output = output.mean(dim = 1)
        output = self.linear_head(output)
        return output
       
class PytorchTransformer(nn.Module):
    def __init__(self, x_dim, model_size, len_context, num_classes, depth, nheads, mlp_dim, dim_head = 64, positional_encoding=False,
                 dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=True, norm_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()   
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}") 
        print ("x_dim", x_dim, "nheads", nheads)
        self.to_embedding = nn.Sequential(
            nn.LayerNorm(x_dim),
            nn.Linear(x_dim, model_size),
            nn.LayerNorm(model_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, nhead=nheads, dim_feedforward=mlp_dim, dropout=dropout,
                                                    activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, 
                                                    norm_first=norm_first, **factory_kwargs)
        encoder_norm = nn.LayerNorm(model_size, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, norm=encoder_norm)

        self.linear_head = nn.Linear(model_size, num_classes)

        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, len_context, model_size)) # learnable positional embedding
        self._reset_parameters()

        self.x_dim = x_dim
        self.model_size = model_size 
        self.len_context = len_context 
        self.num_classes = num_classes 
        self.depth = depth 
        self.nheads = nheads
        self.mlp_dim = mlp_dim 
        self.dim_head = dim_head 
        self.batch_first = batch_first

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_embedding(x)
        if self.positional_encoding: x += self.pos_embedding
        memory = self.encoder(x)
        output = memory.mean(dim=1)
        output = self.linear_head(output)
        return output

class MLP(nn.Module):
    def __init__(self, x_dim, model_size, num_classes, mlp_dim,  device=None, dtype=None) -> None:
        super().__init__()   
        # self.to_embedding = nn.Sequential(
        #     # nn.LayerNorm(x_dim),
        #     nn.Linear(x_dim, model_size),
        #     # nn.LayerNorm(model_size),
        # )
 

        # self.mlp_dim = mlp_dim 
        # if self.mlp_dim < 0:
        #     self.mlp = nn.Identity()
        # else:
        self.mlp = nn.Sequential(
            # nn.LayerNorm(x_dim),
            nn.Linear(x_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.to_embedding(x) # shape: (B, D)
        
        # if self.positional_encoding: x += self.pos_embedding
        output = self.mlp(x)
        # print ("output", output.shape, output)
        return output

class CausalTransformer(nn.Module):
    def __init__(self, x_dim, model_size, num_classes, mlp_dim,  device=None, dtype=None) -> None:

        super().__init__()   
        self.to_embedding = nn.Sequential(
            # nn.LayerNorm(x_dim),
            nn.Linear(x_dim, model_size),
            nn.LayerNorm(model_size),
        )
        self.qkv = nn.Linear(model_size, model_size * 3)

        self.mlp_dim = mlp_dim 
        if self.mlp_dim < 0:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.Sequential(
                # nn.LayerNorm(model_size),
                nn.Linear(model_size, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, num_classes),
            )
 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_embedding(x) # shape: (B, N, D)
        
        # if self.positional_encoding: x += self.pos_embedding
        q, k, v = self.qkv(x).chunk(3, dim=-1) # for head: Q = W_q @ x, K = W_k @ x, V = W_v @ x
        sa = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        x = x + sa # shape: (B, N, D)
        output = self.mlp(x[:, -1, :])
        # print ("output", output.shape, output)
        return output, q, k
 
    
class CausalTransformerOneMinusOneEmbed(nn.Module):
    def __init__(self, x_dim,  mlp_dim,  
                 is_layer_norm = "True",
                 device=None, dtype=None) -> None:

        super().__init__()    
        if is_layer_norm == "True":
            self.to_embedding = nn.Sequential(
                nn.LayerNorm(x_dim), 
                nn.Linear(x_dim, mlp_dim),
                nn.LayerNorm(mlp_dim),

            )
        else:
            self.to_embedding = nn.Identity()
        # self.qkv = nn.Linear(model_size, model_size * 3)
        self.qkv = nn.Linear(mlp_dim, mlp_dim * 3, bias = False)
         
        self.mlp_dim = mlp_dim 
        if self.mlp_dim < 0:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.Sequential(
                nn.LayerNorm(mlp_dim),
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, 1),
            )
 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_embedding(x) # shape: (B, N, D)
        q, k, v = self.qkv(x).chunk(3, dim=-1) # for head: Q = W_q @ x, K = W_k @ x, V = W_v @ x
        scale_factor = 1.0 / (q.size(1) ** 0.5)
        # sa = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0) 
        qlast_dot_kT = torch.matmul(q[:, [-1], :], k.transpose(-2, -1)) * scale_factor # shape: (B, 1, N)
        attn = F.softmax(qlast_dot_kT, dim=-1) # shape: (B, 1, N)
        sa = torch.matmul(attn, v) # shape: (B, 1, D)
        output = self.mlp(x[:, -1, :] + sa[:, -1, :]) # shape: (B, 1)
        # print ("output", output.shape, output)
        return output 
 
class MultiLayerTransformer(nn.Module):
    def __init__(self, x_dim,  mlp_dim,  
                 is_layer_norm = "True",
                 num_layers = 12,
                 device=None, dtype=None) -> None:

        super().__init__()    
        if is_layer_norm == "True":
            self.to_x_embedding = nn.Sequential(
                nn.LayerNorm(x_dim), 
                nn.Linear(x_dim, mlp_dim),
            )
            self.to_y_embedding = nn.Sequential(
                nn.Linear(1, mlp_dim),
            )
        else:
            self.to_embedding = nn.Identity()
        self.qkv_dim = mlp_dim * 2
        self.num_layers = num_layers
        self.mlp_dim_layernorm = nn.LayerNorm(self.qkv_dim)
        self.qkvs = nn.ModuleList([nn.Linear(self.qkv_dim, self.qkv_dim * 3, bias = False) for _ in range(num_layers)])
        self.mlps = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(self.qkv_dim),
                nn.Linear(self.qkv_dim, self.qkv_dim),
                nn.ReLU(),
                nn.Linear(self.qkv_dim, self.qkv_dim),
            ) for _ in range(num_layers)])
    
        self.output = nn.Linear(self.qkv_dim, 1)
 

    def forward(self, x: torch.Tensor,
                target: torch.Tensor = None) -> torch.Tensor:
        x = self.to_x_embedding(x) # shape: (B, N, D)
        new_target = target.clone()
        new_target[:, -1, :] = 0
        new_target = self.to_y_embedding(new_target) # shape: (B, 1, D) 
        # zero out the last element of the target
        
        x = torch.cat((x, new_target), dim=-1) # shape: (B, N, 2*D)
        for lay in range(self.num_layers):
            q, k, v = self.qkvs[lay](self.mlp_dim_layernorm(x)).chunk(3, dim=-1) # for head: Q = W_q @ x, K = W_k @ x, V = W_v @ x
            sa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + sa 
            mlp_output = self.mlps[lay](self.mlp_dim_layernorm(x))
            x = x + mlp_output
        # q, k, v = self.qkv(x).chunk(3, dim=-1) # for head: Q = W_q @ x, K = W_k @ x, V = W_v @ x
        # scale_factor = 1.0 / (q.size(1) ** 0.5)
        # # sa = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0) 
        # qlast_dot_kT = torch.matmul(q[:, [-1], :], k.transpose(-2, -1)) * scale_factor # shape: (B, 1, N)
        # attn = F.softmax(qlast_dot_kT, dim=-1) # shape: (B, 1, N)
        # sa = torch.matmul(attn, v) # shape: (B, 1, D)
        # output = self.mlp(x[:, -1, :] + sa[:, -1, :]) # shape: (B, 1)
        # print ("output", output.shape, output)
        output = self.output(self.mlp_dim_layernorm(x[:, -1, :]))

        return output 
 