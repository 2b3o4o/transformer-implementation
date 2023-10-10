import logging
import torch
from torch import tensor, sin, cos
from math import sqrt
from torch.nn.functional import softmax
import torch.nn as nn


def par_attention(queries: tensor, keys: tensor, values: tensor, dim: int) -> tensor:
    raw_weights = torch.bmm(queries, keys.transpose(1, 2))

    mask = torch.tril(torch.ones_like(raw_weights), diagonal=0)
    raw_weights = raw_weights.masked_fill(mask == 0, float('-inf'))

    scale_factor = sqrt(dim)
    scaled_weights = softmax(raw_weights / scale_factor, dim=2)

    # now scaled weights is a matrix where each row represents the scaled weights produced based on a given query.
    # meanwhile values just has a value vector on each row.

    reshaped_scaled_weights = scaled_weights.view(scaled_weights.shape[0], scaled_weights.shape[1], scaled_weights.shape[2], 1)
    reshaped_values = values.view(values.shape[0], values.shape[1], 1, values.shape[2])

    scaled_values = reshaped_scaled_weights * reshaped_values

    contextualized_values = torch.sum(scaled_values, 2)
    return contextualized_values

class PositionalEncoding(nn.Module):
    def __init__(self, dims, context_len, device):
        super().__init__()
        self.device = device
        self.dims = dims
        self.context_len = context_len
        self.proj = nn.Linear(1, self.dims)

        positional_matrix = torch.zeros([self.context_len, self.dims])
        for pos in range(0, self.context_len):
            for i in range(0, self.dims // 2):
                positional_matrix[pos][2 * i] = sin(torch.tensor(pos / (10000 ** (2 * i / self.dims))))
                positional_matrix[pos][2 * i + 1] = cos(torch.tensor(pos / (10000 ** (2 * i / self.dims))))
        self.register_buffer('positional_matrix', positional_matrix)
        self.positional_matrix = self.positional_matrix.to(self.device)


    def forward(self, x: tensor) -> tensor:
        # x is token ids. we'll say it's context_len integers packed into a tensor, where each one represents a token. it can also be batched.
        output = torch.zeros([x.shape[0], self.context_len, self.dims]).to(self.device)
        for batch in range(0, x.shape[0]):
            output[batch] = self.proj(x[batch].view(x.shape[1], -1))
            output[batch] += self.positional_matrix
        return output
    
class AttentionHead(nn.Module):
    # assumes query, key, and value vectors have the same dimensionality
    def __init__(self, model_dim, vectors_dim):
        super().__init__()
        self.model_dim = model_dim
        self.vectors_dim = vectors_dim
        self.Q_proj = nn.Linear(model_dim, vectors_dim, bias=False)
        self.K_proj = nn.Linear(model_dim, vectors_dim, bias=False)
        self.V_proj = nn.Linear(model_dim, vectors_dim, bias=False)

    def forward(self, x):
        # each row of x is a vector representing the meaning of the token at the corresponding position with whatever context we've attained so far.
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)
        output = par_attention(Q, K, V, self.vectors_dim)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.att_heads = nn.ModuleList([AttentionHead(model_dim, model_dim // num_heads) for _ in range(num_heads)])
        self.proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x):
        head_outputs = [head(x) for head in self.att_heads]
        x = torch.concat(head_outputs, dim=2)
        x = self.proj(x)
        return x
        
class TransformerLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_hidden_dim, context_len):
        super().__init__()
        self.attention_block = MultiHeadAttention(model_dim, num_heads)
        self.norm1 = nn.LayerNorm(normalized_shape=[context_len, model_dim])
        self.ff1 = nn.Linear(model_dim, ff_hidden_dim)
        self.ff_relu = nn.ReLU()
        self.ff2 = nn.Linear(ff_hidden_dim, model_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=[context_len, model_dim])

    def forward(self, x):
        x_res = x
        x = self.attention_block(x)
        x += x_res
        x = self.norm1(x)

        x_res = x
        x = self.ff1(x)
        x = self.ff_relu(x)
        x = self.ff2(x)
        x += x_res
        x = self.norm2(x)

        return x

class TransformerNetwork(nn.Module):
    def __init__(self, output_dict_size: int, device: torch.device=None, context_len: int=16, num_layers=3, model_dim=256, att_heads=4, ff_hidden_dim=1024):
        logging.debug("Initializing model...")
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encode_embed = PositionalEncoding(model_dim, context_len, self.device)
        self.trans_layers = nn.ModuleList([TransformerLayer(model_dim, att_heads, ff_hidden_dim, context_len) for _ in range(num_layers)])
        self.word_predictor = nn.Linear(model_dim * context_len, output_dict_size)

        self.context_len = context_len

    def forward(self, x):
        x = self.encode_embed(x)
        for layer in self.trans_layers:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        x = self.word_predictor(x)
        return x