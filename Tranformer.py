import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f

'''
    
    Q, K, and V are batches of matrices, 
    each with shape (batch_size, seq_length, num_features). 
    Multiplying the query (Q) and key (K) arrays results in a 
    (batch_size, seq_length, seq_length) array, which tells us roughly 
    how important each element in the sequence is. 
    
'''


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.shape[-1] ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


def run_scaled_dot_product_test() -> None:
    query = torch.randn(size=(10, 10, 10))
    key = torch.randn(size=(10, 10, 10))
    value = torch.randn(size=(10, 10, 10))

    assert scaled_dot_product_attention(query=query, key=key, value=value).shape == torch.Size([10, 10, 10]), \
        'Scaled product failed to produce the correct shape for output'


class AttentionHead(nn.Module):
    def __init__(self, dim_input: int, dim_k: int, dim_v: int):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(dim_input, dim_k)
        self.k = nn.Linear(dim_input, dim_k)
        self.v = nn.Linear(dim_input, dim_v)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(q), self.k(k), self.v(v))


def run_attention_head_test() -> None:
    ah = AttentionHead(10, 10, 10)
    query = torch.randn(size=(10, 10, 10))
    key = torch.randn(size=(10, 10, 10))
    value = torch.randn(size=(10, 10, 10))
    assert ah(query, key, value).shape == torch.Size([10, 10, 10]), \
        'Attention head failed to produce correct shape of the output '


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_input: int, dim_k: int, dim_v: int):
        super(MultiHeadAttention, self).__init__()
        self.attention_heads = nn.ModuleList(
            [AttentionHead(dim_input=dim_input, dim_k=dim_k, dim_v=dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_input)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return self.linear(torch.cat([ah(q, k, v) for ah in self.attention_heads], dim=-1))


def run_multi_head_attention_test() -> None:
    mha = MultiHeadAttention(13, 10, 10, 10)
    query = torch.randn(size=(10, 10, 10))
    key = torch.randn(size=(10, 10, 10))
    value = torch.randn(size=(10, 10, 10))
    assert mha(query, key, value).shape == torch.Size([10, 10, 10]), \
        'Multi head attention failed to produce correct shape of the output '


def positional_encoding(seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def run_positional_encoding_test():
    assert positional_encoding(150, 10).shape == torch.Size((1, 150, 10)), \
        'The positional encoding failed to produce data with corret dimension on output'


if __name__ == '__main__':
    run_scaled_dot_product_test()
    run_attention_head_test()
    run_multi_head_attention_test()
    run_positional_encoding_test()
