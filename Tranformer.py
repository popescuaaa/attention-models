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


'''

 Encoder decoder architecture
 encoder: process input and returns a feature vector
 decoder: process the target sequence and incorporates info from encoder
 
 Each of the layers in our encoder and decoder contains a fully connected feed-forward network, which 
 consists of two linear transformations with a ReLU activation in between. 
 The dimensionality of input and output is 512, and the inner-layer has dimensionality 2048.
 
'''


def feed_forward(dim_ff: int = 2048, dim_input: int = 512) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_ff),
        nn.ReLU(),
        nn.Linear(dim_ff, dim_input),
    )


'''

The output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the 
function implemented by the sub-layer itself. … We apply dropout to the output of 
each sub-layer, before it is added to the sub-layer input and normalized.

'''


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super(Residual, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))


class EncodingLayer(nn.Module):
    def __init__(self, dim_ff: int, dim_model: int, num_heads: int, dropout: float):
        super(EncodingLayer, self).__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads=num_heads, dim_input=dim_model, dim_k=dim_k, dim_v=dim_v),
            dimension=dim_model,
            dropout=dropout
        )
        self.ff = Residual(
            feed_forward(dim_input=dim_model, dim_ff=dim_ff),
            dimension=dim_model,
            dropout=dropout
        )

    def forward(self, src: Tensor) -> Tensor:
        """
            :param src:
            :return: feature vector for input
        """
        src = self.attention(src, src, src)
        return self.ff(src)


class Encoder(nn.Module):
    def __init__(self, dim_ff: int, dim_model: int, num_heads: int, dropout: float, num_layers: int):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncodingLayer(dim_ff=dim_ff, dim_model=dim_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += positional_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, dim_ff: int, dim_model: int, num_heads: int, dropout: float):
        super(DecoderLayer, self).__init__()
        dim_k = dim_v = dim_model // num_heads

        self.attention_1 = Residual(
            MultiHeadAttention(num_heads=num_heads, dim_input=dim_model, dim_k=dim_k, dim_v=dim_v),
            dimension=dim_model,
            dropout=dropout,
        )

        self.attention_2 = Residual(
            MultiHeadAttention(num_heads=num_heads, dim_input=dim_model, dim_k=dim_k, dim_v=dim_v),
            dimension=dim_model,
            dropout=dropout,
        )

        self.ff = Residual(
            feed_forward(dim_input=dim_model, dim_ff=dim_ff),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, target: Tensor, memory: Tensor) -> Tensor:
        target = self.attention_1(target, target, target)
        target = self.attention_2(memory, memory, target)
        return self.ff(target)


class Decoder(nn.Module):
    def __init__(self, dim_ff: int, dim_model: int, num_heads: int, dropout: float, num_layers: int):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(dim_model=dim_model, num_heads=num_heads, dim_ff=dim_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, target: Tensor, memory: Tensor) -> Tensor:
        seq_len, dimension = target.size(1), target.size(2)
        target += positional_encoding(seq_len, dimension)
        for layer in self.layers:
            target = layer(target, memory)

        return torch.softmax(self.linear(target), dim=-1)


class Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_model: int = 512,
                 num_heads: int = 6,
                 dim_ff: int = 2048,
                 dropout: float = 0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
            dropout=dropout,
        )

        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
            dropout=dropout,
        )

    def forward(self, src: Tensor, target: Tensor) -> Tensor:
        return self.decoder(target, self.encoder(src))


def run_transformer_test():
    t = Transformer()
    src = torch.rand(64, 16, 512)
    tgt = torch.rand(64, 16, 512)
    out = t(src, tgt)
    print(out.shape)


if __name__ == '__main__':
    run_scaled_dot_product_test()
    run_attention_head_test()
    run_multi_head_attention_test()
    run_positional_encoding_test()
    run_transformer_test()
