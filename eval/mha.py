import dace
import math
import numpy as np

from dace.libraries import blas

import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q = nn.Linear(input_dim, embed_dim, bias=False)
        self.k = nn.Linear(input_dim, embed_dim, bias=False)
        self.v = nn.Linear(input_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()

        # Separate Q, K, V from linear output
        unchunk = lambda x: x.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = map(unchunk, (self.q(x), self.k(x), self.v(x)))

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

B = 8
S = 512
E = 768
H = 16
E_H = E // H
SQRT_H = math.sqrt(E_H)

def projection(x: dace.float32[B, S, E],
        W_Q: dace.float32[E, E_H * H],
        W_K: dace.float32[E, E_H * H],
        W_V: dace.float32[E, E_H * H],
        ):
    Q = np.einsum('bik,kj->bij', x, W_Q)
    K = np.einsum('bik,kj->bij', x, W_K)
    V = np.einsum('bik,kj->bij', x, W_V)
    # split out the head dimension & permute
    Q_unpack = np.einsum('bshe->bhse', Q.reshape([B, S, H, E_H]))
    K_unpack = np.einsum('bshe->bhse', K.reshape([B, S, H, E_H]))
    V_unpack = np.einsum('bshe->bhse', V.reshape([B, S, H, E_H]))
    return Q_unpack, K_unpack, V_unpack
d_projection = dace.program(projection)

def scaled_softmax(x: dace.float32[B, H, S, S]):
    scaled_x = x / SQRT_H
    rowmax = np.maximum.reduce(scaled_x, axis=-1, keepdims=True)
    exponent = np.exp(scaled_x - rowmax)
    rowsum = np.add.reduce(exponent, axis=-1, keepdims=True)
    return exponent / rowsum
d_scaled_softmax = dace.program(scaled_softmax)

def fuse_sg(sdfg):
    from dace.transformation.subgraph import MultiExpansion, SubgraphFusion
    from dace.transformation.dataflow import ReduceExpansion, MapFusion
    sdfg.apply_transformations_repeated(ReduceExpansion)
    sdfg.simplify()
    MultiExpansion.apply_to(sdfg, sdfg.node(0).nodes())
    SubgraphFusion.apply_to(sdfg, sdfg.node(0).nodes())
    sdfg.apply_transformations_repeated(MapFusion)

fused_softmax = d_scaled_softmax.to_sdfg()
fuse_sg(fused_softmax)


def self_attn(Q, K, V):
    scores = np.einsum('bhik,bhjk->bhij', Q, K)
    norm_scores = scaled_softmax(scores)
    # norm_scores: [B, H, S, S]
    return np.einsum("bhik,bhkj->bhij", norm_scores, V)

@dace.program
def d_self_attn(Q, K, V):
    scores = np.einsum('bhik,bhjk->bhij', Q, K)
    norm_scores = fused_softmax(scores)
    # norm_scores: [B, H, S, S]
    return np.einsum("bhik,bhkj->bhij", norm_scores, V)


def mhsa(x: dace.float32[B, S, E],
        W_Q: dace.float32[E, E],
        W_K: dace.float32[E, E],
        W_V: dace.float32[E, E],
        W_O: dace.float32[E, E]):
    Q, K, V = projection(x, W_Q, W_K, W_V)
    values = self_attn(Q, K, V)
    # unpermute
    values_permute = np.einsum('bhse->bshe', values)
    values_reshaped = values_permute.reshape([B, S, E])
    return np.einsum('bik,kj->bij', values_reshaped, W_O)
@dace.program
def d_mhsa(x: dace.float32[B, S, E],
        W_Q: dace.float32[E, E],
        W_K: dace.float32[E, E],
        W_V: dace.float32[E, E],
        W_O: dace.float32[E, E]):
    Q, K, V = d_projection(x, W_Q, W_K, W_V)
    values = d_self_attn(Q, K, V)
    # unpermute
    values_permute = np.einsum('bhse->bshe', values)
    values_reshaped = values_permute.reshape([B, S, E])
    return np.einsum('bik,kj->bij', values_reshaped, W_O)


def test_full():
    with torch.no_grad():
        mod = MultiheadAttention(E, E, H)
        x = torch.randn([B, S, E])
        expected = mod(x)
        parameters = dict(W_Q=mod.q.weight,
                W_K=mod.k.weight,
                W_V=mod.v.weight,
                W_O=mod.o_proj.weight)
        parameters = {k: v.T.numpy().copy() for k, v in parameters.items()}

    result = mhsa(x.numpy(), **parameters)
    np.testing.assert_allclose(result, expected.numpy(),atol=1e-5, rtol=1e-5, err_msg='numpy_failed')
    print("numpy passed")
    
    blas.default_implementation = 'OpenBLAS'
    sdfg = d_mhsa.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(x=x.numpy().copy(), **parameters)
    np.testing.assert_allclose(result, expected.numpy(),atol=1e-5, rtol=1e-5, err_msg='dace_failed')
    print("All passed :)")

def test_sa():
    Q = np.random.rand(8, 16, 512, 48).astype(np.float32)
    K = np.random.rand(8, 16, 512, 48).astype(np.float32)
    V = np.random.rand(8, 16, 512, 48).astype(np.float32)

    expected = self_attn(Q, K, V)
    result = d_self_attn(Q.copy(), K.copy(), V.copy())
    np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5, err_msg='dace_failed')

test_full()
# test_sa()


