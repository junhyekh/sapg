#!/usr/bin/env python3

from typing import (Tuple, Optional, Union, Callable)

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
# Try not to mistake `Mlp` with our custom MLP
from flash_attn.modules.mlp import Mlp as _Mlp

S = Union[int, Tuple[int, ...]]
T = th.Tensor

LN_NAMES = ['ln', 'layer_norm', 'layernorm']
BN_NAMES = ['bn', 'batch_norm', 'batchnorm']
GN_NAMES = ['gn', 'group_norm', 'groupnorm']


def get_activation_function(actv: str) -> nn.Module:
    if not isinstance(actv, str):
        return actv
    actv = actv.lower()
    if actv == 'tanh':
        out = nn.Tanh
    elif actv == 'relu':
        out = nn.ReLU
    elif actv == 'lrelu':
        out = nn.LeakyReLU
    elif actv == 'elu':
        out = nn.ELU
    elif actv == 'relu6':
        out = nn.ReLU6
    elif actv == 'gelu':
        out = nn.GELU
    elif actv == 'selu':
        out = nn.SELU
    elif actv == 'silu':
        out = nn.SiLU
    elif actv == 'none':
        out = nn.Identity
    else:
        raise KeyError(F'Unknown actv={actv}')
    return out


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: th.Tensor) -> th.Tensor:
        batch_shape = input.shape[:-1]
        out = super().forward(input.reshape(-1, input.shape[-1]))
        return out.view(*batch_shape, out.shape[-1])


class LayerNorm1d(nn.LayerNorm):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 *args, **kwds):
        super().__init__(num_features, eps, affine, *args, **kwds)

    def forward(self, input: th.Tensor) -> th.Tensor:
        batch_shape = input.shape[:-1]
        out = super().forward(input.reshape(-1, input.shape[-1]))
        return out.view(*batch_shape, out.shape[-1])


class GroupNorm1d(nn.GroupNorm):
    def __init__(self,
                 num_features: int,
                 num_groups: int = 2,
                 *args, **kwds):
        # assert num_features % num_groups == 0
        super().__init__(num_groups, num_features, *args, **kwds)
        # for k, v in self.named_parameters():
        #     print('par', k, v.shape)

        # for k, v in self.named_buffers():
        #     print('buf', k, v.shape)

    def forward(self, input: th.Tensor) -> th.Tensor:
        batch_shape = input.shape[:-1]
        out = super().forward(input.reshape(-1, input.shape[-1]))
        return out.view(*batch_shape, out.shape[-1])


def get_normalization_function(norm_cls: str) -> nn.Module:
    if isinstance(norm_cls, Callable):
        return norm_cls

    if norm_cls is None:
        return nn.Identity

    norm_cls = norm_cls.lower()
    if norm_cls in LN_NAMES:
        # NOTE(ycho): LayerNorm1d wraps nn.LayerNorm
        # to provide a consistent interface w.r.t. BatchNorm1d
        out = LayerNorm1d
    elif norm_cls in BN_NAMES:
        out = BatchNorm1d
    elif norm_cls in GN_NAMES:
        out = GroupNorm1d
    elif norm_cls in ['none']:
        out = nn.Identity
    else:
        raise KeyError(F'Unknown norm_cls={norm_cls}')
    return out


class LinearNorm(nn.Module):
    """ Linear layer with optimal batch normalization. """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 norm: Optional[str] = None,
                 **kwds):
        super().__init__()

        affine = kwds.pop('affine', False)
        bias = kwds.pop('bias', True)
        self.norm = get_normalization_function(norm)(
            num_features=dim_out,
            affine=affine)
        if isinstance(self.norm, BatchNorm1d):
            bias = False
        self.linear = nn.Linear(
            dim_in, dim_out, bias=bias,
            **kwds)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.linear(x)
        s = x.shape
        x = x.reshape(-1, s[-1])
        x = self.norm(x)
        x = x.reshape(s)
        return x


class MLP(nn.Sequential):
    """ Generic multilayer perceptron. """

    def __init__(self,
                 dims: Tuple[int, ...],
                 bias: bool = True,
                 actv: nn.Module = nn.LeakyReLU,
                 norm: Optional[str] = 'layernorm',
                 affine: bool = False,
                 activate_output: bool = False,
                 ):
        super().__init__()
        assert (len(dims) >= 2)

        if isinstance(actv, str):
            actv = get_activation_function(actv)

        layers = []
        # hidden layers
        for d0, d1 in zip(dims[:-2], dims[1:-1]):
            layers.extend(
                (LinearNorm(
                    d0,
                    d1,
                    bias=bias,
                    norm=norm,
                    affine=affine,
                ),
                    actv(),
                ))
        # last layer
        if activate_output:
            layers.extend((
                LinearNorm(
                    dims[-2],
                    dims[-1],
                    bias=bias,
                    norm=norm,
                    affine=affine,
                ),
                actv()))
        else:
            # FIXME(ycho): not much I can do here except
            # hardcoding... for now
            layers.extend((
                nn.Linear(dims[-2], dims[-1], bias=bias),)
            )
        super().__init__(*layers)


class TransformerEncoderFA(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_layers: int,
        num_heads: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        p_drop: float = 0.0,
        norm_first: bool = True,  # more efficient
        activation=F.gelu,
        rotary_emb_dim: int = 0,
        use_flash_attn: bool = True
    ):
        super().__init__()
        self._use_flash_attn = use_flash_attn

        # Configure reasonable defaults
        if num_heads is None:
            num_heads = dim_model // 64
        if dim_feedforward is None:
            dim_feedforward = dim_model * 4

        # apply params to subclass args
        mixer_cls = partial(
            MHA,
            num_heads=num_heads,
            use_flash_attn=use_flash_attn,
            rotary_emb_dim=rotary_emb_dim
        )

        mlp_cls = partial(_Mlp,
                          hidden_features=dim_feedforward,
                          activation=activation)

        # create layers
        self.layers = nn.ModuleList([
            Block(
                dim_model,
                mixer_cls=mixer_cls,
                mlp_cls=mlp_cls,
                resid_dropout1=p_drop,
                resid_dropout2=p_drop,
                prenorm=norm_first,
            ) for _ in range(num_layers)
        ])

    def forward(self, x: th.Tensor):
        dtype = x.dtype
        B = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])
        with th.cuda.amp.autocast(
            enabled=self._use_flash_attn,
            dtype=th.bfloat16  # or just float16?
        ):
            for layer in self.layers:
                x, _ = layer(x)
        out = x.to(dtype=dtype)
        out = out.reshape(*B, *out.shape[-2:])
        return out


class SplitDim(nn.Module):
    def __init__(self,
                 sizes: Tuple[int, ...],
                 dim: int = -1):
        super().__init__()
        self.dim = dim
        self.sizes = sizes
        self.splits = np.cumsum(sizes)[:-1].tolist()

    def extra_repr(self):
        src = sum(self.sizes)
        return F'{src}->{self.sizes}'

    def forward(self, x: th.Tensor):
        return th.tensor_split(x, self.splits, dim=self.dim)


def test_norm():
    for norm_cls in ['bn', 'ln', 'gn', 'none']:
        if norm_cls == 'gn':
            norm = get_normalization_function(norm_cls)(num_features=8,
                                                        num_groups=4,
                                                        affine=False)
        else:
            norm = get_normalization_function(norm_cls)(num_features=8,
                                                        affine=False)
        x = th.randn((2, 1, 1, 1, 8), dtype=th.float32)
        y = norm(x)
        print(F'{x.shape} -> {y.shape}')


def test_mlp():
    print(MLP([8, 128, 64], actv='tanh'))


def test_xformer():
    B: int = (4, 3)
    T: int = 8
    D: int = 256
    device: str = 'cuda:1'
    xformer = TransformerEncoderFA(D, 3, 8, 512).to(
        device=device)
    print(xformer)

    x = th.zeros((*B, T, D),
                 dtype=th.float32,
                 device=device)
    y = xformer(x)
    print(y.shape)  # -> 4,8,256


def main():
    # test_norm()
    # test_mlp()
    test_xformer()


if __name__ == '__main__':
    main()