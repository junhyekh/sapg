import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from isaacgymenvs.tasks.allegro_kuka.util import merge_shapes
import einops
from icecream import ic

from typing import Optional, Dict, Tuple, Iterable, Union
from collections.abc import Mapping

from flash_attn.modules.mha import FlashCrossAttention, MHA

class LinearBn(nn.Module):
    """ Linear layer with optimal batch normalization. """

    def __init__(self, dim_in: int, dim_out: int,
                 use_bn: bool = True,
                 use_ln: bool = False, **kwds):
        super().__init__()

        if use_bn and use_ln:
            raise ValueError('use_bn and use_ln cannot both be true!')

        affine = kwds.pop('affine', True)
        if use_bn or use_ln:
            if use_bn:
                kwds['bias'] = False
            self.linear = nn.Linear(dim_in, dim_out, **kwds)
            if use_ln:
                self.bn = nn.LayerNorm(dim_out, elementwise_affine=affine)
            else:
                self.bn = nn.BatchNorm1d(dim_out, affine=affine)
        else:
            self.linear = nn.Linear(dim_in, dim_out, **kwds)
            self.bn = nn.Identity()

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.linear(x)
        s = x.shape
        x = x.reshape(-1, s[-1])
        x = self.bn(x)
        x = x.reshape(s)
        return x

def get_activation_function(act_cls: str) -> nn.Module:
    if not isinstance(act_cls, str):
        return act_cls
    act_cls = act_cls.lower()
    if act_cls == 'tanh':
        out = nn.Tanh
    elif act_cls == 'relu':
        out = nn.ReLU
    elif act_cls == 'lrelu':
        out = nn.LeakyReLU
    elif act_cls == 'elu':
        out = nn.ELU
    elif act_cls == 'relu6':
        out = nn.ReLU6
    elif act_cls == 'gelu':
        out = nn.GELU
    elif act_cls == 'selu':
        out = nn.SELU
    elif act_cls == 'none':
        out = nn.Identity
    else:
        raise KeyError(F'Unknown act_cls={act_cls}')
    return out

class MLP(nn.Module):
    """ Generic multilayer perceptron. """

    def __init__(self, dims: Tuple[int, ...],
                 act_cls: nn.Module = nn.LeakyReLU,
                 activate_output: bool = False,
                 use_bn: bool = True,
                 bias: bool = True,
                 use_ln: bool = False,
                 pre_ln_bias: bool = True,
                 affine: bool = True):
        super().__init__()
        assert (len(dims) >= 2)

        if isinstance(act_cls, str):
            act_cls = get_activation_function(act_cls)

        layers = []
        for d0, d1 in zip(dims[:-2], dims[1:-1]):
            # FIXME(ycho): incorrect `bias` logic
            if not use_ln:
                layer_bias = bias
            else:
                layer_bias = pre_ln_bias
            layers.extend(
                (LinearBn(
                    d0,
                    d1,
                    use_bn=use_bn,
                    bias=layer_bias,
                    use_ln=use_ln,
                    affine=affine),
                    act_cls(),
                 ))
        if activate_output:
            layers.extend((
                LinearBn(
                    dims[-2],
                    dims[-1],
                    use_bn=use_bn, bias=bias, use_ln=use_ln,
                    affine=affine),
                act_cls()))
        else:
            # FIXME(ycho): not much I can do here except
            # hardcoding... for now
            layers.extend((
                nn.Linear(dims[-2], dims[-1], bias=bias),)
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor):
        return self.model(x)

class PosEncodingSine(nn.Module):
    """
    \\hat{x} = [x;sin(s*Wx)]
    """

    def __init__(self, dim_in: int, dim_out: int,
                 scale: float = 30.0):
        super().__init__()
        self.linear = nn.Sequential(*[
            nn.Linear(dim_in, dim_out),
            # nn.BatchNorm1d(dim_out)
        ])
        self.out_dim = dim_out + dim_in
        self.scale = scale

        with th.no_grad():
            m = self.linear[0]
            num_input = m.weight.size(-1)
            assert (num_input == dim_in)
            m.weight.uniform_(-1 / num_input,
                              1 / num_input)

    def forward(self, x: th.Tensor):
        s = x.shape
        x_f = x.reshape(-1, x.shape[-1])
        out = th.sin(self.scale * self.linear(x_f))
        out = out.reshape(*s[:-1], out.shape[-1])
        out = th.cat([x, out], dim=-1)
        return out


class PosEncodingLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.out_dim = dim_in
        self.linear = nn.Linear(dim_in, dim_out)
        self.out_dim = dim_out + dim_in

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.linear(x)
        out = th.cat([x, y], dim=-1)
        return out


class PosEncodingMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int,
                 dim_hidden: Tuple[int, ...],
                 act_cls: str = 'gelu'
                 ):
        super().__init__()
        self.out_dim = dim_in
        # self.linear = nn.Linear(dim_in, dim_out)
        self.mlp = MLP(merge_shapes(dim_in, dim_hidden, dim_out),
                       act_cls=get_activation_function(act_cls),
                       activate_output=False,
                       use_bn=False,
                       bias=True,
                       use_ln=False)
        self.out_dim = dim_out + dim_in

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.mlp(x)
        out = th.cat([x, y], dim=-1)
        return out
    
class MHAWrapper(MHA):
    def forward(self, q, m):
        s = q.shape
        q = einops.rearrange(q, '... s d -> (...) s d')
        m = einops.rearrange(m, '... s d -> (...) s d')
        o = super().forward(q, m)
        o = o.reshape(*s[:-2], *o.shape[-2:])
        return o
    
class FlashMHA(MHA):
    def __init__(self, embed_dim,
        num_heads, *args, **kwargs):
        if 'bias' in kwargs:
            bias: bool = kwargs.pop('bias')
            kwargs['qkv_proj_bias'] = bias
        
        if 'attention_dropout' in kwargs:
            dropout: float = kwargs.pop('attention_dropout')
            kwargs['dropout'] = dropout

        super().__init__(embed_dim,
                        num_heads,
                        use_flash_attn=True,
                        **kwargs)
        
    def forward(self, x,
                x_kv=None,
                *args, **kwargs):
        
        output_attn = kwargs.pop('need_weights', False)
        if output_attn:
            ic("[Warning] Logging attention weights is not supported")

        x = super().forward(x, x_kv, *args, **kwargs)

        return x, None
    
class SinusoidalPositionalEncoding(nn.Module):
    """
    NeRF-style positional encoding.
    [Mildenhall et al. 2020].

    Computes the positional encoding for the
    normalized coordinate input in range (-1.0, +1.0).
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 flatten: bool = True,
                 pad: bool = False):
        """
        Args:
            dim_in: dimensionality of coordinate input.
            num_frequencies: Number of higher-frequency elements.
            num_samples: Fallback computation of number of frequencies.
            flatten: If true, flatten positional encoding to one channel.
        """
        super().__init__()

        self.dim_in = dim_in
        self.pad = pad
        if not pad:
            assert ((dim_out % (dim_in * 2)) == 0)
        self.num_frequencies = dim_out // (dim_in * 2)
        self.dim_out = dim_out

        # Precompute the coefficient multipliers.
        self.register_buffer(
            'coefs', th.as_tensor(
                np.pi * (2 ** th.arange(self.num_frequencies)),
                dtype=th.float))
        self.flatten = flatten

    def extra_repr(self):
        return F'{self.dim_in} -> {self.dim_out}'

    def forward(self, coords: th.Tensor) -> th.Tensor:
        """
        Args:
            coords: (..., D)
        Returns:
            pos_enc: (..., (2*F+1)*D) if flatten else (..., (2*F+1), D)
        """
        octaves = coords[..., None, :] * self.coefs[:, None]
        s = th.sin(octaves)
        c = th.cos(octaves)
        out = th.concat([s, c], dim=-2)
        # Optionally flatten the output.
        if self.flatten:
            out = out.view(coords.shape[:-1] + (-1,))

        if self.pad:
            out = F.pad(out, [0, self.dim_out - out.shape[-1]])
        return out
    
def transfer(
        model: nn.Module,
        state_dict: Union[str, Dict[str, th.Tensor]],
        prefix_map: Optional[Dict[str, str]] = None,
        substrs: Optional[Iterable[str]] = None,
        strict: bool = False,
        freeze: bool = False,
        verbose: bool = False
):
    """
    transfer weights to model from state_dict, optionally
    rewriting weight named according to `prefix_map`.
    Furthermore, it is possible to filter the entries in
    `state_dict` according to membership in`substrs`.
    """

    if not isinstance(state_dict, Mapping):
        # assume `state_dict` is a path
        state_dict = th.load(state_dict)
    print('keys', list(state_dict.keys()))

    if prefix_map is None:
        prefix_map = {}

    def _replace_prefix(s: str, p: Dict[str, str]):
        for src, dst in p.items():
            if not s.startswith(src):
                continue
            s = dst + s[len(src):]
        return s

    renamed_state_dict = {
        _replace_prefix(k, prefix_map): v
        for (k, v) in state_dict.items()
    }

    if substrs is not None:
        # filter by substrs
        renamed_state_dict = {k: v for (k, v) in renamed_state_dict.items()
                              if any(s in k for s in substrs)}

    if verbose:
        source_keys = list(renamed_state_dict.keys())
        target_keys = [k for (k, v) in model.named_parameters()]
        update_keys = set(source_keys).intersection(target_keys)
        print(F'source = {source_keys}')
        print(F'target = {target_keys}')
        print(F'update = {update_keys}')
    out = model.load_state_dict(renamed_state_dict, strict)
    if freeze:
        for k, v in model.named_parameters():
            if k in out.missing_keys:
                continue
            if k in out.unexpected_keys:
                continue
            v.requires_grad_(False)
            # v.eval()
    return out