#!/usr/bin/env python3

from typing import (
    Tuple,
    Iterable,
    Optional,
    List
)

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract_expression
from rl_games.algos_torch.domi_layers import (
    MLP,
    SplitDim,
    get_activation_function,
    get_normalization_function,
    LN_NAMES
)

from icecream import ic

ASSUME_H = False

class LayerNorm1dH(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x1):
        v, m = th.var_mean(x1[..., :-1],
                           dim=-1,
                           keepdim=True,
                           unbiased=False)
        out = (x1 - m) / th.sqrt(v + self.eps)
        return out

class GateModLinear(nn.Module):
    def __init__(self,
                 d_i: int,
                 d_o: int,
                 act_cls='elu',
                 norm_cls='layernorm',
                 gate: bool = True):
        super().__init__()
        self.d_i = d_i
        self.d_o = d_o
        if act_cls == 'elu':
            self.act = nn.ELU()
        else:
            self.act = nn.Identity()
        if norm_cls == 'layernorm':
            self.norm = nn.LayerNorm(d_o, elementwise_affine=False)
        else:
            self.norm = nn.Identity()

        b = 1024 * 10  # batch size
        i = d_o  # output dim
        m = 8  # num modules
        j = d_i  # input dim
        self.gate = gate
        if gate:
            self.f_Wx = contract_expression(
                'bi,bm,mij,bj->bi',
                (b, i),
                (b, m),
                (m, i, j),
                (b, j),
                #    (b,i),
                #    backend='torch'
            )
            self.f_b = contract_expression(
                'bj,bm,mj->bj',
                (b, j),
                (b, m),
                (m, j),
                #    (b,j),
                #    backend='torch'
            )
        else:
            self.f_Wx = contract_expression(
                'bm,mij,bj->bi',
                (b, m),
                (m, i, j),
                (b, j),
            )
            self.f_b = contract_expression(
                'bm,mj->bj',
                (b, m),
                (m, j)
            )

    def extra_repr(self):
        return F'{self.d_i}->{self.d_o}'

    def forward(self,
                x: th.Tensor,
                # weight/bias (modules)
                Ws: th.Tensor,
                bs: th.Tensor,

                # module selection probabilities
                pW: th.Tensor,
                pb: th.Tensor,

                # output gates
                gW: Optional[th.Tensor] = None,
                gb: Optional[th.Tensor] = None):
        if self.gate:
            Wx = self.f_Wx(gW, pW, Ws, x)
            b = self.f_b(gb, pb, bs)
        else:
            Wx = self.f_Wx(pW, Ws, x)
            b = self.f_b(pb, bs)
        # Wx = th.einsum('...i, ...m, mij, ...j -> ...i', gW, pW, Ws, x)
        # b = th.einsum('...j, ...m, mj -> ...j', gb, pb, bs)
        return self.act(self.norm(Wx + b))

class GateNormModLinear(nn.Module):
    def __init__(self, d_i: int, d_o: int,
                 act_cls='tanh',
                 norm_cls='layernorm',
                 gate: bool = True):
        super().__init__()
        self.d_i = d_i
        self.d_o = d_o
        if act_cls == 'elu':
            self.act = nn.ELU()
        else:
            self.act = nn.Identity()
        if norm_cls == 'layernorm':
            self.norm = nn.LayerNorm(d_o, elementwise_affine=False)
        else:
            self.norm = nn.Identity()

        b = 1024 * 10  # batch size
        i = d_o  # output dim
        m = 8  # num modules
        j = d_i  # input dim
        self.f_Wx = contract_expression('bm,mij,bj->bi',
                                        (b, m),
                                        (m, i, j),
                                        (b, j),
                                        #    (b,i),
                                        #    backend='torch'
                                        )
        self.f_b = contract_expression('bm,mj->bj',
                                       (b, m),
                                       (m, j),
                                       #    (b,j),
                                       #    backend='torch'
                                       )

    def extra_repr(self):
        return F'{self.d_i}->{self.d_o}'

    def forward(self,
                x: th.Tensor,
                # weight/bias (modules)
                Ws: th.Tensor,
                bs: th.Tensor,

                # module selection probabilities
                pW: th.Tensor,
                pb: th.Tensor,

                # output gates
                gW: th.Tensor,
                gb: th.Tensor):
        Wx = self.f_Wx(pW, Ws, x)
        b = self.f_b(pb, bs)
        # Wx = th.einsum('...i, ...m, mij, ...j -> ...i', gW, pW, Ws, x)
        # b = th.einsum('...j, ...m, mj -> ...j', gb, pb, bs)
        # gb is centered at 1
        return self.act(gW*self.norm(Wx + b)+(gb-1))
    
class GateModMLP(nn.Module):
    def __init__(self,
                 dims: Tuple[int, ...],
                 gate: bool = True,
                 after_norm:bool=False):
        super().__init__()
        print(F'GateModMLP got dims = {dims}')
        num_layer: int = len(dims) - 1
        last_idx: int = num_layer - 1
        layer_cls = GateModLinear if not after_norm else GateNormModLinear
        self.gate = gate
        self.layers = nn.ModuleList([
            layer_cls(d_i, d_o,
                          act_cls='elu' if (l != last_idx) else 'none',
                          norm_cls='layernorm' if (l != last_idx) else 'none',
                          gate=gate
                          )
            for l, (d_i, d_o) in enumerate(zip(dims[:-1], dims[1:]))
        ])

    def forward(self, x: th.Tensor,
                # weight/bias (modules)
                # shape: Lx(M,Di,Do)
                Ws: List[th.Tensor],
                # shape: Lx(M,Do)
                bs: List[th.Tensor],

                # module selection probabilities
                # shape: (...,L,M)
                pWs: th.Tensor,
                # shape: (...,L,M)
                pbs: th.Tensor,

                # output gates
                # shape: Lx(..., Do)
                gWs: Optional[th.Tensor] = None,
                # shape: Lx(..., Do)
                gbs: Optional[th.Tensor] = None):
        if self.gate:
            for layer, W, b, pW, pb, gW, gb in zip(self.layers,
                                                   Ws, bs, pWs, pbs, gWs, gbs):
                x = layer(x, W, b, pW, pb, gW.squeeze(dim=-1), gb)
        else:
            for layer, W, b, pW, pb in zip(self.layers, Ws, bs, pWs, pbs):
                x = layer(x, W, b, pW, pb)
        return x


class MAGNet(nn.Module):
    """
    Modulation-And-Gating Network.

    Args:
        dims : Backbone dims (dims[0]=input, dims[-1]=latent task embedding dim)
        num_param : number of independent parameters (#weights+#biases)
        num_module: number of compatible modules per parameter
        out_dims: (for gate only) feature dimensions at each layer
        init_std: initial standard deviations of scale headers
    """

    def __init__(self,
                 dims: Tuple[int, ...],
                 num_param: int,
                 num_module: int,
                 out_dims: Tuple[int, ...],

                 actv: str = 'gelu',
                 norm: str = 'layernorm',
                 affine: bool = True,
                 init_std: float = 0.008
                 ):
        super().__init__()
        self.num_param = num_param
        self.num_module = num_module

        # predict task embeddings
        self.backbone = MLP(dims,
                            actv=actv,
                            norm=norm,
                            affine=affine,
                            activate_output=True)

        # linearly project to logit outputs
        self.logits = nn.Linear(dims[-1],
                                num_param * num_module)

        # also output gates (scales)
        self.out_dims = out_dims
        h_out = sum(out_dims)
        self.scale_header = nn.Linear(dims[-1], h_out)
        self.split_header = SplitDim(self.out_dims)

        # init gate with small std
        with th.no_grad():
            layer = self.scale_header
            nn.init.uniform_(layer.weight, -init_std, init_std)
            nn.init.zeros_(layer.bias)

    def forward(self, x: th.Tensor) -> Tuple[List[th.Tensor],
                                             List[th.Tensor]]:
        z = self.backbone(x)

        # -> module activations
        y = self.logits(z).reshape(*x.shape[:-1],
                                   self.num_param,
                                   self.num_module)
        p = th.softmax(y, dim=-1)
        p = th.unbind(p, dim=-2)

        # -> gates
        scale = 1 + self.scale_header(z)
        scale = self.split_header(scale)
        return p, scale


class RangeActorCritic(nn.Module):
    """
    Range Network (Actor-Critic Variant)
    TODO(ycho): support arbitrary number of subnets?
    """

    def __init__(self,
                 num_module: int,
                 gate: bool,
                 fuse: bool,

                 # --magnet args--
                 mod_dims: Tuple[int, ...],
                 mod_actv: str = 'gelu',
                 mod_norm: str = 'ln',
                 mod_affine: bool = True,
                 init_gate_std: float = 0.008,

                 # --subnet args--
                 # -> MLP/GateModMLP
                 actor_args=[],
                 actor_kwds={},
                 critic_args=[],
                 critic_kwds={},
                 ):
        super().__init__()
        self.gate = gate
        self.fuse = fuse

        # == parameters ==
        paramss = []

        self.num_params = []
        for i in range(num_module):
            ps = []
            for sa, sk in [(actor_args, actor_kwds),
                           (critic_args, critic_kwds)]:
                subnet_affine = sk.pop('affine',
                                       False)
                assert (not subnet_affine)
                subnet = MLP(*sa, **sk,
                             affine=subnet_affine)
                params = list(subnet.parameters())
                # NOTE(ycho): for now, we don't allow buffers.
                bufs = subnet.buffers()
                assert (len(list(bufs)) == 0)
                ps.extend(params)

                # FIXME(ycho): this is a hacky workaround
                # that exploits the fact that
                # actor_args/actor_kwds iterates
                # before `critic_args/critic_kwds`.
                if i == 0:
                    self.num_params.append(len(params))
            params = nn.ParameterList(ps)
            paramss.append(params)
        stacked_params = [th.stack(p, dim=0) for p in zip(*paramss)]
        if fuse:
            ps = [th.stack(p, dim=0) for p in zip(*paramss)]
            stacked_params = [th.cat([W, b[..., None]], dim=-1)
                              for (W, b) in zip(ps[0::2], ps[1::2])]

            self.params = nn.ParameterList(stacked_params)
        else:
            self.params = nn.ParameterList(stacked_params)
        # == modulator ==
        self.modulator = MAGNet(
            mod_dims,
            len(self.params),
            num_module,
            # NOTE(ycho):
            # "p.shape[1]" logic is valid because:
            # for weight-type: (M, d_o, d_i)[1] = d_o
            # for bias-type: (M, d_o)[1] = d_o
            [p.shape[1] for p in self.params],

            actv=mod_actv,
            norm=mod_norm,
            affine=mod_affine,
            init_std=init_gate_std
        )
        # == executor ==
        # TODO(ycho): consider parallelizing
        self.actor_executor = GateModMLP(*actor_args,
                                         **actor_kwds,
                                         gate=gate,
                                        )
        self.critic_executor = GateModMLP(*critic_args,
                                          **critic_kwds,
                                          gate=gate,
                                          )
        self.S = self.num_params[0] 

        S = self.S
        self.q_a = self.params[:S]
        self.q_c = self.params[S:]

    def forward(self,
                z: th.Tensor,
                x: th.Tensor,
                c: Optional[th.Tensor] = None):
        batch_shape = z.shape[:-1]

        # flatten batch dims
        z = z.view(-1, z.shape[-1])
        x = x.view(-1, x.shape[-1])

        probs, scales = self.modulator(z)

        S = self.S
        # q_a, q_c = self.params[:S], self.params[S:]
        # p_a, p_c = probs[:S], probs[S:]
        # g_a, g_c = scales[:S], scales[S:]
        q_a = self.q_a 
        q_c = self.q_c
        p_a = probs[:S]
        p_c = probs[S:]
        g_a = scales[:S]
        g_c = scales[S:]

        y_a = self.actor_executor(
            x, q_a[::2], q_a[1::2], p_a[::2], p_a[1::2], g_a
        )
        y_a = y_a.view(*batch_shape, y_a.shape[-1])
        if c is None:
            c = x
        y_c = self.critic_executor(
            c, q_c[::2], q_c[1::2], p_c[::2], p_c[1::2], g_c
        )
        y_c = y_c.view(*batch_shape, y_c.shape[-1])
        return (y_a, y_c)


def test_range_ac():
    device: str = 'cuda'
    dim_b: int = 512 * 10

    dim_z: int = 128
    dim_x: int = 256
    dim_a: int = 21
    dim_v: int = 4
    dim_h: int = 127

    # dim_z: int = 3
    # dim_x: int = 4
    # dim_a: int = 5
    # dim_v: int = 6
    # dim_h: int = 7

    net = RangeActorCritic(4, False, False,
                            # MAGNet args
                            mod_dims=[dim_z, 4],
                            # subnet MLP args
                            actor_kwds=dict(
                                dims=[dim_x, 512,256,128, dim_a],
                            ),
                            critic_kwds=dict(
                                dims=[dim_x, 512,256,128, dim_v],
                            ))
        # net = th.compile(net) # hmm~~
        # with th.no_grad():
        #     for k, v in net.named_parameters():
        #         v.normal_()
    net.to(device)
    ic(net)
    net(th.randn(1, dim_z).to(device), th.randn(1, dim_x).to(device))
        # for k, v in net.named_parameters():
        #     if v.grad is None:
        #         ic(k, None)
        #     else:
        #         ic(k, v.grad.shape)
        #         ic(v)
        #         ic(v.grad)


def main():
    test_range_ac()


if __name__ == '__main__':
    main()