import torch
from e3nn import o3
from torch_scatter import scatter

from .radial_func import RadialProfile
from .tensor_product_rescale import (
    LinearRS,
    TensorProductRescale,
    sort_irreps_even_first,
)

_MAX_ATOM_TYPE = 21
_RESCALE = True
_USE_BIAS = True


class NodeEmbeddingNetwork(torch.nn.Module):

    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE, bias=True):
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)),
                                      self.irreps_node_embedding, bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)

    def forward(self, node_atom):
        '''
            `node_atom` is a LongTensor.
        '''
        node_atom_onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).float()
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)

        return node_embedding, node_attr, node_atom_onehot


def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output,
                           internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined.
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []

    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))

    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
                    for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
                              irreps_output, instructions,
                              internal_weights=internal_weights,
                              shared_weights=internal_weights,
                              bias=bias, rescale=_RESCALE)
    return tp


class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0

    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num ** 0.5)
        return out

    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)


class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num):
        super().__init__()
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding,
                            bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding,
                                         irreps_edge_attr, irreps_node_embedding,
                                         internal_weights=False, bias=False)
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)

    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0,
                                           dim_size=node_features.shape[0])
        return node_features
