import numpy as np
import torch

def print_batch(batch):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            print("{}: size {}".format(k, str(batch[k].size())))
        elif isinstance(batch[k], list):
            print("{}: # item {}".format(k, len(batch[k])))
        else:
            print("{}: {}".format(k, batch[k]))

def istensor(data):
    return isinstance(data, torch.Tensor)

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim)-1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def tensor2numpy(ptdata):
    return ptdata.detach().cpu().numpy()

def to_data(ptdata):
    if ptdata is None: return ptdata
    if isinstance(ptdata, list):
        return [tensor2numpy(dt) for dt in ptdata]
    elif isinstance(ptdata, dict):
        return {k:tensor2numpy(dt) for k,dt in ptdata.items()}
    else:
        return tensor2numpy(ptdata)

def where(cond, x1, x2):
    """ Differentiable equivalent of np.where (or tf.where)
        Note that type of three variables should be same.
    Args:
        cond: condition
        x1: selected value if condition is 1 (True)
        x2: selected value if condition is 0 (False)
    """
    return (cond * x1) + ((1-cond) * x2)

def loc2mask(loc, feat_mask):
    B, L = feat_mask.size()
    nfeatstamps = to_data(feat_mask.sum(dim=1))
    loc = to_data(loc)

    mask = np.zeros((B,L))
    for bi in range(B):
        sIdx = int(loc[bi, 0] * nfeatstamps[bi])
        eIdx = int(loc[bi, 1] * nfeatstamps[bi])
        mask[bi, sIdx:eIdx+1] = 1

    return mask

""" Computation helpers """
def apply_on_sequence(layer, inp):
    " For nn.Linear, this fn is DEPRECATED "
    inp = to_contiguous(inp)
    inp_size = list(inp.size())
    output = layer(inp.view(-1, inp_size[-1]))
    output = output.view(*inp_size[:-1], -1)
    return output
