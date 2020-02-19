import torch

'''
As PyTorch Named Tensors proved to be unstable and lacking some basic required features,
they are made customly and moved to this file 
'''

def torch_stack(tensors, dim, new_name):
    # NamedTensors do not yet support "stack"...

    names = (new_name,) + tensors[0].names
    tensors = [tensor.rename(None) for tensor in tensors]
    return torch.stack(tensors, dim=0).refine_names(*names)

def torch_gather(tensor, indexes, without):
    # NamedTensors do not yet support "gather", so temporary kludge here...

    indexes = indexes.align_as(tensor).rename(None)
    d = tensor.names.index(without)
    data = tensor.rename(None)
    indexes = indexes.expand_as(data).select(dim=d, index=0).unsqueeze(dim=d)
    res = data.gather(dim=d, index=indexes)
    res = res.refine_names(*tensor.names).squeeze(without)
    return res

def torch_unflatten(tensor, dim, new_dims):
    # PyTorch Named Tensors 1.3.0 is really unstable :(
    # unflatten do not support reducing dimension case :[
        
    new_dims = list(new_dims)
    if len(new_dims) == 0:
        return tensor.squeeze(dim)
    else:
        return tensor.unflatten(dim, new_dims)

from torch.nn.functional import one_hot
def torch_one_hot(indexes, num_classes, new_name):
    names = indexes.names + (new_name,)
    return one_hot(indexes.rename(None), num_classes).refine_names(*names)

def torch_index(data, index):
    names = data.names
    data = data.rename(None)
    return data[index].refine_names(*names)

def torch_min(t1, t2):
    assert t1.names == t2.names
    names = t1.names
    return torch.min(t1.rename(None), t2.rename(None)).refine_names(*names)