import torch

class Representation():
    '''
    Base class for representations in this framework.
    Args:
        system - System instance
        tensor - Tensor, containing representation with shape described in class method "shape".

    Consider that blocks of representation work with specific dimension in the tensor.
    We use PyTorch 1.3.0 unstable Named Tensors feature to simplify aligning procedures.
    '''
    def __init__(self, system, tensor):
        assert len(self.shape(system)) == 0 or tensor.shape[-len(self.shape(system)):] == self.shape(system)
        assert None not in tensor.names, "Error: unnamed tensor in representation."

        self.system = system
        self.tensor = tensor
    
    @classmethod
    def from_linear(cls, system, tensor):
        '''
        Representation constructor from output of linear layer.
        input: system - System instance
        input: tensor - Tensor, containing unprocessed representation in its last dimension.
        output: Representation
        '''
        assert tensor.shape[-1] == cls.shape(system).numel()

        # PyTorch Named Tensors 1.3.0 is really unstable :(
        # unflatten do not support reducing dimension :[
        tensor = tensor.refine_names(..., "features")
        if len(cls.shape(system)) == 0:
            tensor = tensor.squeeze("features")
        else:
            tensor = tensor.unflatten("features", list(zip(cls.names(), cls.shape(system))))
        
        return cls(system, tensor)
    
    @classmethod
    def stack(cls, representations):
        '''
        Stacks representations, adding new dimension of batch_shape.
        input: representations - list of Representation
        output: Representation
        '''
        # PyTorch Named Tensors 1.3.0 is really unstable :(
        # torch.stack does not work with NamedTensors :(
        names = ("timesteps",) + representations[0].tensor.names
        tensors = [r.tensor.rename(None) for r in representations]
        tensor = torch.stack(tensors, dim=0).refine_names(*names)
        return cls(representations[0].system, tensor)
    
    @classmethod
    def shape(cls, system):
        '''
        Returns representation shape.
        Consider that tensor also has some batch dimensions at the beginning.
        output: torch.Size
        '''
        raise NotImplementedError()
    
    @classmethod
    def names(cls):
        '''
        Returns names of all dimensions in the representation.
        output: tuple of strings
        '''
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()