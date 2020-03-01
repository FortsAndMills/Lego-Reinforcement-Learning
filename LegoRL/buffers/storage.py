from LegoRL.representations.representation import Representation, Which
from LegoRL.buffers.transition import Transition
from LegoRL.representations.standard import State, Action, Reward, Discount

import numpy as np
from LegoRL.utils.namedTensorsUtils import torch_stack

def stack(to_stack):
    '''
    Stacks objects in list along the first dimension
    input: to_stack - list of either Representations, numpy arrays or scalars. 
    output: numpy array or Representation
    '''
    # list of representations case
    if isinstance(to_stack[0], Representation):
        tensor = torch_stack([r.tensor for r in to_stack], 0, "timesteps")
        return type(to_stack[0])(tensor)
    
    # list of numpy arrays case
    if isinstance(to_stack[0], np.ndarray):
        return np.stack(to_stack)
    
    # list of floats case
    return np.array(to_stack)

class Storage(dict):
    """
    Main storage class of the framework.
    Keys:
        states - State, (batch_size x *observation_shape)
        actions - Action, (batch_size x *action_shape)
        rewards - Reward, (batch_size x *reward_shape)
        next_states - State, (batch_size x *observation_shape)
        discounts - Discount, (batch_size x *reward_shape)

    Other keys store additional information.
    They are also used as cache for many modules (e.g. forward passes through networks).
    It is used for recording (e.g. values of q-function) and training (e.g. log_probs of policy).
    """ 
    @classmethod
    def from_list(cls, transitions):      
        """
        Creates Storage from list of Transition
        input: transitions - list of Transition
        output: Storage
        """
        states, actions, rewards, next_states, discounts = zip(*transitions)
        return cls(
            states = states, 
            actions = actions,
            rewards = rewards,
            next_states = next_states,
            discounts = discounts
        )

    def transitions(self):        
        """
        Generator for all transitions in this storage.
        All additional information stored (log_probs, q-values, etc.) will not be yielded
        yields: Transition
        """
        for s, a, r, ns, d in zip(self.states.numpy, self.actions.numpy, self.rewards.numpy, self.next_states.numpy, self.discounts.numpy):
            yield Transition(s, a, r, ns, d)

    def crop_states(self, which):
        '''
        Returns subset of states corresponding to marker
        input: which - Which marker:
            current - states
            next - next_states
            last - next_states
            all - both
        output: Representation
        '''
        if which is Which.current:
            return self.states
        if which is Which.next:
            return self.next_states
        if which is Which.last:
            return self.next_states
        if which is Which.all:
            return stack([self.states, self.next_states])
        raise Exception("Error: 'which' marker is None?")

    def average(self, name):        
        """
        Returns average (weighted if weights are provided) loss for batch of losses.
        input: name - name of loss to average, str
        output: scalar (PyTorch) - average loss
        """
        loss_b = self[name]
        assert loss_b.tensor.shape == self.rewards.tensor.shape, "Error! Batch loss has wrong shape!"
        
        if "weights" in self:
            assert loss_b.tensor.shape == self.weights.tensor.shape, "Error! Weights do not correspond to loss shape!"
            return (loss_b.tensor * self.weights.tensor).sum()
            
        return loss_b.tensor.mean()

    def __getitem__(self, key):  
        """Converts data to representations for standard keys."""
        value = super().__getitem__(key)
        if key in {"states", "next_states", "all_states"} and not isinstance(value, State):
            self[key] = value = self.mdp[State](value)
        if key == "actions" and not isinstance(value, Action):
            self[key] = value = self.mdp[Action](value)
        if key == "rewards" and not isinstance(value, Reward):
            self[key] = value = self.mdp[Reward](value)
        if key == "discounts" and not isinstance(value, Discount):
            self[key] = value = self.mdp[Discount](value)
        return value

    # TODO: think
    def from_full_rollout(self, data):
        return data.batch(self.indices)

    # TODO: think
    @property
    def storage_type(self):
        return "storage"

    # interface functions ----------------------------------------------------------------
    @property
    def batch_size(self):    
        """
        Returns the size of batch dimension
        output: int
        """
        return self.states.batch_size

    @property
    def additional_keys(self):  
        """
        Returns all keys with additional information
        output: set of strings
        """
        standard_keys = {"states", "actions", "rewards", "next_states", "discounts"}
        return self.keys() - standard_keys

    def __len__(self):          
        """
        Returns number of transitions stored in batch
        output: int
        """
        return self.batch_size

    def __getattr__(self, key):
        if key in self.keys(): return self[key]
        raise AttributeError()

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def _default_name(cls):
        return "Storage"

    def __repr__(self):
        return f"Batch of size {self.batch_size}. Additional information stored: {self.additional_keys}"