from LegoRL.buffers.transition import Transition
from LegoRL.buffers.batch import Batch
from LegoRL.representations.representation import Representation

import torch

class Rollout(Batch):
    """
    This class stores rollouts of transitions.
        states - numpy array, (rollout + 1 x batch_size x *observation_shape)
        actions - numpy array (rollout x batch_size x *action_shape)
        rewards - (rollout x batch_size)
        discounts - (rollout x batch_size)
        
    This class is intended for policy gradient methods and rollouts collections.
    """
    def append(self, batch):
        '''
        Adds batch of size (batch_size) to the end of rollout.
        It is assumed that states of added batch are exactly equal to the last states.
        input: Batch
        '''
        # initializing rollout
        if "observations" not in self:
            self.observations = [batch.states]
            for key, item in batch.items():
                if key != "states" and key != "next_states":
                    self[key] = []

        # adding new transitions
        self.observations.append(batch.next_states)
        for key, item in batch.items():
            if key != "states" and key != "next_states":
                self[key].append(item)

    def transitions(self):        
        """
        Generator for all transitions in this batch.
        All additional information stored (log_probs, q-values, etc.) will not be yielded
        yields: Transition
        """
        for t in range(self.rollout_length):
            for b in range(self.batch_size):
                yield Transition(self.observations[t][b], self.actions[t][b], self.rewards[t][b], self.observations[t+1][b], self.discounts[t][b])
    
    def to_torch(self, system):
        super().to_torch(system)       

        # concatenates all tensors in additional information along time axis
        for key in self:
            if key not in ["observations", "states", "actions", "rewards", "next_states", "discounts"]:
                if isinstance(self[key][0], Representation):
                    self[key] = type(self[key][0]).stack(self[key])
                else:
                    # torch.stack does not work with NamedTensors :(
                    names = ("timesteps",) + self[key][0].names
                    tensors = [tensor.rename(None) for tensor in self[key]]
                    self[key] = torch.stack(tensors, dim=0).refine_names(*names)
        return self

    @property
    def states(self):
        return self.observations[:-1]

    @property
    def last_states(self):
        return self.observations[-1]

    @property
    def rollout_length(self):
        if "observations" in self:
            return len(self.observations) - 1
        return 0

    @property
    def batch_size(self):
        return self.observations[0].shape[0]

    @staticmethod
    def names():
        return ("timesteps", "batch",)
    
    @property
    def shape(self):
        return (self.rollout_length, self.batch_size)

    def at(self, t):
        return Batch(states=self.observations[t], 
                     actions=self.actions[t],
                     rewards=self.rewards[t],
                     next_state=self.observations[t+1],
                     discounts=self.discounts[t])

    def __len__(self):
        return self.rollout_length * self.batch_size

    def __repr__(self):
        additional = self.keys() - {"observations", "actions", "rewards", "discounts"}
        return f"Rollout of length {self.rollout_length} of size {self.batch_size}. Additional information stored: {additional}"