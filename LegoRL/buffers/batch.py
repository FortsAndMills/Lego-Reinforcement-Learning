from LegoRL.buffers.transition import Transition

class Batch(dict):
    """
    This class stores an array of transitions.
        states - numpy array, (batch_size x *observation_shape)
        actions - numpy array (batch_size x *action_shape)
        rewards - (batch_size)
        next_states - numpy array, (batch_size x *observation_shape)
        discounts - (batch_size)
        
    This class is intended for parallel environments (batch_size is num_envs then).
    During interaction, policy can store here additional information.
    It is used for recording (values of q-function) and training (log_probs of policy).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_list(cls, transitions):      
        """
        Creates Batch from list of Transitions
        input: transitions - list of Transition
        output: Batch
        """
        states, actions, rewards, next_states, discounts = zip(*transitions)
        batch = cls(states=states, 
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    discounts=discounts)
        return batch

    def transitions(self):        
        """
        Generator for all transitions in this batch.
        All additional information stored (log_probs, q-values, etc.) will not be yielded
        yields: Transition
        """
        for s, a, r, ns, d in zip(self.states, self.actions, self.rewards, self.next_states, self.discounts):
            yield Transition(s, a, r, ns, d)
    
    def average(self, name):        
        """
        Returns average (weighted if weights are provided) loss for batch of losses.
        input: name - name of loss to average, str
        """
        loss_b = self[name]
        assert loss_b.shape == self.rewards.shape, "Error! Batch loss has wrong shape!"
        
        if "weights" in self:
            assert loss_b.shape == self.weights.shape, "Error! Weights do not correspond to loss shape!"
            return (loss_b * self.weights).sum()
            
        return loss_b.mean()

    def to_torch(self, system):
        '''
        Moves all transition information to PyTorch Tensors.
        All other fields are considered to be stored in PyTorch format.
        '''
        observation_names = (None,) * len(system.observation_shape)
        for key in ["states", "next_states", "observations"]:
            if key in self:
                self[key] = system.FloatTensor(self[key], names=self.names() + observation_names)

        if "rewards" in self:            
            action_names = (None,) * len(system.action_shape)
            self.actions = system.ActionTensor(self.actions, names=self.names() + action_names)
            self.rewards = system.FloatTensor(self.rewards, names=self.names())
            self.discounts = system.FloatTensor(self.discounts, names=self.names())
        return self
    
    def to_numpy(self):
        '''
        Moves all transition information to numpy array.
        All other fields are considered to be stored in PyTorch format.
        '''
        for key in ["observations", "states", "actions", "rewards", "next_states", "discounts"]:
            if key in self:
                self[key] = self[key].detach().cpu().numpy()
        return self

    @property
    def last_states(self):
        return self.next_states

    @staticmethod
    def names():
        return ("batch",)

    @property
    def batch_size(self):
        return self.states.shape[0]
    
    @property
    def shape(self):
        return (self.batch_size,)

    def __len__(self):
        return self.batch_size

    def __repr__(self):
        additional = self.keys() - {"states", "actions", "rewards", "next_states", "discounts"}
        return f"Batch of size {self.batch_size}. Additional information stored: {additional}"