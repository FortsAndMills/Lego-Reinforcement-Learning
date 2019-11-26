from LegoRL.representations.V import V

import torch

def Q(parclass):
    class Quality(parclass):
        """
        State-action value function (Quality function, Q-function) representation.

        Adds action dimension to representation tensor.
        """
        @classmethod
        def shape(cls, system):
            return torch.Size([system.num_actions]) + super().shape(system)
        
        @classmethod
        def names(cls):
            return ("actions",) + super().names()

        @classmethod
        def constructor(cls):
            dims = super().constructor()
            dims["actions"] = Q
            return dims

        def greedy(self):
            '''
            Returns greedy action.
            output: LongTensor, (*batch_shape)
            '''
            return self.tensor.max(dim="actions").indices
            
        def gather(self, action_b):
            '''
            Selectes this representation for given actions.
            input: action_b - LongTensor, (*batch_shape)
            output: V (actions dimension reduced)
            '''
            # NamedTensors do not yet support "gather", so temporary kludge here...
            actions = action_b.align_as(self.tensor).rename(None)
            d = self.tensor.names.index("actions")
            data = self.tensor.rename(None)
            actions = actions.expand_as(data).select(dim=d, index=0).unsqueeze(dim=d)
            res = data.gather(dim=d, index=actions)
            res = res.refine_names(*self.tensor.names).squeeze("actions")
            return self.construct(res)

            # It should have worked something like this:
            # return construct(self.tensor.gather(dim="actions", index=action_b), without="actions")
        
        def value(self, policy=None):
            '''
            Returns optimal value if policy is None, expectation otherwise.
            input: policy - None or FloatTensor, (*batch_shape, num_actions)
            output: V
            '''
            if policy is None:
                return self.construct(self.tensor.max(dim="actions").values)
            return self.construct((self.tensor * policy.align_as(self.tensor)).sum("actions"))

        def __repr__(self):    
            return f'Q-function for {self.system.num_actions} actions'
    return Quality