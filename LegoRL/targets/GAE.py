from LegoRL.core.cache import storage_cached
from LegoRL.representations.representation import Which
from LegoRL.buffers.storage import stack, Storage
from LegoRL.targets.maxtrace import MaxTrace

import torch

class GAE(MaxTrace):
    """
    Generalized Advantage Estimation (GAE) upgrade of A2C.
    Based on: https://arxiv.org/abs/1506.02438
    
    Args:
        tau - float, from 0 to 1
        truncated_gae - bool, if true gae will be divided by tau + tau**2 + tau**3 ...
        evaluator - RLmodule with "V" method
        baseline - RLmodule with "V" method (evaluator used if not given)

    Provides: returns, advantage
    """
    def __init__(self, tau=0.95, truncated_gae=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.truncated_gae = truncated_gae

    @storage_cached("returns")
    def _rollout_returns(self, rollout):
        '''
        Calculates GAE return.
        input: RolloutStorage
        output: V
        '''
        self.debug("starts computing GAE returns", open=True)

        with torch.no_grad():
            values = self.evaluator.V(rollout, Which.all)
            
            returns = []
            gae = 0
            normalizing = 1
            for step in reversed(range(rollout.rollout_length)):
                advantage = values[step + 1].one_step(rollout.rewards[step], rollout.discounts[step]).subtract_v(values[step])
                
                # TODO: special operation for ensembling?
                gae = advantage.tensor + rollout.discounts[step].tensor * self.tau * gae
                returns.append(type(values)(gae / normalizing + values[step].tensor))
                
                # computes tau + tau**2 + tau**3 + ...
                normalizing += self.tau * normalizing * self.truncated_gae

        self.debug(close=True)
        return stack(returns[::-1])
    
    def hyperparameters(self):
        return {"GAE tau": self.tau, "use truncated GAE": self.truncated_gae}

    def __repr__(self):
        return f"Estimates GAE returns using <{self.evaluator.name}> to evaluate last state and <{self.baseline.name}> as baseline"