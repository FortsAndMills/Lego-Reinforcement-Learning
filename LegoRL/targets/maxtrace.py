from LegoRL.representations.representation import Which
from LegoRL.core.reference import Reference
from LegoRL.core.cache import storage_cached
from LegoRL.buffers.storage import stack, Storage
from LegoRL.targets.rolloutTarget import RolloutTarget

import torch

class MaxTrace(RolloutTarget):
    """
    MaxTrace advantage estimator.
    A(s, a) = r(s') + r(s'') + ... V(s_{last})
    
    Args:
        evaluator - RLmodule with "V" method
        baseline - RLmodule with "V" method (evaluator used if not given)

    Provides: returns, advantage
    """
    def __init__(self, evaluator, baseline=None):
        super().__init__()
        self.evaluator = Reference(evaluator)
        self.baseline = Reference(baseline or evaluator)

    @storage_cached("returns")
    def _rollout_returns(self, rollout):
        '''
        Calculates max trace returns, estimating the V of last state using critic.
        input: RolloutStorage
        output: V
        '''
        self.debug("starts computing max trace returns", open=True)

        with torch.no_grad():
            returns = [self.evaluator.V(rollout, Which.last)]
            for step in reversed(range(rollout.rollout_length)):
                returns.append(returns[-1].one_step(rollout.rewards[step], rollout.discounts[step]))

        self.debug(close=True)        
        return stack(returns[-1:0:-1])

    @storage_cached("advantage")
    def _rollout_advantage(self, rollout):
        return self.returns(rollout).subtract_v(self.baseline.V(rollout))
        
    def __repr__(self):
        return f"Estimates maxtrace returns using <{self.evaluator.name}> to evaluate last state and <{self.baseline.name}> as baseline"