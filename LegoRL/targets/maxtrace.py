from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.cache import batch_cached
from LegoRL.core.composed import Reference

import torch

class MaxTrace(RLmodule):
    """
    MaxTrace advantage estimator.
    A(s, a) = r(s') + r(s'') + ... V(s_{last})
    
    Args:
        evaluator - RLmodule with "V" method

    Provides: returns
    """
    def __init__(self, evaluator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = Reference(evaluator)

    def returns(self, rollout):
        '''
        Calculates max trace returns, estimating the V of last state using critic.
        input: rollout - Rollout
        output: V
        '''
        self.debug("starts computing max trace returns", open=True)

        with torch.no_grad():
            returns = [self.evaluator.V(rollout, of="last state")]
            for step in reversed(range(rollout.rollout_length)):
                returns.append(returns[-1].one_step(rollout.at(step)))

        self.debug(close=True)
        return type(returns[0]).stack(returns[-1:0:-1])
        
    def __repr__(self):
        return f"Estimates maxtrace returns using <{self.evaluator.name}> to evaluate last state"