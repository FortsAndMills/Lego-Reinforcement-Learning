from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference

class Target(RLmodule):
    """
    Calculates target as r + V(s')
    
    Args:
        value - RLmodule with "V" method

    Provides: target
    """
    def __init__(self, evaluator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.evaluator = Reference(evaluator) 
    
    def returns(self, batch):
        '''
        input: batch - Batch
        output: V
        '''
        return self.evaluator.V(batch, of="next state").one_step(batch)
        
    def __repr__(self):
        return f"Calculates one-step TD target using <{self.evaluator.name}> as next state estimator"