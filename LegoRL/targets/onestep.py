from LegoRL.representations.representation import Which
from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference

class OneStep(RLmodule):
    """
    Calculates target as r + V(s')
    
    Args:
        value - RLmodule with "V" method

    Provides: returns
    """
    def __init__(self, evaluator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.evaluator = Reference(evaluator) 
    
    def returns(self, storage):
        '''
        input: Storage
        output: V
        '''
        return self.evaluator.V(storage, Which.next).one_step(storage.rewards, storage.discounts)
        
    def __repr__(self):
        return f"Calculates one-step TD target using <{self.evaluator.name}> as next state estimator"