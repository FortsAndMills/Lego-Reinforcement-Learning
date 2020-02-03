from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference

class Double(RLmodule):
    """
    Value estimation based on decoupled action selection and evaluation:
    V = Q1(argmax Q2)

    Args:
        selector - RLmodule with "Q" method
        evaluator - RLmodule with "Q" method.

    Provides: V
    """
    def __init__(self, selector, evaluator):
        super().__init__()

        assert selector is not evaluator, "sorry, what?"
        self.selector = Reference(selector)
        self.evaluator = Reference(evaluator)

    def V(self, storage, which):
        self.debug("estimates value.")
        chosen_actions = self.selector.Q(storage, which).greedy()
        return self.evaluator.Q(storage, which).gather(chosen_actions)

    def __repr__(self):
        return f"Evaluates value as Q from <{self.evaluator.name}> of actions selected by <{self.selector.name}>"
