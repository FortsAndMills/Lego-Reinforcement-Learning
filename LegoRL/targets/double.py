from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference

class Double(RLmodule):
    """
    Value estimation based on decoupled action selection and evaluation.

    Args:
        selector - RLmodule with "Q" method
        evaluator - RLmodule with "Q" method.

    Provides: V
    """
    def __init__(self, selector, evaluator, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert selector is not evaluator, "what?"
        self.selector = Reference(selector)
        self.evaluator = Reference(evaluator)

    def V(self, batch, of):
        self.debug("estimates value.")

        chosen_actions = self.selector.Q(batch, of).greedy()
        return self.evaluator.Q(batch, of).gather(chosen_actions)

    def __repr__(self):
        return f"Evaluates value as Q from <{self.evaluator.name}> of actions selected by <{self.selector.name}>"
