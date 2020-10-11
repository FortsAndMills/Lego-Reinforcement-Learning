from LegoRL.models.model import Model
from LegoRL.buffers.storage import Storage
from LegoRL.representations.policy import Policy
from LegoRL.representations.discretePolicy import DiscretePolicy
from LegoRL.representations.gaussianPolicy import GaussianPolicy

class PolicyNetwork(Model):
    """
    Provides a head for Policy.
    Provides: act
    """
    def __init__(self, par, *args, output=None, **kwargs):
        if par.mdp.space == "discrete":
            output = output or DiscretePolicy
        elif par.mdp.space == "continuous":
            output = output or GaussianPolicy

        assert issubclass(output, Policy), "Error: output representation must be Policy"
        Model.__init__(self, par, output=output, *args, **kwargs)

    def act(self, states, *args, **kwargs):
        distribution = self(states)
        actions = distribution.sample()
        return Storage(actions=actions, distribution=distribution)