from LegoRL.representations.representation import Representation
from LegoRL.representations.standard import Action

class Policy(Representation):
    '''
    FloatTensor representing policy
    '''
    @property
    def distribution(self):
        '''
        output: torch.Distribution
        '''        
        raise NotImplementedError()

    def sample(self):
        '''
        output: Action
        '''
        return self.mdp[Action](self.distribution.sample())

    def rsample(self):
        '''
        output: Action
        '''
        return self.mdp[Action](self.distribution.rsample())

    def log_prob(self, actions):
        '''
        input: Action
        output: FloatTensor
        '''
        #NamedTensors issue
        component_prob = self.distribution.log_prob(actions.tensor.rename(None))
        return component_prob

    def entropy(self):
        '''
        output: FloatTensor
        '''
        component_entr = self.distribution.entropy()
        return component_entr

    #def __getattr__(self, name):
    #    return getattr(self.distribution, name)

