from LegoRL.core.RLmodule import RLmodule
from LegoRL.representations.standard import Action
from LegoRL.buffers.storage import Storage
   
class RandomPolicy(RLmodule):
    """
    Random Policy

    Provides: act
    """
    def __call__(self, states, *args, **kwargs):
        '''
        Substitues actions to random actions with eps probability
        input: State
        output: Action
        '''
        actions = [self.mdp.action_space.sample() for _ in range(states.batch_size)]
        return Storage(actions = self.mdp[Action](actions))

    def __repr__(self):
        return f"Acts randomly"
