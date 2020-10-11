from LegoRL.core.RLmodule import RLmodule
from LegoRL.representations.standard import State, Reward
from LegoRL.utils.runningMeanStd import RunningMeanStd

from copy import copy

#TODO: save, load?

class StateNormalizer(RLmodule): 
    """
    Normalizes observations using running mean and standard deviation
    
    Args:
        clip_obs - limits to clip observation, float or None
    """       
    def __init__(self, sys, clip_obs=None):
        super().__init__(sys)

        self.clip_obs = clip_obs
        self.obs_rmv = RunningMeanStd(self.mdp.observation_shape, clip=self.clip_obs)
        
    def apply(self, states):
        '''
        input: State
        output: State
        '''
        return self.mdp[State](self.obs_rmv.apply(states.numpy))

    def update(self, states):
        '''
        input: State
        '''
        self.obs_rmv.update(states.numpy)

    def hyperparameters(self):
        return {"clip_obs": self.clip_obs}

    def __repr__(self):
        return f"Normalizes observations"


class RewardNormalizer(RLmodule): 
    """
    Normalizes reward using running standard deviation of total return
    
    Args:
        clip_ret - limits to clip reward, float or None
    """       
    def __init__(self, sys, clip_rew=None):
        super().__init__(sys)

        self.clip_rew = clip_rew
        self.ret_rmv = RunningMeanStd(self.mdp.reward_shape, clip=self.clip_rew)
        
    def apply(self, rewards):
        '''
        input: Reward
        output: Reward
        '''
        return self.mdp[Reward](self.ret_rmv.apply(rewards.numpy, center=False))

    def update(self, total_return):
        '''
        input: Reward
        '''
        self.ret_rmv.update(total_return.numpy)

    def hyperparameters(self):
        return {"clip_ret": self.clip_rew}

    def __repr__(self):
        return f"Normalizes returns"