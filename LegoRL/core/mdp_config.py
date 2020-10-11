from LegoRL.representations.standard import Embedding

import gym
import torch
import numpy as np
from LegoRL.utils.namedTensorsUtils import torch_one_hot

class MDPconfig():
    """
    Stores parameters of MDP, i.e. state space, action space, reward space.
        
    Args:
        env - gym environment
        gamma - discount factor, float from 0 to 1
    """
    def __init__(self, env, gamma=1):
        # TODO: move EVERYTHING to system?!
        # we are not going to dynamically change mdp! :(
        USE_CUDA = torch.cuda.is_available()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.FloatTensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs).float().cuda() if USE_CUDA else torch.tensor(*args, **kwargs).float()
        self.LongTensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs).cuda() if USE_CUDA else torch.tensor(*args, **kwargs)
        self.BoolTensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs).cuda() if USE_CUDA else torch.tensor(*args, **kwargs)
        
        self.gamma = gamma
        self.observation_shape = env.observation_space.shape
        self.reward_shape = tuple()

        self.action_space = env.action_space
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.space = "discrete"
            self.num_actions = env.action_space.n
            self.action_shape = tuple()
            self.action_description_shape = (self.num_actions,)
            self.action_preprocessing = lambda actions: torch_one_hot(actions, self.num_actions, "features").float()
            self.rescale_action = lambda actions: actions
            self.ActionTensor = self.LongTensor
        elif isinstance(env.action_space, gym.spaces.Box):
            self.space = "continuous"
            self.num_actions = np.array(env.action_space.shape).prod()
            self.action_shape = env.action_space.shape
            self.action_description_shape = self.action_shape
            self.action_preprocessing = lambda actions: actions
            self.rescale_action = lambda actions: (actions + 1) * (env.action_space.high - env.action_space.low) / 2 + env.action_space.low
            self.ActionTensor = self.FloatTensor
        else:
            raise Exception("Error: this action space is not supported!")

        self._representations = dict()
        
    def __getitem__(self, repr):
        '''
        Wrapps class by adding class property "mdp", which provides
        access to MDP parameters.
        input: repr - Representation class or str (than scalar embedding is created)
        output: class
        '''
        clsname = repr if isinstance(repr, str) else repr._default_name()
        if clsname in self._representations:
           return self._representations[clsname]

        repr = Embedding(emb_name=repr) if isinstance(repr, str) else repr
        class MDP_wrapped_repr(repr):
            mdp = self

        self._representations[clsname] = MDP_wrapped_repr      
        return MDP_wrapped_repr

    def __repr__(self):
        if self.space == "discrete":
            action_descr = f"discrete, {self.num_actions} actions"
        else:
            action_descr = f"continuous, shape: {self.action_shape}"

        return \
f"""Gamma: {self.gamma}
Observation space: {self.observation_shape}
Action space: """ + action_descr + f"""
Rewards space: {self.reward_shape}
"""
        