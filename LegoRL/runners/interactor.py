from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference
from LegoRL.buffers.storage import Storage
from LegoRL.buffers.rolloutStorage import RolloutStorage

import numpy as np
from itertools import count
from LegoRL.utils.multiprocessing_env import DummyVecEnv, SubprocVecEnv

class Interactor(RLmodule):
    """
    Basic interface for interacting with enviroment

    Args:        
        threads - number of environments to create with make_envs, int
        policy - RLmodule with method "act" or None (random behavior will be used)
        env_max_T - timer limit for environment; will not trigger done=True if limit is reached, int or None

    Provides:
        step - plays one step in env; returns Storage, rewards and lengths of finished episodes
        play - plays one episode; returns RolloutStorage
        act  - writes actions into given Storage; default behavior is random. 
    """
    def __init__(self, threads=1, policy=None, env_max_T=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.policy = Reference(policy or self)
        self._threads = threads
        self.env_max_T = env_max_T

    def _initialize(self):
        if self.system.make_env is not None:
            try:
                if self._threads == 1:
                    self.env = DummyVecEnv([self.system.make_env()])
                else:
                    print(self.name + ": environment initialization...", end="")
                    self.env = SubprocVecEnv([self.system.make_env() for _ in range(self._threads)])
                    print(" Finished.")
            except:
                raise Exception("Error during environments creation. Try to run make_env() to find the bug!")
        elif self.system.env is not None:
            self.env = self.system.env
            self.system.env = None
        else:
            raise Exception("Runner can't create environment (the only instance is already taken?)")

        self._reset()
    
    def _reset(self):
        '''Resets environment.'''        
        self.ob = self.env.reset()
        assert self.ob.max() > self.ob.min(), "BLANK STATE AFTER INIT ERROR"        
        
        self._id = 0                                              # id of created storages
        self.done = np.ones((self.env.num_envs), dtype=np.bool)   # dones from previous step
        self.R = np.zeros((self.env.num_envs), dtype=np.float32)  # cumulative rewards
        self.T = np.zeros((self.env.num_envs))                    # episode lengths
        
    def act(self, storage):
        '''
        Fills actions in constructed transitions. Default behavior is random.
        input: Storage
        '''
        storage.actions = [self.mdp.action_space.sample() for _ in range(len(storage))]

    def step(self):
        """
        Plays one step in parallel environment and updates "observation" property.
        output: transitions - Storage
        output: info - dictionary with additional data
        """
        self.R[self.done] = 0
        self.T[self.done] = 0

        transitions = self.mdp[Storage](states=self.ob, id=self._id, start=self.done)
        self.policy.act(transitions)
        
        try:
            self.ob, r, self.done, info = self.env.step(transitions.actions.numpy)
        except:
            self._reset()
            raise Exception(f"Error during environment step. May be wrong action format? Last actions: {transitions.actions.numpy}")
        
        self._id += 1 
        self.R += r
        self.T += 1

        discounts = self.mdp.gamma * (1 - self.done * (self.T < self.env_max_T if self.env_max_T else 1))
        transitions.update(rewards=r, next_states=self.ob, discounts=discounts)
        
        return transitions, info

    def play(self, render=False, store_frames=True, time_limit=None):     
        """
        Plays full game until first done.        
        If env is vectorized, only first environment's game will be recorded.
        input: render - bool, whether to draw game inline (can be rendered in notebook)
        input: store_frames - bool, whether to store rendered frames in rollout.
        input: time_limit - number of frames limit, int or None
        output: RolloutStorage
        """
        self._reset()

        self._rollout = []
        if store_frames:
            frames = self.env.render(mode = 'rgb_array')
        
        timer = range(time_limit) if time_limit else count()
        for t in timer:
            transitions, _ = self.step()
            
            if store_frames:
                transitions.frames = frames       
                frames = self.env.render(mode = 'rgb_array')

            self._rollout.append(transitions)
            
            if render:
                import matplotlib.pyplot as plt
                from IPython.display import clear_output

                clear_output(wait=True)
                plt.imshow(self.env.render(mode='rgb_array'))
                plt.show()
            
            if self.done[0]:
                break
        
        self._rollout = self.mdp[RolloutStorage].from_list(self._rollout)               
        return self._rollout

    def __repr__(self):
        raise NotImplementedError()
