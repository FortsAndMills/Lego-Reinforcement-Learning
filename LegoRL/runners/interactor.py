from LegoRL.core.RLmodule import RLmodule
from LegoRL.buffers.storage import Storage
from LegoRL.representations.standard import State, Reward, Discount, Flag

import numpy as np
from itertools import count
from LegoRL.utils.multiprocessing_env import DummyVecEnv, SubprocVecEnv

class Interactor(RLmodule):
    """
    Basic interface for interacting with enviroment

    Args:        
        threads - number of environments to create with make_envs, int
        env_max_T - timer limit for environment; will not trigger done=True if limit is reached, int or None

    Provides:
        step - plays one step in env; returns Storage, rewards and lengths of finished episodes
        play - plays one episode; returns RolloutStorage
        act  - writes actions into given Storage; default behavior is random. 
    """
    def __init__(self, sys, threads=1, env_max_T=None):
        super().__init__(sys)

        self.env_max_T = env_max_T

        if self.system.make_env is not None:
            try:
                if threads == 1:
                    self.env = DummyVecEnv([self.system.make_env()])
                else:
                    print(self.name + ": environment initialization...", end="")
                    self.env = SubprocVecEnv([self.system.make_env() for _ in range(threads)])
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
        self.states = self.mdp[State](self.env.reset())
        assert self.states.numpy.max() > self.states.numpy.min(), "BLANK STATE AFTER INIT ERROR"        
                                                
        self.done = np.ones((self.env.num_envs), dtype=np.bool)   # dones from previous step
        self.is_start = self.mdp[Flag](self.done)                 # is this step first step in episode
        self.R = self.mdp[Reward](np.zeros((self.env.num_envs), dtype=np.float32))  # cumulative rewards
        self.disc_R = self.mdp[Reward](np.zeros((self.env.num_envs), dtype=np.float32))  # cumulative discounted rewards
        self.T = np.zeros((self.env.num_envs))                    # episode lengths
        
    def _perform_step(self, actions):
        """
        Plays one step in parallel environment and updates "observation" property.
        input: actions - Action
        output: Storage (s, a, r, s', discount)
        """
        self.R[self.done] = 0
        self.disc_R[self.done] = 0
        self.T[self.done] = 0

        storage = self.get_state()

        try:
            rescaled_actions = self.mdp.rescale_action(actions.numpy)
            self.states, self.r, self.done, self.info = self.env.step(rescaled_actions)
        except:
            self._reset()
            raise Exception(f"Error during environment step. May be wrong action format? Last actions: {actions.numpy}")
        
        self.states = self.mdp[State](self.states)
        self.r = self.mdp[Reward](self.r)

        self.R += self.r
        self.disc_R *= self.mdp.gamma
        self.disc_R += self.r
        self.T += 1

        self.interrupt = (self.T < self.env_max_T if self.env_max_T else 1)
        self.discounts = self.mdp[Discount](self.mdp.gamma * (1 - self.done * self.interrupt))
        self.is_start = self.mdp[Flag](self.done)

        storage.update(
            actions = actions,
            rewards = self.r,
            next_states = self.states,
            discounts = self.discounts,
        )
        return storage

    def get_state(self):
        """
        output: Storage
        """
        return Storage(states = self.states, is_start = self.is_start)
        
    def play(self, policy, render=False, store_frames=False, time_limit=None):     
        """
        Plays full game until first done.        
        If env is vectorized, only first environment's game will be recorded.
        input: policy - callable, takes states as input and outputs Storage with "actions" key
        input: render - bool, whether to draw game inline (can be rendered in notebook)
        input: store_frames - bool, whether to store rendered frames in rollout.
        input: time_limit - number of frames limit, int or None
        output: Storage
        """
        self._reset()

        rollout = []
        if store_frames:
            frames = self.env.render(mode = 'rgb_array')
        
        timer = range(time_limit) if time_limit else count()
        for _ in timer:
            storage = self.get_state()

            storage.update(policy(**storage))
            self._perform_step(storage.actions)

            storage.update(rewards=self.r,
                           next_states=self.states,
                           discounts=self.discounts)

            if store_frames:
                storage.update(frames = frames)    
                frames = self.env.render(mode = 'rgb_array')

            rollout.append(storage)
            
            if render:
                import matplotlib.pyplot as plt
                from IPython.display import clear_output

                if not store_frames:
                    frames = self.env.render(mode = 'rgb_array')

                clear_output(wait=True)
                plt.imshow(frames)
                plt.show()
            
            if self.done[0]:
                break
                     
        return Storage.from_list(rollout)   

    def __repr__(self):
        raise NotImplementedError()
