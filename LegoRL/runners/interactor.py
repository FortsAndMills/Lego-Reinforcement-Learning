from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference
from LegoRL.buffers.batch import Batch
from LegoRL.buffers.rollout import Rollout

import numpy as np
from itertools import count
from LegoRL.utils.multiprocessing_env import DummyVecEnv, SubprocVecEnv

class Interactor(RLmodule):
    """
    Basic interface for interacting with enviroment

    Args:        
        threads - number of environments to create with make_envs, int
        policy - RLmodule with method "act" or None (random behavior will be used)

    Provides: was_reset
    """
    def __init__(self, threads=1, policy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.policy = Reference(policy or self)

        self._interrupted = False
        self._threads = threads
        self._was_reset = None

    @property
    def was_reset(self):
        return self._was_reset

    def _initialize(self):
        if self.system.make_env is not None:
            # Else create different environment instances.
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
        '''
        Resets environment.
        '''
        self._interrupted = True
        
        self.ob = self.env.reset()
        assert self.ob.max() > self.ob.min(), "BLANK STATE AFTER INIT ERROR"        
        
        self.R = np.zeros((self.env.num_envs), dtype=np.float32)
        self.T = np.zeros((self.env.num_envs), dtype=np.float32)
        
    def act(self, transitions):
        '''
        Fills actions in constructed transitions. Default behavior is random.
        Input: Batch
        '''
        transitions.actions = [self.system.action_space.sample() for _ in range(len(transitions))]

    def step(self):
        """
        Plays one step in parallel environment and updates "observation" property.
        output: transitions - Batch
        output: results - rewards of finished episodes, list of floats
        output: lengths - lengths of finished episodes, list of ints
        """
        self._was_reset = self._interrupted
        self._interrupted = False        
        
        transitions = Batch(states=self.ob)
        self.policy.act(transitions)
        
        try:
            self.ob, r, done, info = self.env.step(transitions.actions)
        except:
            self._reset()
            raise Exception("Error during environment step. May be wrong action format? Last actions: {}".format(a))
        
        transitions.update(rewards=r, next_states=self.ob, discounts=self.system.gamma * (1 - done))
                
        self.R += r
        self.T += 1
        results = self.R[done]
        lengths = self.T[done]
        self.R[done] = 0
        self.T[done] = 0
        
        return transitions, results, lengths

    def play(self, render=False, store_frames=True):     
        """
        Plays full game until first done.        
        If env is vectorized, only first environment's game will be recorded.
        input: render - bool, whether to draw game inline (can be rendered in notebook)
        input: store_frames - bool, whether to store rendered frames in rollout.
        output: Rollout
        """
        self._reset()

        self._rollout = Rollout()
        if store_frames:
            self._rollout.frames = [np.copy(self.env.render(mode = 'rgb_array'))]
        
        for t in count():
            transitions, _, _ = self.step()
            
            self._rollout.append(transitions)
            if store_frames:       
                self._rollout.frames.append(np.copy(self.env.render(mode = 'rgb_array')))
            
            if render:
                import matplotlib.pyplot as plt
                from IPython.display import clear_output

                clear_output(wait=True)
                plt.imshow(self.env.render(mode='rgb_array'))
                plt.show()
            
            if self._rollout.discounts[-1][0] == 0:
                break
        
        self._interrupted = True
               
        return self._rollout

    def __repr__(self):
        raise NotImplementedError()
