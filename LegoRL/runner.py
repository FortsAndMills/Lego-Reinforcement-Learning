from .RLmodule import *

class Runner(RLmodule):
    """
    Basic interface for interacting with enviroment

    Args:
        policy - RLmodule with method "act" or None (random behavior will be used)

    Provides: observation, was_reset
    """
    def __init__(self, policy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.policy = Reference(policy or self)

        self.initialized = False
        self.frames_done = 0
        self.episodes_done = 0

        self._was_reset = None
        self._observation = None

    @property
    def was_reset(self):
        return self._was_reset
    
    @property
    def observation(self):
        return self._observation

    def initialize(self):       
        # TODO: wtf? create here personal instance!
        self.env = self.system.env

    def wait(self):
        self._was_reset = None
        self._observation = None

    def act(self, state):
        '''
        Default behavior is random.
        Input: state - np.array, (batch_size, *observation_shape)
        Output: action - list, (batch_size)
        '''
        return [self.env.action_space.sample() for _ in range(state.shape[0])]
    
    def iteration(self):
        """
        Makes one step and logs results.
        """
        self.debug("plays one step.", +1)
        start = time.time()
        results = self.step()

        self.log("playing time", time.time() - start, "training iteration", "seconds")
        for res in results:
            self.episodes_done += 1
            self.log("rewards", res, "episode", "reward", self.episodes_done)
            self.log("episode ends", self.frames_done)
        self.debug("", -1)

    def step(self):
        """
        Plays one step in parallel environment and updates "observation" property.
        output: rewards of finished episodes, list of floats
        """
        if not self.initialized:
            self._was_reset = True
            self.ob = self.env.reset()
            assert self.ob.max() > self.ob.min(), "BLANK STATE AFTER INIT ERROR"        
            self.R = np.zeros((self.env.num_envs), dtype=np.float32)
            self.initialized = True
        else:
            self._was_reset = False        

        a = self.policy.act(self.ob)
        
        try:
            self.next_ob, r, done, info = self.env.step(a)
        except:
            self.initialized = False
            raise Exception("Error during environment step. May be wrong action format? Last actions: {}".format(a))
        
        self._observation = TransitionBatch(self.ob, a, r, self.next_ob, done)
        
        self.R += r
        results = self.R[done]
        self.R[done] = 0

        self.ob = self.next_ob
        self.frames_done += self.env.num_envs
        
        return results

    def play(self, render=False, record=False):
        """
        Resets environment and play one game.
        If env is vectorized, only first environment's game will be recorded.
        input: render - bool, whether to draw game inline (can be rendered in notebook)
        input: record - bool, whether to store the game and show
        output: cumulative reward, float
        """
        self.initialized = False   
        
        if record:
            self.record = defaultdict(list)
            # TODO: what? self.record["frames"].append(self.env.render(mode = 'rgb_array'))
        
        for t in count():
            results = self.step()
            
            if record:                
                self.record["frames"].append(self.env.render(mode = 'rgb_array'))
                self.record["reward"].append(self._observation.reward[0])
            if render:
                import matplotlib.pyplot as plt
                from IPython.display import clear_output

                clear_output(wait=True)
                plt.imshow(self.env.render(mode='rgb_array'))
                plt.show()
            
            if self._observation.done[0]:
                break        
        return results[0]

    def write(self, f):
        pickle.dump(self.frames_done, f)
        pickle.dump(self.episodes_done, f)
        
    def read(self, f):
        self.frames_done = pickle.load(f)
        self.episodes_done = pickle.load(f)

    def __repr__(self):
        # TODO: bring back {self.env.num_envs}
        return f"Makes step in parallel environments each {self.timer} iteration using {self.policy.name} policy"
