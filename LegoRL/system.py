from .RLmodule import *
from .preprocessing.multiprocessing_env import VecEnv, DummyVecEnv, SubprocVecEnv

class System():
    """
    Communication environment for all modules in learning system.
        
    Args:
        agent - RLmodule
        env - gym environment
        make_env - function returning function to create a new instance of environment.
        threads - number of environments to create with make_envs, int
        gamma - discount factor, float from 0 to 1
        file_name - file name to save model, str or None
        save_timer - timer for saving models, int
    """
    def __init__(self, agent, env=None, make_env=None, threads=1, gamma=1, file_name=None, save_timer=1000):
        # creating environment
        if env is not None:            
            # If environment given, create DummyVecEnv shell if needed:
            if isinstance(env, VecEnv):
                self.env = env
            else:
                # TODO: zeroed space error!
                self.env = DummyVecEnv([lambda: env])
        elif make_env is not None:
            # Else create different environment instances.
            try:
                if threads == 1:
                    self.env = DummyVecEnv([make_env()])
                else:
                    self.env = SubprocVecEnv([make_env() for _ in range(threads)])
            except:
                raise Exception("Error during environments creation. Try to run make_env() to find the bug!")
        else:
            raise Exception("Environment env or function make_env is not provided")
        
        # useful updates
        self.gamma = gamma
        self.observation_shape = self.env.observation_space.shape
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.num_actions = self.env.action_space.n
            self.action_shape = tuple()
            self.ActionTensor = LongTensor
        elif isinstance(self.env.action_space, gym.spaces.Continuous):
            self.num_actions = np.array(self.env.action_space.shape).prod()
            self.action_shape = self.env.action_space.shape
            self.ActionTensor = Tensor
        else:
            raise Exception("Error: this action space is not supported!")
        self.Batch = Batch(self.ActionTensor)        

        # logging
        self.iterations = 0
        self.logger = defaultdict(list)
        self.logger_times = defaultdict(list)
        self.logger_labels = defaultdict(tuple)

        # saving
        self.file_name = file_name
        self.save_timer = save_timer

        # debugging
        self.debug_on = False
        self.debug_level = 0

        # initialize agent
        assert isinstance(agent, RLmodule), "agent must be an instance of RLmodule"
        self.agent = agent
        self.agent._connect_to_system(self)
        self.agent.initialize()

    def log(self, key, value, x_axis=None, y_axis=None, x_value=None):
        """
        Log one value for given key
        input: key - name of logged value, str
        input: value - storing value
        input: x_axis - name of axis for plotting the value, str
        input: y_axis - name of axis for plotting the value, str
        input: x_value - value of x_axis for this log (system.iterations if None), int
        """
        self.logger[key].append(value)
        self.logger_times[key].append(x_value or self.iterations)
        if x_axis is not None:
            self.logger_labels[key] = (x_axis, y_axis)

    def debug(self, author, message, level=0):
        '''
        Prints debugging message if in debug regime
        input: author - name of message sender, str
        input: message - message to output
        input: level - grow spacing if +1, close if -1.
        '''
        if self.debug_on:
            if level < 0: self.debug_level += level
            if message: print("  "*self.debug_level + author + ": " + message)
            if level > 0: self.debug_level += level

    def run(self, iterations=1, debug=False):
        """
        Performs one iteration of training.
        All modules 'train' method is called, then 'visualize' to show logs.
        input: iterations - number of iterations, int
        """
        self.debug_on = debug

        for _ in range(iterations):
            self.iterations += 1

            # performing iteration and logging time
            start = time.time()
            self.agent.iteration()                
            self.log("time", time.time() - start, "training iteration", "seconds")
            
            # visualizing
            self.agent.visualize()

            # saving
            if self.file_name is not None and self.iterations % self.save_timer == 0:
                self.save(self.file_name)

        self.debug_on = False 
    
    # saving and loading functions
    def save(self, name=None):
        """saving to file"""
        name = name or self.file_name
        if name is None:
            raise Exception("Error. File name is not provided.")

        # saving logs to one file
        f = open(name, 'wb')
        pickle.dump(self.logger, f)
        pickle.dump(self.logger_times, f)
        pickle.dump(self.logger_labels, f)
        pickle.dump(self.iterations, f)

        # agent can also write something to this file
        self.agent.write(f)
        f.close()

        # he can also store something in personal files
        self.agent.save(name)
        
    def load(self, name=None):
        """loading from file"""
        name = name or self.file_name
        if name is None:
            raise Exception("Error. File name is not provided.")
        
        # reading from common file
        f = open(name, 'rb')
        self.logger = pickle.load(f)
        self.logger_times = pickle.load(f)
        self.logger_labels = pickle.load(f)
        self.iterations = pickle.load(f)
        self.agent.read(f)
        f.close()

        # reading from personal files
        self.agent.load(name)

        # logging the fact that we were reloaded
        # it is important as, for example, replay buffers do not store their memory usually
        # so reloads reflect the learning procedure.
        self.log("reloads iterations", self.iterations)