from LegoRL.core.RLmodule import RLmodule
from LegoRL.buffers.batch import Batch

from LegoRL.utils.multiprocessing_env import VecEnv, DummyVecEnv
from collections import defaultdict

import pickle
import time
import gym
import torch

'''
System class provides communication between all modules inside the agent.

It controls saving, logging and storing constants like observation and action spaces shapes.
'''

class System():
    """
    Communication environment for all modules in learning system.
        
    Args:
        agent - RLmodule
        env - gym environment
        make_env - function returning function to create a new instance of environment.
        gamma - discount factor, float from 0 to 1
        file_name - file name to save model, str or None
        save_timer - timer for saving models, int
        rare_logs_timer - timer for computing expensive logs like average magnitude.
    """
    def __init__(self, agent, env=None, make_env=None, gamma=1, file_name=None, save_timer=1000, rare_logs_timer=100):
        # creating environment creation function for runners.
        if env is None and make_env is None: 
            raise Exception("Environment env or function make_env must be provided")
        
        self.env = None
        self.make_env = make_env

        # If environment is given explicitly, create DummyVecEnv shell if needed:
        if env is not None:
            if isinstance(env, VecEnv):
                self.env = env
            else:
                self.env = DummyVecEnv([lambda: env])
        else:
            self.env = DummyVecEnv([make_env()])

        # useful constants
        self.USE_CUDA = torch.cuda.is_available()
        self.FloatTensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs).float().cuda() if self.USE_CUDA else torch.tensor(*args, **kwargs).float()
        self.LongTensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs).cuda() if self.USE_CUDA else torch.tensor(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"    
        
        # useful updates
        self.gamma = gamma
        self.observation_shape = self.env.observation_space.shape
        self.action_space = self.env.action_space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.num_actions = self.env.action_space.n
            self.action_shape = tuple()
            self.ActionTensor = self.LongTensor
        elif isinstance(self.env.action_space, gym.spaces.Continuous):
            self.num_actions = np.array(self.env.action_space.shape).prod()
            self.action_shape = self.env.action_space.shape
            self.ActionTensor = self.FloatTensor
        else:
            raise Exception("Error: this action space is not supported!")
        self.initial_state_example = Batch(states=self.env.reset()).to_torch(self)

        # logging
        self.iterations = 0
        self.logger = defaultdict(list)
        self.logger_times = defaultdict(list)
        self.logger_labels = defaultdict(tuple)
        self.reload_messages = []
        self.time_for_rare_logs = lambda: self.iterations % rare_logs_timer == 0

        # saving
        self.file_name = file_name
        self.save_timer = save_timer

        # debugging
        self.debug_on = False
        self._debug_level = 0

        # initialize agent
        assert isinstance(agent, RLmodule), "agent must be an instance of RLmodule"
        self.agent = agent
        self.agent._connect_to_system(self)
        self.agent._initialize()

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

    def add_message(self, message):
        self.reload_messages.append(f"iteration {self.iterations}: " + message) 

    def debug(self, author, message="", open=False, close=False):
        '''
        Prints debugging message if in debug regime
        input: author - name of message sender, str
        input: message - message to output
        input: open - grow spacing, bool
        input: close - close previous spacing, bool
        '''
        assert message or open or close, "Debug empty message error!"
        if self.debug_on:
            if close: self._debug_level -= 1
            if message: print("  "*self._debug_level + author + ": " + message)
            if open: self._debug_level += 1

    def run(self, iterations=1, debug=False):
        """
        Performs one iteration of training.
        All modules 'train' method is called, then 'visualize' to show logs.
        input: iterations - number of iterations, int
        input: debug - whether to pring debug messages, bool
        """
        self.debug_on = debug

        for _ in range(iterations):
            assert self.agent.timer == 1, "Agent's timer is not 1, no sense in empty iterations"
            assert not self.agent.frozen, "Agent is frozen"

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

    def wallclock(self):
        """
        Returns time in seconds spent for training.
        output: float
        """
        return sum(self.logger["time"])
    
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
        pickle.dump(self.reload_messages, f)

        # agent can also write something to this file
        self.agent._write(f)
        f.close()

        # he can also store something in personal files
        self.agent._save(name)
        
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
        self.reload_messages = pickle.load(f)
        self.agent._read(f)
        f.close()

        # reading from personal files
        self.agent._load(name)

        # logging the fact that we were reloaded
        # it is important as, for example, replay buffers do not store their memory usually
        # so reloads reflect the learning procedure.
        self.add_message("reloaded (replay buffers are lost)")