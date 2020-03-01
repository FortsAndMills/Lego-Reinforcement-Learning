from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.mdp_config import MDPconfig
from LegoRL.buffers.storage import Storage

from LegoRL.utils.multiprocessing_env import VecEnv, DummyVecEnv
from collections import defaultdict

import os
import yaml
import pickle
import time
import torch

'''
System class provides communication between all modules inside the agent.
It controls saving and logging.
'''

class System():
    """
    Communication environment for all modules in learning system.
        
    Args:
        agent - RLmodule
        env - gym environment
        make_env - function returning function to create a new instance of environment.
        already_vectorized - tell that env already returns lists of obs, r, done, bool
        gamma - discount factor, float from 0 to 1
        folder_name - folder name to save model, str or None
        save_timer - timer for saving models, int
        rare_logs_timer - timer for computing expensive logs like average magnitude.
    """
    def __init__(self, agent, env=None, make_env=None, already_vectorized=False, gamma=1, folder_name=None, save_timer=1000, rare_logs_timer=100):
        # creating environment creation function for runners.
        if env is None and make_env is None: 
            raise Exception("Environment env or function make_env must be provided")
        
        self.env = env
        self.make_env = make_env

        # if environment is given explicitly, create DummyVecEnv shell if needed:
        if not already_vectorized:
            if env is not None:
                if not isinstance(env, VecEnv):
                    self.env = DummyVecEnv([lambda: env])
            else:
                self.env = DummyVecEnv([make_env()])

        # useful constants
        self.mdp = MDPconfig(self.env, gamma)
        self.initial_state_example = self.mdp[Storage](states=self.env.reset())

        # logging
        self.iterations = 0
        self.logger = defaultdict(list)                  # lists of values
        self.logger_times = defaultdict(list)            # lists of timestamps when values were recorded
        self.logger_labels = defaultdict(tuple)          # xaxis and yaxis names
        self.reload_messages = []                        # additional messages like reloading from file
        self.time_for_rare_logs = lambda: self.iterations % rare_logs_timer == 0

        # saving
        self.folder_name = folder_name
        self.save_timer = save_timer

        # debugging mode
        self.debug_on = False
        self._debug_level = 0

        # initialize agent
        assert isinstance(agent, RLmodule), "agent must be an instance of RLmodule"
        self.agent = agent
        self.agent._connect_to_system(self)
        self.agent.initialize()

    def log(self, key, value, y_axis=None, x_axis="training iteration", x_value=None):
        """
        Log one value for given key
        input: key - name of logged value, str
        input: value - storing value, scalar
        input: y_axis - name of y-axis for plotting the value (no drawing if None), str
        input: x_axis - name of x-axis for plotting the value, str
        input: x_value - value of x-axis for this log (system.iterations if None), int
        """
        self.logger[key].append(value)
        self.logger_times[key].append(x_value or self.iterations)
        if y_axis is not None:
            self.logger_labels[key] = (x_axis or "iteration", y_axis)

    def add_message(self, message):
        """
        Stores additional information message like reloading from file.
        input: message - str
        """
        self.reload_messages.append(f"iteration {self.iterations}: " + message) 

    def debug(self, author, message="", open=False, close=False):
        '''
        Prints debugging message if in debug regime
        input: author - name of message sender, str
        input: message - message to output
        input: open - open new spacing, bool
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
            self.log("time", time.time() - start, "seconds")
            
            # visualizing
            start = time.time()
            self.agent.visualize()

            # saving
            if self.folder_name is not None and self.iterations % self.save_timer == 0:
                self.save(self.folder_name)
            self.log("visualization time", time.time() - start)

        self.debug_on = False

    def wallclock(self):
        """
        Returns time in seconds spent for training.
        output: float
        """
        return sum(self.logger["time"])
    
    def viz_wallclock(self):
        """
        Returns time in seconds spent for visualization and saving.
        output: float
        """
        return sum(self.logger["visualization time"])
    
    # saving and loading functions
    def save(self, folder_name=None):
        """saving to file"""
        folder_name = folder_name or self.folder_name
        if folder_name is None:
            raise Exception("Error. Folder name is not provided.")
        
        # creating directory
        os.makedirs(folder_name, exist_ok=True)

        # storing hyperparameters        
        hp = self.agent.hyperparameters()
        with open(folder_name + "/hyperparameters.yaml", 'w') as yaml_file:
            yaml.dump(hp, yaml_file, default_flow_style=False)

        # storing agent scheme        
        with open(folder_name + "/agent.txt", 'w') as f:
            f.write(self.agent.__repr__())

        # saving logs to one file
        f = open(folder_name + "/system", 'wb')
        pickle.dump(self.logger, f)
        pickle.dump(self.logger_times, f)
        pickle.dump(self.logger_labels, f)
        pickle.dump(self.iterations, f)
        pickle.dump(self.reload_messages, f)
        f.close()

        # he can also store something in personal files
        self.agent._save(folder_name)
        
    def load(self, folder_name=None):
        """loading from file"""
        folder_name = folder_name or self.folder_name
        if folder_name is None:
            raise Exception("Error. File name is not provided.")
        
        # reading from common file
        f = open(folder_name + "/system", 'rb')
        self.logger = pickle.load(f)
        self.logger_times = pickle.load(f)
        self.logger_labels = pickle.load(f)
        self.iterations = pickle.load(f)
        self.reload_messages = pickle.load(f)
        f.close()

        # reading from personal files
        self.agent._load(folder_name)

        # logging the fact that we were reloaded
        # it is important as, for example, replay buffers do not store their memory usually
        # so reloads reflect the learning procedure.
        self.add_message("reloaded (replay buffers are lost)")