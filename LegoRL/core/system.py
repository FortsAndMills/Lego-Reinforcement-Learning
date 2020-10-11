from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.mdp_config import MDPconfig
from LegoRL.representations.standard import State

from LegoRL.utils.multiprocessing_env import VecEnv, DummyVecEnv
from collections import defaultdict

import os
import yaml
import pickle
import time

'''
System class provides communication between all modules.
It controls saving and logging.
'''

class System(RLmodule):
    """
    Communication environment for all modules in learning system.
        
    Args:
        env - gym environment
        make_env - function returning function to create a new instance of environment.
        already_vectorized - tell that env already returns lists of obs, r, done, bool
        gamma - discount factor, float from 0 to 1
        folder_name - folder name to save model, str or None
        save_timer - timer for saving models, int
        rare_logs_timer - timer for computing expensive logs like average magnitude.
    """
    def __init__(self, env=None, make_env=None, already_vectorized=False, gamma=0.99, folder_name=None, save_timer=1000, rare_logs_timer=100):
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
        self.system = self
        #self.parent = None
        self.name = ""
        self.modules = []
        self._mdp = MDPconfig(self.env, gamma)
        self.initial_state_example = self.mdp[State](self.env.reset())

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

    @property
    def mdp(self):
        '''
        Returns system MDP.
        output: MDP_config
        '''
        return self._mdp

    def log(self, key, value, y_axis=None):
        """
        Log one value for given key
        input: key - name of logged value, str
        input: value - storing value, scalar
        input: y_axis - name of y-axis for plotting the value (no drawing if None), str
        """
        self.logger[key].append(value)
        self.logger_times[key].append(self.iterations)
        if y_axis is not None:
            self.logger_labels[key] = ("training iteration", y_axis)

    def add_message(self, message):
        """
        Stores additional information message like reloading from file.
        input: message - str
        """
        self.reload_messages.append(f"iteration {self.iterations}: " + message)

    def run(self, iterations=1):
        """
        Performs one iteration of training.
        All modules 'train' method is called, then 'visualize' to show logs.
        input: iterations - number of iterations, int
        """
        for _ in range(iterations):
            self.iterations += 1

            # performing iteration and logging time
            start = time.time()
            self.iteration()      
            self.log("time", time.time() - start, "seconds")
            
            # visualizing
            start = time.time()
            self.visualize()

            # saving
            if self.folder_name is not None and self.iterations % self.save_timer == 0:
                self.save(self.folder_name)
            self.log("visualization time", time.time() - start)

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
        hp = self.hyperparameters()
        with open(folder_name + "/hyperparameters.yaml", 'w') as yaml_file:
            yaml.dump(hp, yaml_file, default_flow_style=False)

        # saving logs to one file
        f = open(folder_name + "/system", 'wb')
        pickle.dump(self.logger, f)
        pickle.dump(self.logger_times, f)
        pickle.dump(self.logger_labels, f)
        pickle.dump(self.iterations, f)
        pickle.dump(self.reload_messages, f)
        f.close()

        # saving to personal files
        super().save(folder_name)
        
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
        super().load(folder_name)

        # logging the fact that we were reloaded
        # it is important as, for example, replay buffers do not store their memory usually
        # so reloads reflect the learning procedure.
        self.add_message("reloaded (replay buffers are lost)")

    def hyperparameters(self):
        '''
        Returns dictionary of all hyperparameters
        output: dict
        '''
        hp = super().hyperparameters()
        hp.update({"gamma": self.mdp.gamma})        
        return hp