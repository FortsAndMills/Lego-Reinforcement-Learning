class RLmodule():
    """
    Base class for all modules in this framework
    
    Args:
        timer - how often to call "iteration" methods, int
        frozen - if frozen, "iteration" method is never called, bool
    """
    def __init__(self, timer=1, frozen=False):
        self.timer = timer
        self.frozen = frozen
        self.performed = False

    def _connect_to_system(self, system, name=None, all_modules=[]):
        '''
        Auxiliary method to connect container with system.
        input: system - System
        input: name - name for this module
        input: all_modules - list of dictionaries <name-module>, see Composed for usage.
        '''
        self.system = system
        self.name = name

    def log(self, *args, **kwargs):
        '''Adds something to logs'''
        self.system.log(*args, **kwargs)
    
    def debug(self, *args, **kwargs):
        '''Prints debug message if system is in debugging mode'''
        self.system.debug(self.name, *args, **kwargs)

    def _initialize(self):
        '''Called after connecting to system and dereferencing procedure'''
        pass

    def iteration(self):
        '''Called by timer to train the module'''
        pass

    def visualize(self):
        '''Called to draw plots if necessary'''
        pass

    def _write(self, f):
        '''Called to load data from common file'''
        pass
        
    def _read(self, f):
        '''Called to read data from common file'''
        pass

    def _save(self, name):
        '''
        Called when saved if something must be stored in separate files 
        (like, neural net state dict)
        '''
        pass
        
    def _load(self, name):
        '''Called to read from personal files'''
        pass

    @classmethod
    def defaultname(cls):
        '''output: default name of module, str'''
        return cls.__name__

    def __repr__(self):
        return f"{type(self).__name__} module (no description provided)"