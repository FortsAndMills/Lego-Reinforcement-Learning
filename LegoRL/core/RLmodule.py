class RLmodule():
    """
    Base class for all modules in this framework
    
    Args:
        timer - how often to call "iteration" methods, int
        frozen - if frozen, "iteration" method is never called, bool

    Provides:
        initialize: initializes module
        iteration: training of module
        visualize: additional logs and plots drawing
    """
    def __init__(self, timer=1, frozen=False):
        self.timer = timer
        self.frozen = frozen
        
        self._active = lambda: not self.frozen and self.system.iterations % self.timer == 0        
        self._initialized = False  # set to true when initialized     
        self._performed = False    # set to false before each iteration

    def _connect_to_system(self, system, name=None, all_modules=[]):
        '''
        Auxiliary method to connect container with system.
        input: System
        input: name - name for this module, str
        input: all_modules - list of dictionaries <name-module>, see Composed for usage.
        '''
        self.system = system
        self.name = name

    def _initialize(self):
        '''Initialization (override this in subclasses)'''
        pass            

    def initialize(self):
        '''Called after connecting to system and dereferencing procedure'''
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _iteration(self):
        '''What to do each iteration (override this in subclasses)'''
        pass

    def iteration(self):
        '''Called by timer to train the module'''
        self._performed = False
        if self._active(): self._iteration()

    def _visualize(self):
        '''Plot drawing or logging'''
        pass

    def visualize(self):
        '''Called to draw plots or write additional logs'''
        if self._active(): self._visualize()

    @property
    def mdp(self):
        '''
        Returns MDP which this module works with.
        By default, it is system's environment MDP, but some modules may introduce modifications.
        output: MDP_config
        '''
        return self.system.mdp

    # interface functions ----------------------------------------------------------------
    def log(self, *args, **kwargs):
        '''Adds something to logs'''
        self.system.log(*args, **kwargs)
    
    def debug(self, *args, **kwargs):
        '''Prints debug message if system is in debugging mode'''
        self.system.debug(self.name, *args, **kwargs)

    def _save(self, folder_name):
        '''
        Called when saved if something must be stored in separate files 
        (like, neural net state dict)
        '''
        pass
        
    def _load(self, folder_name):
        '''Called to read from separate files'''
        pass

    @classmethod
    def _defaultname(cls):
        '''output: default name of module, str'''
        return cls.__name__

    def __repr__(self):
        return f"{type(self).__name__} module (no description provided)"