from .utils import *

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
        '''Prints debug message if in debugging mode'''
        self.system.debug(self.name, *args, **kwargs)

    def initialize(self):
        '''Called after connecting to system and dereferencing'''
        pass

    def iteration(self):
        '''Called by timer to train the module'''
        pass

    def wait(self):
        '''Called if not trained this iteration'''
        pass

    def visualize(self):
        '''Called to draw plots if necessary'''
        pass

    def write(self, f):
        '''Called to load data from common file'''
        pass
        
    def read(self, f):
        '''Called to read data from common file'''
        pass

    def save(self, name):
        '''
        Called when saved if something must be stored in separate files 
        (like, neural net state dict)
        '''
        pass
        
    def load(self, name):
        '''Called to read from personal files'''
        pass

    def __repr__(self):
        return f"{type(self).__name__} module (no description provided)"

class Reference():
    '''
    Auxiliary class to store reference to another module before system initialization
    It helps with recursive connections between composed modules of framework

    Args: 
        ref - RLmodule or None (then this class does nothing) or str
              path to modules inside Composed modules are indicated with '.' symbol.
    '''
    def __init__(self, ref):
        assert ref is None or isinstance(ref, str) or isinstance(ref, RLmodule), "Reference to another RLmodule must be its name or its instance"
        self.ref = ref

    def _dereference(self, all_modules):
        '''
        Finds the referenced module in the tree or raises an error.
        input: all_modules - list of dictionaries of modules on each level of namespace
                             starting from the current one
        output: RLmodule
        '''
        if isinstance(self.ref, RLmodule) or self.ref is None:
            return self.ref
        elif isinstance(self.ref, str):
            # getting the full path
            path = self.ref.split('.')  

            # checking all levels of namespace from current to the top
            for modules in all_modules:
                # first look up for the root
                t = 0
                while True:
                    # current name must be in current namespace
                    if path[t] in modules:
                        module = modules[path[t]]
                    else:
                        break
                    
                    # if it is and we are not asked to dive deeper, we found our reference!
                    t += 1
                    if t == len(path):
                        return module

                    # otherwise we should look inside deeper namespace
                    # if the current module is not "Composed", then path is not found
                    if hasattr(module, "modules"):
                        modules = module.modules
                    else:
                        break
        raise Exception(f"Fatal error: {self.ref} is a reference to unknown module. System is invalid.")

    def __getattr__(self, x):
        '''
        Convinient usage of this class before dereferencing procedure
        '''
        if isinstance(self.ref, RLmodule):
            return getattr(self.ref, x)
        elif x == "name":
            return self.ref
        raise Exception(f"This is a reference to another module {self.ref} in the system. System is not connected yet.")

class ReferenceList(list):
    '''
    Auxiliary class to store list of references
    Args: 
        ref - list of referencable objects
    '''
    def __init__(self, references):
        assert isinstance(references, list), "ReferenceList must be a list!"
        super().__init__([Reference(ref) for ref in references])

    def _dereference(self, all_modules):
        return [ref._dereference(all_modules) for ref in self]