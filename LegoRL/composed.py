from .RLmodule import *

class Composed(RLmodule):
    """
    RLmodule incorporating several RLmodules.
    
    Args:
        dictionary of modules
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.iterations = 0

        # storing all modules inside this container
        self.modules = dict()
        for module in args:
            self.add_module(module)
        for name, module in kwargs.items():
            self.add_module(module, name)

    def add_module(self, module, name=None):
        '''
        Adds module to container
        input: module - RLmodule
        input: name - str or None
        '''
        assert isinstance(module, RLmodule), "Error: module must be an instance of RLmodule"
        assert not hasattr(module, "system"), "Error: module has already been connected to system"
        
        # determining name for this module
        name = name or type(module).__name__.lower()
        assert name not in self.modules, f"Error: module with name {name} already exists"
        module.name = name

        # adding it to our dict of modules
        self.modules[name] = module

    def _connect_to_system(self, system, name=None, all_modules=[]):
        '''
        Auxiliary method to connect container with system.
        input: system - System
        input: name - name for this module
        input: all_modules - list of dictionaries of modules on each level of namespace
                             starting from the current one (see recursion usage below)
        '''
        super()._connect_to_system(system, name)

        # We add a new level of namespace and store it at the beggining:
        # if reference is found on lower level then it is accepted.
        all_modules = [self.modules] + all_modules

        # Creating references between modules
        # if some property of module is Reference or ReferenceList,
        # the reference is changed into direct link to desired module
        for module in self.modules.values():
            for attr_name, attr in module.__dict__.items():
                if isinstance(attr, Reference) or isinstance(attr, ReferenceList):
                    setattr(module, attr_name, attr._dereference(all_modules))
        
        # We must change ids of submodules so that do not interfere in cache
        # although the tree remains untouched, the modules' names are pathes in the whole tree
        for module in self.modules.values():
            prefix = "" if self.name is None else self.name + "."
            module._connect_to_system(system, prefix + module.name, all_modules)

    def initialize(self):
        for module in self.modules.values():
            module.initialize()

    def iteration(self):
        """
        Calls each module's iteration method if it is not frozen and it is time
        """
        self.iterations += 1        
        for module in self.modules.values():
            if not module.frozen and self.iterations % module.timer == 0:
                module.iteration()
            else:
                module.wait()

    def visualize(self):
        for module in self.modules.values():
            if not module.frozen and self.iterations % module.timer == 0:
                module.visualize()        
    
    def write(self, f):
        for module in self.modules.values():
            module.write(f)
        
    def read(self, f):
        for module in self.modules.values():
            module.read(f)

    def save(self, name):
        for module in self.modules.values():
            module.save(name)
        
    def load(self, name):
        for module in self.modules.values():
            module.load(name)

    def __repr__(self):
        s = ""
        for name, module in self.modules.items():
            if isinstance(module, Composed):
                s += name + ":\n"
                descr = module.__repr__()
                for line in descr.split('\n'):
                    s += "    " + line + "\n"
            else:
                s += name + ": " + module.__repr__() + "\n"
        return s

    def __getitem__(self, name):
        return self.modules[name]