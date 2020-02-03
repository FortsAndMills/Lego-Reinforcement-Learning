from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference, ReferenceList

'''
The key feature of this framework is the ability of uniting several RL modules in one.
This is achieved using the Composed module.

All incorporated modules are stored in "modules" dictionary.
Iteration of training yields calling iteration of training for all incorporated modules.

Composed modules can be combined into arbitrary trees.

To simplify connections between modules from different parts of structure,
all references to other modules can be given as names (strings). 
See demos for usage examples.
'''

class Composed(RLmodule):
    """
    RLmodule incorporating several RLmodules.
    
    Args: modules to incorporate
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

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
        input: name - str or None (default name will be used)
        '''
        assert isinstance(module, RLmodule), "Error: module must be an instance of RLmodule"
        assert not hasattr(module, "system"), "Error: module has already been connected to system"
        assert not module._initialized, "Error: module has already been initialized"
        
        # determining name for this module
        # just use module name by default
        module.name = name or module._defaultname().lower()
        assert module.name not in self.modules, f"Error: module with name {module.name} already exists"

        # adding it to our dict of modules
        self.modules[module.name] = module

    def _connect_to_system(self, system, name=None, all_modules=[]):
        '''
        Auxiliary method to connect container with system.
        input: System
        input: name - name for this module, str or None
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
            attributes = list(module.__dict__.items())
            for attr_name, attr in attributes:
                if isinstance(attr, Reference) or isinstance(attr, ReferenceList):
                    setattr(module, attr_name, attr._dereference(all_modules))
        
        # We must change ids of submodules so that do not interfere in cache
        # although the tree remains untouched, the modules' names should be pathes in the whole tree
        for module in self.modules.values():
            prefix = "" if self.name is None else self.name + "."
            module._connect_to_system(system, prefix + module.name, all_modules)

    def _initialize(self):
        for module in self.modules.values():
            module.initialize()

    def _iteration(self):
        for module in self.modules.values():
            module.iteration()

    def _visualize(self):
        for module in self.modules.values():
            module.visualize()        
    
    # interface functions ----------------------------------------------------------------
    def _write(self, f):
        for module in self.modules.values():
            module._write(f)
        
    def _read(self, f):
        for module in self.modules.values():
            module._read(f)

    def _save(self, name):
        for module in self.modules.values():
            module._save(name)
        
    def _load(self, name):
        for module in self.modules.values():
            module._load(name)

    def __repr__(self):
        s = ""
        for name, module in self.modules.items():
            if isinstance(module, Composed):
                s += "<" + name + ">:\n"
                descr = module.__repr__()
                for line in descr.split('\n'):
                    s += "    " + line + "\n"
            else:
                s += "<" + name + ">: " + module.__repr__() + "\n"
        return s

    def __getitem__(self, name):
        return self.modules[name]

    def __getattr__(self, name):
        if name in self.modules:
            return self.modules[name]
        raise AttributeError()