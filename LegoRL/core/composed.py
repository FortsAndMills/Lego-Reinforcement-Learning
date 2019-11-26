from LegoRL.core.RLmodule import RLmodule

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

class Reference():
    '''
    Auxiliary class to store reference to another module before system initialization
    It helps with recursive connections between composed modules of framework

    Args: 
        ref - RLmodule or None (in both cases this class does nothing) or str;
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
    
    def __repr__(self):
        if isinstance(self.ref, RLmodule):
            return self.ref.__repr__()
        return f"Reference at RLmodule named {self.ref}. It will be changed to direct link after system initialization"

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

    def __repr__(self):
        if isinstance(self.ref, RLmodule):
            return self.ref.__repr__()
        return f"List of references at RLmodule named {[r.ref for r in self]}. They will be changed to direct links after system initialization"


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
        # just use module name by default
        module.name = name or module.defaultname().lower()
        assert module.name not in self.modules, f"Error: module with name {module.name} already exists"

        # adding it to our dict of modules
        self.modules[module.name] = module

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
            module._initialize()

    def iteration(self):
        """
        Calls each module's iteration method if it is not frozen and it is time
        """
        self.iterations += 1        
        for module in self.modules.values():
            module.performed = False
            if not module.frozen and self.iterations % module.timer == 0:
                module.iteration()

    def visualize(self):
        for module in self.modules.values():
            if not module.frozen and self.iterations % module.timer == 0:
                module.visualize()        
    
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