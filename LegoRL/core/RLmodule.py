class RLmodule():
    """
    Base class for all modules in this framework
    
    Args:
        parent - RLmodule

    Provides:
        iteration: performing one stage of doing something
        visualize: additional logs and plots drawing
    """
    def __init__(self, parent):
        #self.parent = parent
        self.system = parent.system
        self.name = parent._connect(self)
        self.modules = []

    def _connect(self, child):
        '''
        Adds module to list of child modules
        input: RLmodule
        output: str, name of this module
        '''
        name = self.name + ("." if self.name else "") + type(child).__name__
        existing_names = {module.name for module in self.modules}
        if name in existing_names:
            k = 2
            while name + str(k) in existing_names:
                k += 1
            name = name + str(k)

        self.modules.append(child)
        return name

    def iteration(self):
        '''Override this to provide module behavior'''
        for module in self.modules:
            module.iteration()

    def visualize(self):
        '''Called after each iteration to collect additional logs or plot them'''
        for module in self.modules:
            module.visualize()

    def hyperparameters(self):
        '''
        Returns all hyperparameters that this module introduces to the algorithm
        output: dict
        '''
        hp = {}
        for module in self.modules:
            mhp = module.hyperparameters()
            if mhp:
                hp[module.name] = mhp
        return hp       

    # interface functions ----------------------------------------------------------------
    @property
    def mdp(self):
        '''
        Returns system MDP.
        output: MDP_config
        '''
        return self.system.mdp

    def log(self, *args, **kwargs):
        '''Adds something to logs'''
        self.system.log(*args, **kwargs)

    def save(self, folder_name):
        '''
        Called when saved if something must be stored in separate files 
        (like, neural net state dict)
        '''
        for module in self.modules:
            module.save(folder_name)
        
    def load(self, folder_name):
        '''Called to read from separate files'''
        for module in self.modules:
            module.load(folder_name)

    def __repr__(self):
        '''
        Returns module description.
        output: str
        '''
        if len(self.modules) == 0:
            return f"{self.name} RL module; no description provided."

        s = ""
        for module in self.modules:
            if len(module.modules) > 0:
                s += "<" + module.name + ">:\n"
                descr = module.__repr__()
                for line in descr.split('\n'):
                    s += "    " + line + "\n"
            else:
                s += "<" + module.name + ">: " + module.__repr__() + "\n"
        return s
    