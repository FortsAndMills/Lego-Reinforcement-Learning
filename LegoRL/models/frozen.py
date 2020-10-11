from LegoRL.core.RLmodule import RLmodule
from LegoRL.models.model import Model

from copy import deepcopy

class HardUpdate():
    def __init__(self, timer=100):
        self.timer = timer

    def __call__(self, module, source):
        if module.system.iterations % self.timer == 0:
            module.net.load_state_dict(source.net.state_dict())

    def hyperparameters(self):
        return {"timer": self.timer}

class SoftUpdate():
    def __init__(self, tau=0.01):
        self.tau = tau

    def __call__(self, module, source):
        for param, source_param in zip(module.net.parameters(), source.net.parameters()):
            param.data.copy_(self.tau * source_param.data + (1 - self.tau) * param.data)

    def hyperparameters(self):
        return {"tau": self.tau}


def Frozen(parent, source, updater=HardUpdate()):  
    '''
    Class fabric for target network heuristic implementation.
    
    TODO
    '''
    parclass = type(source)
    #assert isinstance(source, Model), "Can freeze only RLmodules inherited from Model"

    class FrozenModule(parclass):
        def __init__(self, parent, source, updater):
            self.source = source
            self.updater = updater

            if issubclass(parclass, Model):
                super().__init__(parent, network=None, 
                                     input=source.input_representation, 
                                     output=source.output_representation)
                self.net = deepcopy(source.net)
            else:
                RLmodule.__init__(self, parent)
                for name in dir(source):
                    module = getattr(source, name)
                    if isinstance(module, RLmodule) and module in source.modules:
                        setattr(self, name, Frozen(self, module, updater))

        def update(self):
            if issubclass(parclass, Model):
                self.updater(self, self.source)
            else:
                for module in self.modules:
                    module.update()

        def visualize(self):
            '''Deletes additional logging for this network'''
            pass

        def save(self, folder_name):
            '''Saving is not required for this model'''
            pass

        def load(self, name):
            if issubclass(parclass, Model):
                self.net.load_state_dict(self.source.net.state_dict())
            else:
                RLmodule.load(self, name)

        def hyperparameters(self):
            return self.updater.hyperparameters()

        @classmethod
        def _default_name(cls):
            return cls.__name__ + parclass.__name__

        def __repr__(self):
            return f"Frozen copy of <{self.source.name}>"
    return FrozenModule(parent, source, updater)
