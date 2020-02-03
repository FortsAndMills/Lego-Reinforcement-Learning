from LegoRL.core.reference import Reference
from LegoRL.transformations.head import Head
from LegoRL.representations.standard import Embedding
from LegoRL.buffers.storage import Which

class ForwardDynamics(Head):
    """
    Decoder trying to predict next state in some embedding space.
    Takes as input embedding of state and action.
    
    Provides: curiosity
    """
    @property
    def input_representation(self):
        return self.mdp[Embedding(self.backbone.output_representation.shape() + 
                                  np.prod(self.mdp.action_description_shape))]

    @property
    def output_representation(self):
        return self.backbone.output_representation

    def get_input(self, storage, which):
        return torch.cat([self.backbone(storage).tensor, 
                          self.mdp.action_preprocessing(storage.actions)], dim="features")

    def curiosity(self, storage):
        prediction = self(storage)
        truth = self.backbone(storage, Which.next)
        return prediction.compare(truth)