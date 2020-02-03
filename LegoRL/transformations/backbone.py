from LegoRL.transformations.transformation import Transformation
from LegoRL.representations.standard import State, Embedding

class Backbone(Transformation):
    '''
    FeatureExtractor, transforming state into embedding
    
    Args:
        network - nn.Module
        embedding_size - int
    '''

    def __init__(self, network=None, embedding_size=100):
        super().__init__(network=network)
        self._output_representation = Embedding(embedding_size)

    @property
    def input_representation(self):
        return self.mdp[State]

    @property
    def output_representation(self):
        return self.mdp[self._output_representation]

    def _get_input(self, storage, which):
        return storage.crop_states(which).tensor  

    def __repr__(self):
        return f"Feature extractor network"