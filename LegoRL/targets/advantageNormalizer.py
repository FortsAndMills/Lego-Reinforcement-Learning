from LegoRL.core.RLmodule import RLmodule

class AdvantageNormalizer(RLmodule):
    """
    Makes sure advantage has 0 mean and std=1 across the batch
    Popular heuristic for PPO implementations
    """
    def __call__(self, A):
        '''
        Centers advantage from another target generator
        input: V
        output: V
        '''
        adv = A.tensor
        return type(A)((adv - adv.mean()) / (adv.std() + 1e-8))

    def __repr__(self):
        return f"Normalizes advantages in the batch to zero mean and unit std"