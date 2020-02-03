from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference

class PrioritiesUpdater(RLmodule):
    """
    Updates priorities in prioritized sampler using some loss as proxy of transitions' importances
    Based on: https://arxiv.org/abs/1511.05952

    Args:
        sampler - RLmodule with "sample" and "update_priorities" methods
        priority_provider - RLmodule with "batch_loss" property
        rp_alpha - float, degree of prioritization, from 0 to 1
    """
    def __init__(self, sampler, priority_provider, rp_alpha=0.6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sampler = Reference(sampler)
        self.priority_provider = Reference(priority_provider)
        self.rp_alpha = rp_alpha

    def _iteration(self):
        '''
        Checks if sampled batch has new priorities.
        It is assumed that priorities come from losses, stored inside this batch.
        '''
        sample = self.sampler.sample()
        if sample is None:
            self.debug("sampler has no sample => no priorities update.")
            return

        # get priorities of batch
        self.debug("asks for new priorities", open=True)
        sample.new_priorities = self.mdp["Priorities"](self.priority_provider.batch_loss(sample).tensor ** self.rp_alpha)
        
        # update priorities
        self.sampler.update_priorities(sample)
        self.debug("priorities are updated", close=True)

    def __repr__(self):
        return f"Updates priorities of <{self.sampler.name}> using priorities from <{self.priority_provider.name}>"