from .RLmodule import *

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

    def iteration(self):
        '''
        Checks if sampled batch has new priorities.
        It is assumed that priorities come from losses, stored inside this batch.
        '''
        if self.sampler._sample is None:
            self.debug("sampler has no sample => no priorities update.")
            return

        # get priorities of batch
        self.debug("asks for new priorities", open=True)
        batch_priorities = self.priority_provider.batch_loss(self.sampler._sample).detach().cpu().numpy()
        assert batch_priorities.shape == (len(self.sampler._sample),)

        # update priorities
        self.sampler.update_priorities(batch_priorities ** self.rp_alpha)
        self.debug("priorities are updated", close=True)

    def __repr__(self):
        return f"Each {self.timer} iteration updates priorities of {self.sampler.name} using priorities from {self.priority_provider.name}"