from .DQN_loss import *
    
class DoubleDQN_loss(DQN_loss):
    """
    Double DQN implementation.
    Based on: https://arxiv.org/abs/1509.06461
    """                    
    def estimate_next_state(self, batch):
        self.debug("estimates next state.")

        chosen_actions = self.q_head.q(batch, for_next_state=True).greedy()
        return self.critic.q(batch, for_next_state=True).gather(chosen_actions)

    def __repr__(self):
        return f"Calculates Double DQN loss for {self.q_head.name} using {self.critic.name} as estimator and data from {self.sampler.name}"
