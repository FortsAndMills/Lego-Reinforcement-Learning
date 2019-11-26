from LegoRL.losses.optimalCriticLoss import OptimalCriticLoss

# TODO: move to separate modules folder "targets"?
# also advantage estimators should go there?
class DoubleCriticLoss(OptimalCriticLoss):
    """
    Double DQN implementation.
    Based on: https://arxiv.org/abs/1509.06461

    Requires target_critic to have "Q" method.

    Provides: loss, batch_loss
    """                    
    def estimate_next_state(self, batch):
        assert self.target_critic is not self.critic, "Double Critic Loss has two same critics. That does not make sense."
        self.debug("estimates next state.")

        chosen_actions = self.critic.Q(batch, of="next state").greedy()
        return self.target_critic.Q(batch, of="next state").gather(chosen_actions)

    def __repr__(self):
        return f"Calculates Double DQN loss for <{self.critic.name}> using <{self.target_critic.name}> as estimator and data from <{self.sampler.name}>"
