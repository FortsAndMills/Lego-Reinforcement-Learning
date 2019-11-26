from LegoRL.runners.interactor import Interactor

class Player(Interactor):
    """
    Plays full game on each iteration.

    Provides: rollout
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self._threads == 1, "Player must have one thread."

        self._rollout = None

    def iteration(self):
        """
        Makes one step and logs results.
        """
        rollout = self.play()
        self.log("evaluation rewards", sum(rollout.rewards)[0], "evaluation stages", "reward")

    def rollout(self):
        """
        Gives rollout of one game.
        output: Rollout
        """
        if self.performed:
            self.debug("returns same rollout")
            return self._rollout
        self.performed = True

        self.play()
        return self._rollout

    def __repr__(self):
        return f"Plays full game each {self.timer} iteration using {self.policy.name} policy"
