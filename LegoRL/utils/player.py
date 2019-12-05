from LegoRL.runners.interactor import Interactor

import imageio

class Player(Interactor):
    """
    Plays full games on each iteration to evaluate some policy.
    From time to time these games can be recorded and put to a separate folder.

    Args:
        records_file_name - name of file, str or None
        record_timer - frequency of storing the frames from evaluation, int or None
    """
    def __init__(self, records_file_name=None, record_timer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self._threads == 1, "Player must have one thread."

        self.records_file_name = records_file_name
        self.record_timer = record_timer or self.timer*10
        assert self.record_timer % self.timer == 0, "Error: record timer must be synced with Player timer!"

    def visualize(self):
        """
        Plays one game and logs results.
        """
        store_frames = self.records_file_name is not None and self.system.iterations % self.record_timer == 0

        rollout = self.play(render=False, store_frames=store_frames)
        self.log(self.name + " evaluation", sum(rollout.rewards)[0], "evaluation stages", "reward")

        if store_frames:
            imageio.mimwrite(f'{self.records_file_name} (iter. {self.system.iterations}).mp4', rollout["frames"])

    def __repr__(self):
        return f"Plays full game each {self.timer} iteration using {self.policy.name} policy"
