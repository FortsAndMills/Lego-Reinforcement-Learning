from LegoRL.core.RLmodule import RLmodule
from LegoRL.runners.interactor import Interactor

import os
import imageio

class Player(Interactor):
    """
    Plays full games on each iteration to evaluate some policy.
    From time to time these games can be recorded and put to a separate folder.

    Args:
        record_timer - frequency of storing the frames from evaluation, int or None
        time_limit - limitation in steps for one game
    """
    def __init__(self, par, policy, timer=1000, *args, record_timer=None, time_limit=10000, **kwargs):
        super().__init__(par, *args, **kwargs)
        assert self.env.num_envs == 1, "Player must have one thread."

        self.policy = policy
        self.timer = timer
        self.record_timer = record_timer
        self.time_limit = time_limit
        assert self.record_timer is None or self.record_timer % self.timer == 0, "Error: record timer must be synced with Player timer!"
        
        if self.record_timer is not None:
            assert self.system.folder_name is not None, "Player is not able to store videos if system is not provided with folder_name"
            
            self.videos_path = os.path.join(self.system.folder_name, "videos")
            os.makedirs(self.videos_path, exist_ok=True)

    def visualize(self):
        """
        Plays one game and logs results.
        """
        if self.system.iterations % self.timer != 0:
            return
            
        store_frames = self.record_timer is not None and self.system.iterations % self.record_timer == 0

        rollout = self.play(self.policy, render=False, store_frames=store_frames, time_limit=self.time_limit)
        self.log(self.name + " evaluation", sum(rollout.rewards.numpy)[0], "reward")

        if store_frames:
            path = os.path.join(self.videos_path, f'iter. {self.system.iterations}.mp4')
            imageio.mimwrite(path, rollout["frames"])

    def __repr__(self):
        return f"Plays full game each {self.timer} iteration"
