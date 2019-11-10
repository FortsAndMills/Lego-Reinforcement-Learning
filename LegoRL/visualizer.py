from .RLmodule import *

import matplotlib.pyplot as plt
from IPython.display import clear_output

class Visualizer(RLmodule):
    """
    Basic logger visualizer
    
    Args:
        reward_smoothing - additional reward smoothing, int or None
        points_limit - limit of points to draw on one plot, int

    Provides: visualize
    """
    def __init__(self, timer=100, reward_smoothing=100, points_limit=1000, *args, **kwargs):
        super().__init__(timer=timer, *args, **kwargs)

        self.reward_smoothing = reward_smoothing
        self.points_limit = points_limit

    def _sliding_average(self, a, window_size):
        """one-liner for sliding average for array a with window size window_size"""
        return np.convolve(np.concatenate([np.ones((window_size - 1)) * a[0], a]), np.ones((window_size))/window_size, mode='valid')

    def visualize(self):
        """
        Draws plots with logs
        """
        clear_output(wait=True)    
        
        # getting what plots do we want to draw
        coords = [self.system.logger_labels[key] for key in self.system.logger.keys() if key in self.system.logger_labels]
        k = 0
        plots = {}
        for p in coords:
            if p not in plots:
                plots[p] = k; k += 1
                
        if len(plots) == 0:
            print("No logs in logger yet...")
            return
        
        plt.figure(2, figsize=(16, 4.5 * ((len(plots) + 1) // 2)))
        plt.title('Training...')
        
        # creating plots
        axes = []
        for i, plot_labels in enumerate(plots.keys()):
            axes.append(plt.subplot((len(plots) + 1) // 2, 2, i + 1))
            plt.xlabel(plot_labels[0])
            plt.ylabel(plot_labels[1])
            plt.grid()        
        
        for key, value in self.system.logger.items():
            if key in self.system.logger_labels:
                # id of plot in which we want to draw a line
                ax = axes[plots[self.system.logger_labels[key]]]

                # we do not want to draw many points
                k = len(value) // self.points_limit + 1
                value = np.array(value + [value[-1]] * ((k - len(value) % k) % k))
                ax.plot(self.system.logger_times[key][::k], value.reshape(-1, k).mean(axis=1), label=key)
                ax.legend()
                
                # smoothing main plot!
                if key == "rewards" and self.reward_smoothing is not None:
                    ax.plot(self.system.logger_times[key][::k], self._sliding_average(value, self.reward_smoothing)[::k], label="smoothed rewards")
        plt.show()

    def __repr__(self):
        return f"Plots logs every {self.timer} iteration"
