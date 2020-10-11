from LegoRL.core.RLmodule import RLmodule

import matplotlib.pyplot as plt
from IPython.display import clear_output

import numpy as np
from scipy.signal import convolve, gaussian

def smoothen(values, window_size):
    kernel = gaussian(window_size, std=window_size)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, 'valid')

def sliding_average(a, window_size):
    """one-liner for sliding average for array a with window size window_size"""
    return np.convolve(np.concatenate([np.ones((window_size - 1)) * a[0], a]), np.ones((window_size))/window_size, mode='valid')


class Visualizer(RLmodule):
    """
    Basic logger visualizer
    
    Args:
        points_limit - limit of points to draw on one plot, int
        maxmin_points_limit - limit of points to draw for max-min-std plots, int

    Provides: visualize
    """
    def __init__(self, sys, timer=1000, points_limit=500, maxmin_points_limit=100):
        super().__init__(sys)

        self.timer = timer
        self.points_limit = points_limit
        self.maxmin_points_limit = maxmin_points_limit

    def visualize(self):
        """
        Draws plots with logs
        """
        if self.system.iterations % self.timer != 0:
            return
        
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

        min_y = [np.inf]*len(axes)
        max_y = [-np.inf]*len(axes)       
        
        for key, values in self.system.logger.items():
            if key in self.system.logger_labels:
                # id of plot in which we want to draw a line
                plot_idx = plots[self.system.logger_labels[key]]
                ax = axes[plot_idx]

                # we do not want to draw many points
                k = len(values) // self.points_limit + 1
                smooth = smoothen(np.array(values + [values[-1]] * ((k - len(values) % k) % k)), k)[::k]
                patch = ax.plot(self.system.logger_times[key][::k], smooth, lw=2, label=key)
                ax.legend()

                # sigma interval, min-max
                if k > 1:
                    k = len(values) // self.maxmin_points_limit + 1
                    values = np.array(values + [values[-1]] * ((k - len(values) % k) % k))
                    smooth = smoothen(values, k)[::k]
                
                    sigma = values.reshape(-1, k).std(axis=1)
                    ax.fill_between(self.system.logger_times[key][::k],  
                                    smooth - sigma, 
                                    smooth + sigma,
                                    color=patch[0].get_color(), alpha=0.2)
                    
                    ax.plot(self.system.logger_times[key][::k], values.reshape(-1, k).min(axis=1), linestyle = "--", color=patch[0].get_color(), alpha=0.7)
                    ax.plot(self.system.logger_times[key][::k], values.reshape(-1, k).max(axis=1), linestyle = "--", color=patch[0].get_color(), alpha=0.7)

                # plot limits
                q0 = min(smooth)
                q1 = np.quantile(smooth, 0.1)
                q2 = np.quantile(smooth, 0.9)
                q3 = max(smooth)

                q0 -= 0.02 * (q3 - q0)
                q3 += 0.02 * (q3 - q0)

                min_y[plot_idx] = min(max(q0, 2*q1 - q2), min_y[plot_idx])
                max_y[plot_idx] = max(min(q3, 2*q2 - q1), max_y[plot_idx])

                ax.set_ylim(min_y[plot_idx], max_y[plot_idx])
                
        clear_output(wait=True)    
        plt.show(block=False)

    def __repr__(self):
        return f"Plots logs every {self.timer} iteration"
