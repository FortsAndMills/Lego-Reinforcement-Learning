# Project 1: Mario

**Algorithm**: PPO, ~55 hours, ~2M frames, ~2K games played; gamma = 0.98, reward function:

| Reward                   | Coeff. |
| -------------            |:------:|
| score                    | 0.01   |
| life loss                | -5     |
| move-to-the-right oracle | +0.01  |

**Results**: stuck in the middle of level 2 (?!) because of mushroom:

![](https://github.com/FortsAndMills/Lego-Reinforcement-Learning/blob/master/Demo%20Projects/results/Mario_stuck2.jpg)

**Fun fact**: Average mushrooms per game when training with score reward raised to 0.8 from 0.1 of random policy. After stucking in this local optima, this plot dropped back to 0.1! No more eating mushrooms o_O

Yet the reward is now in plato, here is the last game played:

![](https://github.com/FortsAndMills/Lego-Reinforcement-Learning/blob/master/Demo%20Projects/results/Mario%20PPO%20iter.%205000000.gif)

Good news: he can shoot turtles! (it gives Mario a lot of points). Bad news: he is stuck again  because of long pit :(

# Project 2.1: Ant (PyBullet)

**Algorithm**: TD3, ~13 hours, ~2M frames, ~2K games played;

Rendering is still an issue, but reward indicates that it worked.

![](https://github.com/FortsAndMills/Lego-Reinforcement-Learning/blob/master/Demo%20Projects/results/Ant.gif)

# Project 2.2: Bipedal Walker

**Algorithm**: TD3, ~13 hours, ~2M frames, ~2K games played;

Rendering is still an issue, but reward indicates that it worked.

![](https://github.com/FortsAndMills/Lego-Reinforcement-Learning/blob/master/Demo%20Projects/results/BipedalWalker.gif)
