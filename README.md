# Lego-Reinforcement-Learning

**Current plan:**
- [x] Rainbow DQN
- [x] Quantile Regression
- [x] A2C
- [X] PPO
- [x] DDPG
- [x] TD3, SAC
- [ ] Recurrent networks support
- [ ] Reproduce RND, ICM (again)
- [ ] Multi-reward / Multi-gamma support
- [ ] World models

**11/10/20** - Initial commit of v.2.0 code. Full pipeline of algorithm is now inside the parent class :( Though it makes a bit clearer what is happening, code is a bit more cumbersome; mostly technical parts of algorithms are put inside the library files. There is a small-small-small hope that this approach will finally be flexible enough! The first goal will be to finally beat some *basic* benchmarks, including continuous control tasks: tested PPO, [DDPG](https://arxiv.org/abs/1509.02971), [TD3](https://arxiv.org/pdf/1802.09477) and [SAC](https://arxiv.org/abs/1801.01290) on Pendulum, latter three even work.

Also pushed initial commit for [theory book](https://github.com/FortsAndMills/RL-Theory-book) (rus): intro chapters, evolutionary algorithms, dyn. programming basics, TD-learning basics, DQN, DQN+modifications. Hoping to keep up with MSU RL course... 

**18/10/20** - [Theory book](https://github.com/FortsAndMills/RL-Theory-book): added Distributional RL, ch. 4.3 (seems like now the hardest part is done =D). Initiated Demo Project 1: Mario. Let's try again, now with fixed oracle (no negative penalty for moving left) and bugfixed reward function. More details [here](https://github.com/FortsAndMills/Lego-Reinforcement-Learning/tree/master/Demo%20Projects).
