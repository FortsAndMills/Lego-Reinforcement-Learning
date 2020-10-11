import pytest

import gym
import gym.spaces
import numpy as np

class DummyEnv():
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=np.zeros((7,4)), high=np.ones((7,4)), shape=(7,4))
        self.action_space = gym.spaces.Discrete(5)

    def reset(self):
        self.state = np.zeros((7,4))
        self.state[3] = 1
        return self.state
    
    def step(self, a):
        self.state += a
        return self.state, 1 - np.abs(self.state).mean(), self.state[0, 0] == 1, {}

class DummyContEnv(DummyEnv):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=np.zeros((7,4)), high=np.ones((7,4)), shape=(7,4))

from LegoRL import *
from LegoRL.representations.standard import State, Action, Reward, Discount, Flag

def test_buffers():
    env = DummyEnv()
    system = System(env)

    # test Storage --------------------------------------
    data = Storage(b = system.mdp[Reward]([3, 4, 5]), c = system.mdp[State](np.zeros((3, 7, 4))))
    data.d = system.mdp[Discount]([-1, -2, -3])
    subdata = data[("b", "d")]
    assert isinstance(subdata, Storage)
    assert subdata.keys() == {"b", "d"}

    # test ReplayBuffer --------------------------------------
    replay = ReplayBuffer(system, capacity=4)
    replay.store(data)
    assert len(replay) == 3
    assert replay.at([2]).keys() == {"b", "c", "d"}
    assert not hasattr(replay.at([1])["b"], "_tensor")
    assert replay.at([1])["b"].numpy == np.array([4])
    assert replay.at([0, 2])["c"].numpy.shape == (2, 7, 4)

    idx = replay.store(data)
    assert len(replay) == 4
    assert idx == [3, 0, 1]
    assert replay.at([1])["b"].numpy == np.array([5])

def test_exploration1():
    env = DummyEnv()
    system = System(env)

    actions = system.mdp[Action](np.array([0, 3, 2, 0, 4]))

    # test Storage ------------------------------------------
    assert (actions.tensor.cpu().numpy() == np.array([0, 3, 2, 0, 4])).all()
    actions[3] = 3
    assert (actions.tensor.cpu().numpy() == np.array([0, 3, 2, 3, 4])).all()

    # test eGreedy ------------------------------------------
    egreedy = eGreedy(system)
    new_actions = egreedy(actions)
    assert (actions.numpy == np.array([0, 3, 2, 3, 4])).all()

def test_exploration2():
    env = DummyContEnv()
    system = System(env)
    
    actions = system.mdp[Action](np.ones((2,7,4)))
    is_start = system.mdp[Flag](np.array([0, 1]))

    # test mdp_config ----------------------------------------
    assert (system.mdp.rescale_action(-np.ones((7,4))) == np.zeros((7,4))).all()
    assert (system.mdp.rescale_action(np.ones((7,4))) == np.ones((7,4))).all()
    
    # test clippedNoise --------------------------------------
    clipnoise = ClippedNoise(system)
    new_actions = clipnoise(actions)
    assert (actions.numpy == 1).all()
    assert (new_actions.numpy - 1).std() >= 0.01
    assert np.abs(new_actions.numpy - 1).max() <= clipnoise.cliprange
    assert (new_actions.numpy != 1).all()

    # test OUnoise -------------------------------------------
    ounoise = OUnoise(system)
    new_actions = ounoise(actions, is_start)
    assert (actions.numpy == 1).all()
    assert (new_actions.numpy != 1).all()

def test_runner():
    env = DummyEnv()
    system = System(env, gamma=0.9)

    actions = system.mdp[Action](np.array([2]))

    # test Runner ------------------------------------------
    runner = Runner(system)
    transition = runner.step(actions)
    assert (transition.states.numpy != transition.next_states.numpy).any()
    
    copy_state = transition.states.numpy.copy()
    copy_next_state = transition.next_states.numpy.copy()
    transition2 = runner.step(actions)
    assert (transition.states.numpy == copy_state).all()
    assert (transition.next_states.numpy == copy_next_state).all()

    # test Latency ------------------------------------------
    copy_state2 = transition2.states.numpy.copy()
    copy_next_state2 = transition2.next_states.numpy.copy()

    latency = NstepLatency(system, n_steps=3)
    assert latency.add(transition) is None
    assert latency.add(transition2) is None

    transition3 = runner.step(actions)
    copy_state3 = transition3.states.numpy.copy()
    copy_next_state3 = transition3.next_states.numpy.copy()

    latent = latency.add(transition3)
    assert latent is not None
    assert (latent.states.numpy == copy_state).all()
    assert (latent.next_states.numpy == copy_next_state3).all()
    assert np.allclose(latent.rewards.numpy, transition.rewards.numpy + 
                                            system.mdp.gamma * transition2.rewards.numpy +
                                            (system.mdp.gamma**2) * transition3.rewards.numpy, atol=1e-2)
    assert np.allclose(latent.discounts.numpy, system.mdp.gamma**3, atol=1e-3)
