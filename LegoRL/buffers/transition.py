class Transition:
    """
    Transition is a tuple (s, a, r, s', d), where:
        state - numpy array, (*observation_shape)
        action - int or numpy array (*action_shape)
        reward - float or numpy array (*reward_shape)
        next_state - numpy array, (*observation_shape)
        discount - float or numpy array (*reward_shape)
        
    Discount is equal to gamma * (1 - done);
    there can be different gammas for different rewards.

    This class in intended for storing in off-policy replay buffers, so
    it should be memory-efficient.
    """
    def __init__(self, state, action, reward, next_state, discount):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.discount = discount

    def __iter__(self):
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.discount