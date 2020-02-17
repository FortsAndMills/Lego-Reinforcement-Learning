from LegoRL.representations.representation import Which
from LegoRL.buffers.transition import Transition
from LegoRL.buffers.storage import Storage, stack

class RolloutStorage(Storage):
    """
    This class stores rollouts of transitions.
        all_states - State, (rollout + 1 x batch_size x *observation_shape)
        actions - Action, (rollout x batch_size x *action_shape)
        rewards - Reward, (rollout x batch_size x *reward_shape)
        discounts - Discount, (rollout x batch_size x *reward_shape)
    """
    @classmethod
    def from_list(cls, storages):      
        """
        Creates RolloutStorage from list of Storage
        input: storages - list of Storage
        output: RolloutStorage
        """
        self = cls()

        for key in storages[0].keys():
            if key != "next_states":
                name = key if key != "states" else "all_states"
                
                to_stack = [storage[key] for storage in storages]
                if key == "states":
                    to_stack.append(storages[-1]["next_states"])

                self[name] = stack(to_stack)
        
        return self        

    def transitions(self):        
        """
        Generator for all transitions in this storage.
        All additional information stored (log_probs, q-values, etc.) will not be yielded
        yields: Transition
        """
        for t in range(self.rollout_length):
            for s, a, r, ns, d in zip(self.all_states.numpy[t], self.actions.numpy[t], self.rewards.numpy[t], self.all_states.numpy[t+1], self.discounts.numpy[t]):
                yield Transition(s, a, r, ns, d)

    def crop_states(self, which):
        '''
        Returns subset of states corresponding to marker
        input: which - Which marker:
            current - all except the last
            next - all except the first
            last - only last (no timestep dimension)
            all - all states from rollout
        output: Representation
        '''
        if which is Which.current:
            return self.all_states[:-1]
        if which is Which.next:
            return self.all_states[1:]
        if which is Which.last:
            return self.all_states[-1]
        if which is Which.all:
            return self.all_states
        raise Exception("Error: 'which' marker is None?")

    # interface functions ----------------------------------------------------------------
    @property
    def rollout_length(self):
        return self.all_states.rollout_length - 1

    @property
    def batch_size(self):
        return self.all_states.batch_size

    @property
    def additional_keys(self):  
        """
        Returns all keys with additional information
        output: set of strings
        """
        standard_keys = {"all_states", "actions", "rewards", "discounts"}
        return self.keys() - standard_keys

    def __len__(self):
        return self.rollout_length * self.batch_size

    @classmethod
    def _default_name(cls):
        return "RolloutStorage"

    def __repr__(self):
        return f"Rollout of length {self.rollout_length} of size {self.batch_size}. Additional information stored: {self.additional_keys}"