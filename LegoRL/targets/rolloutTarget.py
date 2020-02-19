from LegoRL.core.RLmodule import RLmodule

class RolloutTarget(RLmodule):
    """
    Interface for targets that can be computed only for rollouts!

    Provides: returns, advantage
    """
    def _rollout_returns(self, rollout):
        '''
        Calculates target with guarantees that input is Rollout.
        input: RolloutStorage
        output: V
        '''
        raise NotImplementedError()

    def returns(self, storage):
        '''
        Calculates target by checking that storage is rollout.
        input: Storage
        output: V
        '''
        rollout = storage.full_rollout
        returns = self._rollout_returns(rollout)
        return storage.from_full_rollout(returns)

    def _rollout_advantage(self, rollout):
        '''
        Calculates advantage with guarantees that input is Rollout.
        input: RolloutStorage
        output: V
        '''
        raise NotImplementedError()

    def advantage(self, storage):
        '''
        Calculates advantage by checking that storage is rollout.
        input: Storage
        output: V
        '''
        rollout = storage.full_rollout
        A = self._rollout_advantage(rollout)
        return storage.from_full_rollout(A)