from LegoRL.heads.head import Head

import torch

class CriticHead(Head):
    """
    Provides a head for critics.
    """
    def _evaluate(self, batch, of="state"):
        '''
        Calculates value function:
            for state if of="state", 
            for last state of rollouts or next state of the Batch if of="last state",
            for states except the first one of rollouts if of="next state"
        input: Batch
        input: of - str, "state", "next state" or "last state"
        output: Value
        '''
        assert of in ["state", "next state", "last state"]
        
        if of in ["next state", "last state"]:
            last_state = self(batch.last_states, storage=batch, cache_name="last state")
            if of == "last state" or batch.names()[0] != "timesteps":
                return last_state

        first_states = self(batch.states, storage=batch, cache_name="state")
        if of == "next state":
            return torch.cat(first_states[1:], last_state[None], dim=0)
        
        return first_states