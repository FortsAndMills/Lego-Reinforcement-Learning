from .RLmodule import *

class DQN_loss(RLmodule):
    """
    Classic deep Q-learning algorithm (DQN).
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        sampler - RLmodule with "sample" property
        q_head - RLmodule with "q" method
        critic - RLmodule with "v" method

    Provides: loss
    """
    def __init__(self, sampler, q_head, critic=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = Reference(sampler)
        self.critic = Reference(critic or q_head)
        self.q_head = Reference(q_head)

    def estimate_next_state(self, batch):
        '''
        Calculates estimation of V* for next state.
        input: Batch
        output: FloatTensor, (*batch_shape, *value_shape)
        '''
        return self.critic.v(batch, for_next_state=1)      
    
    def comparative_loss(self, q, target):
        '''
        Calculates batch loss
        input: q - current model output, FloatTensor, (*batch_shape)
        input: target - FloatTensor, (*batch_shape)
        output: FloatTensor, (*batch_shape)
        '''
        return (target - q).pow(2)

    def loss(self):
        '''
        Loss calculation based on TD-error from DQN algorithm.
        output: Tensor, scalar
        '''
        batch = self.sampler.sample
        if batch is None: return None
        
        # TODO: now what?
        # self.q_net.train()

        q = self.q_head.q(batch).gather(batch.action)
        with torch.no_grad():
            next_v = self.estimate_next_state(batch)
            target = self.q_head.one_step_q(batch, next_v)
            
        loss_b = self.comparative_loss(q, target)        
        loss = batch.average(loss_b)

        self.log(self.name + " loss", loss.detach().cpu().numpy(), "training iteration", "loss")

        return loss

    def __repr__(self):
        return f"Calculates DQN loss for {self.q_head.name} using {self.critic.name} as target calculator and data from {self.sampler.name}"