from .RLmodule import *

class DQN_loss(RLmodule):
    """
    Classic deep Q-learning algorithm (DQN).
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        sampler - RLmodule with "sample" property
        q_head - RLmodule with "q" method
        critic - RLmodule with "v" method

    Provides: loss, batch_loss
    """
    def __init__(self, sampler, q_head, critic=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = Reference(sampler)
        self.critic = Reference(critic or q_head)
        self.q_head = Reference(q_head)

        self._loss = None

    @property
    def loss(self):
        return self._loss

    def wait(self):
        self._loss = None

    def estimate_next_state(self, batch):
        '''
        Calculates estimation of V* for next state.
        input: Batch
        output: FloatTensor, (*batch_shape, *value_shape)
        '''
        return self.critic.v(batch, for_next_state=1)      
    
    def batch_loss(self, batch):
        '''
        Calculates loss for batch based on TD-error from DQN algorithm.
        input: batch - Batch
        output: Tensor, (*batch_shape)
        '''
        # checking if this loss is cached
        if self.name in batch.losses:
            self.debug("loss for the batch has already been calculated, used from cache")
            return batch.losses[self.name]

        # TODO: now what?
        # self.q_net.train()

        q = self.q_head.q(batch)
        batch_q = q.gather(batch.action)
        with torch.no_grad():
            next_v = self.estimate_next_state(batch)
            target = type(q).one_step_q(batch, next_v)

        batch.losses[self.name] = type(q).compare(batch_q, target)            
        return batch.losses[self.name] 

    def iteration(self):
        '''
        Calculates loss for batch from replay
        '''
        batch = self.sampler.sample
        if batch is None:
            self._loss = None
            self.debug("no batch is found, loss is None.")
            return

        self.debug("starts loss calculation.", open=True)
        
        # caching loss
        self.batch_loss(batch)        
        self._loss = batch.average(self.name)

        self.debug("finished loss calculation.", close=True)

        self.log(self.name, self._loss.detach().cpu().numpy(), "training iteration", "loss")

    def __repr__(self):
        return f"Calculates DQN loss for {self.q_head.name} using {self.critic.name} as target calculator and data from {self.sampler.name}"