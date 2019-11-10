from .backbone import *
    
class Q():
    '''
    FloatTensor representing Q-function, (*batch_shape x num_actions)
    '''
    def __init__(self, tensor):
        self.tensor = tensor

    def greedy(self):
        '''
        Returns greedy action based on the output of net
        output: LongTensor, (*batch_shape)
        '''
        return self.tensor.max(-1)[1]
        
    def gather(self, action_b):
        '''
        Returns output of net for given batch of actions
        input: action_b - LongTensor, (*batch_shape)
        output: FloatTensor, (*batch_shape)
        '''
        return self.tensor.gather(-1, action_b.unsqueeze(-1)).squeeze(-1)
    
    def value(self):
        '''
        Returns value of action, chosen greedy
        output: FloatTensor, (*batch_shape)
        '''
        return self.tensor.max(-1)[0]

class toQ(Hat):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def required_shape(self):
        return self.num_actions

    def forward(self, x):
        assert x.shape[-1] == self.num_actions
        return Q(x)

    def extra_repr(self):    
        return 'output interpreted as Q-function for {} actions'.format(self.num_actions)
    
class Q_head(RLmodule):
    """
    Provides a head for Q-function.
    
    Args:
        backbone - RLmodule for backbone with "mount_head"
        HeadNetwork - nn.Module class for head
        hat - Hat class to transform final output layer to desired shape

    Provides: act
    """
    def __init__(self, backbone, headNetwork=nn.Linear, hat=toQ, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = Reference(backbone)
        self.headNetwork = headNetwork
        self.hat = hat

    def initialize(self):
        '''Initializes net'''
        self.net = self.backbone.mount_head(self.name, self.system.observation_shape, 
                                         self.headNetwork, self.hat(self.system.num_actions))
        
    def act(self, state):        
        '''
        Default behavior is random.
        Input: state - np.array, (batch_size, *observation_shape)
        Output: action - list, (batch_size)
        '''        
        # TODO: now what?
        # if self.system.is_learning:
        #    self.net.train()
        # else:
        #    self.net.eval()
        self.debug("received act query.")
        
        with torch.no_grad():
            q = self.net(Tensor(state))
            
            # TODO: now what?
            #if self.is_recording:
            #    self.record["q"].append(q[0:1].cpu().numpy())
            
            return q.greedy().cpu().numpy()

    def v(self, batch, for_next_state=False):
        '''
        Calculates V* for state if for_next_state is false, otherwise calculates for next state.
        input: Batch
        output: FloatTensor, (*batch_shape, *value_shape)
        '''
        return self.q(batch, for_next_state).value()

    def q(self, batch, for_next_state=False):
        '''
        Calculates Q* for state if for_next_state is false, otherwise calculates for next state.
        input: Batch
        output: FloatTensor, (*batch_shape, num_actions, *value_shape)
        '''
        if for_next_state:
            return self.net(batch.next_state, storage=batch.next_state_storage)
        return self.net(batch.state, storage=batch.state_storage)

    def one_step_q(self, batch, next_v):
        '''
        Calculates one-step approximation using next_v as V*(s') estimation
        input: Batch
        input: next_v, (*batch_shape, *value_shape)
        output: FloatTensor, (*batch_shape, *value_shape)
        '''
        # TODO: it does not work with these shapes!
        return batch.reward + (self.system.gamma**batch.n_steps) * next_v * (1 - batch.done)
    
    # WHAT TO DO NOW?
    # def get_priorities(self, batch):
    #     '''
    #     Calculates importance of transitions in batch by loss
    #     input: batch - FloatTensor, (batch_size)
    #     output: FloatTensor, (batch_size)
    #     '''
    #     # TODO: wtf?
    #     return batch.dqn_loss**0.5

    # TODO: what to do now?   
    #def show_record(self):
    #    show_frames_and_distribution(self.record["frames"], np.array(self.record["qualities"]), "Qualities", np.arange(self.config["num_actions"]))

    def __repr__(self):
        return f"Head, representing Q-function, connected to {self.backbone.name}"