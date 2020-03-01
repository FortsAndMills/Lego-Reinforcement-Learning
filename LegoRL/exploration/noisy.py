import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def NoisyLinear(std_init=0.4):
    class NoisyLinear(nn.Linear):
        """NoisyLinear module for Noisy Net exploration technique"""
        
        def __init__(self, in_features, out_features):
            super(NoisyLinear, self).__init__(in_features, out_features)
            
            sigma_init = std_init / math.sqrt(in_features)
            self.n_params = in_features*out_features + out_features
            
            self.sigma_weight = nn.Parameter(torch.FloatTensor(out_features, in_features).fill_(sigma_init))
            self.register_buffer("epsilon_input", torch.zeros(1, in_features))
            self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
            
            self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features).fill_(sigma_init))
        
        def forward(self, x):
            if self.training:
                # hack: we need to generate matrix in x out, but that's too many samples
                # instead we generate in + out noises and multiply them pairwise
                torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
                torch.randn(self.epsilon_output.size(), out=self.epsilon_output)

                scale = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
                eps_in = scale(self.epsilon_input)
                eps_out = scale(self.epsilon_output)
                
                weight = self.weight + self.sigma_weight * torch.mul(eps_in, eps_out)
                bias = self.bias + self.sigma_bias * eps_out.t()
            else:
                weight = self.weight
                bias   = self.bias
            
            return F.linear(x, weight, bias)
        
        def magnitude(self):
            # returns summed magnitudes of noise and number of noisy parameters
            return (self.sigma_weight.abs().sum() + self.sigma_bias.abs().sum()).detach().cpu().numpy(), self.n_params
    return NoisyLinear

def NoisyLinearRT(std_init=0.4):
    class NoisyLinearRT(nn.Linear):
        """
        Adds noise to output of layer to reduce number of parameters!
        Should still work as Noisy Net exploration technique
        """
        
        def __init__(self, in_features, out_features):
            super(NoisyLinearRT, self).__init__(in_features, out_features)
            
            self.n_params = out_features

            self.sigmas = nn.Parameter(torch.FloatTensor(out_features).fill_(std_init))
            self.register_buffer("epsilon", torch.zeros(out_features))
        
        def forward(self, x):
            x = super().forward(x)
            if self.training:
                torch.randn(self.epsilon.size(), out=self.epsilon)
                x = x + self.sigmas * self.epsilon
            
            return x
        
        def magnitude(self):
            # returns summed magnitudes of noise and number of noisy parameters
            return self.sigmas.abs().sum().detach().cpu().numpy(), self.n_params
    return NoisyLinearRT