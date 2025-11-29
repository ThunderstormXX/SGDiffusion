import torch
from torch.optim.optimizer import Optimizer
import math

class MUON(Optimizer):
    """
    Implementation of MUON (Momentum with Uniform Noise) optimizer.
    
    This optimizer adds uniform noise to gradients during the optimization process,
    which can help with exploration and escaping local minima.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        noise_scale (float, optional): scale of the noise to add (default: 0.01)
        noise_type (str, optional): type of noise to add ('gaussian' or 'uniform', default: 'gaussian')
    """
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, noise_scale=0.01, noise_type='gaussian'):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if noise_scale < 0.0:
            raise ValueError(f"Invalid noise_scale value: {noise_scale}")
        if noise_type not in ['gaussian', 'uniform']:
            raise ValueError(f"Invalid noise_type: {noise_type}, must be 'gaussian' or 'uniform'")
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, noise_scale=noise_scale, 
                        noise_type=noise_type)
        super(MUON, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            noise_scale = group['noise_scale']
            noise_type = group['noise_type']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = buf
                
                # Add noise
                if noise_scale > 0:
                    if noise_type == 'gaussian':
                        noise = torch.randn_like(d_p) * noise_scale
                    else:  # uniform
                        noise = (torch.rand_like(d_p) * 2 - 1) * noise_scale
                    d_p = d_p + noise
                
                # Update parameters
                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class SignSGD(Optimizer):
    """
    Implementation of SignSGD optimizer.
    
    SignSGD uses only the sign of the gradients to update the parameters,
    which can be more robust to outliers and lead to better generalization.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=0.01, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SignSGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                
                # Take the sign of gradients
                d_p = d_p.sign()
                
                # Update parameters
                p.data.add_(d_p, alpha=-group['lr'])

        return loss


