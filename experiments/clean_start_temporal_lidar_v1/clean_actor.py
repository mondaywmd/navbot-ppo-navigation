"""Random-initialized Actor with exact left/right equivariance."""
import torch
from torch import nn
from observation78 import mirror_action,mirror_observation

class CleanMirrorActor(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(78,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,2),nn.Tanh())
    def forward(self,observation):
        device=next(self.parameters()).device
        x=torch.as_tensor(observation,dtype=torch.float32,device=device)
        if x.dim()==1:x=x.unsqueeze(0)
        direct=self.network(x)
        reflected=mirror_action(self.network(mirror_observation(x)))
        return 0.5*(direct+reflected)
