import torch
import torch.nn as nn
from radgraph import F1RadGraph

class myRadGraph(nn.Module):
    def __init__(self, reward_level="partial"):
        
        super().__init__()
         
        self.myradgraph_scorer = F1RadGraph(reward_level=reward_level)
    
    def forward(self, hyps, refs):
        
        return self.myradgraph_scorer(refs, hyps)[0]