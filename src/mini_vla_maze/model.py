import math
import torch
from torch import nn
from torch.nn import functional as F
from .env import N, CELL

HORIZON=4

class MazeVLA(nn.Module):
    """Shared pixel encoder, factorised grounding, semantic map and BC action head.

    The policy uses RGB, two grammar tokens and a measured phase. It receives
    neither true object coordinates nor an oracle occupancy map.
    """
    def __init__(self):
        super().__init__()
        self.patch=nn.Sequential(nn.Conv2d(3,32,CELL,stride=CELL),nn.ReLU(),nn.Conv2d(32,32,1),nn.ReLU())
        self.color=nn.Embedding(4,16); self.shape=nn.Embedding(2,16)
        self.ckey=nn.Conv2d(32,16,1); self.skey=nn.Conv2d(32,16,1)
        self.semantic=nn.Conv2d(32,13,1)
        self.spatial=nn.Sequential(nn.Conv2d(33,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,16,3,padding=1),nn.ReLU())
        self.actions=nn.Sequential(nn.Linear(16*N*N+34,192),nn.ReLU(),nn.Linear(192,HORIZON*5))

    def forward(self,image,tokens,phase):
        f=self.patch(image)
        cq=self.color(tokens[:,0]); sq=self.shape(tokens[:,1])
        c=(self.ckey(f)*cq[:,:,None,None]).sum(1)/4
        s=(self.skey(f)*sq[:,:,None,None]).sum(1)/4
        joint=c+s; attention=joint.flatten(1).softmax(-1).reshape(-1,1,N,N)
        spatial=self.spatial(torch.cat((f,attention),1)).flatten(1)
        angles=phase.float()*2*math.pi/6
        clock=torch.stack((angles.sin(),angles.cos()),1)
        actions=self.actions(torch.cat((spatial,cq,sq,clock),1)).reshape(-1,HORIZON,5)
        return {'actions':actions,'semantic':self.semantic(f),'color':c.flatten(1),'shape':s.flatten(1),'target':joint.flatten(1)}

def loss_fn(out,labels,tokens,target,action):
    sem=F.cross_entropy(out['semantic'],labels)
    def distribution(mask):
        mask=mask.flatten(1).float()
        return mask/mask.sum(1,keepdim=True).clamp_min(1)
    obj=labels-4
    color=distribution((obj>=0)&(obj<8)&(obj//2==tokens[:,0,None,None]))
    shape=distribution((obj>=0)&(obj<8)&(obj%2==tokens[:,1,None,None]))
    ground=F.cross_entropy(out['target'],target)
    attributes=-(color*F.log_softmax(out['color'],1)).sum(1).mean()-(shape*F.log_softmax(out['shape'],1)).sum(1).mean()
    bc=F.cross_entropy(out['actions'].flatten(0,1),action.flatten())
    total=bc+sem+ground+.4*attributes
    return total,{'action_ce':bc.item(),'map_ce':sem.item(),'ground_ce':ground.item()}
