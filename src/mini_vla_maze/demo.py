import argparse
from pathlib import Path
import torch
from .env import make_world,KINDS
from .planner import expert
from .evaluate import load_model,rollout
from .visualize import animation

def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(); p.add_argument('--kind',choices=KINDS,default='tower'); p.add_argument('--controller',choices=('direct','hybrid','oracle'),default='hybrid'); p.add_argument('--seed',type=int,default=250000); p.add_argument('--checkpoint',default=str(root/'results/maze/weights.pt')); p.add_argument('--out',default='runs/demo'); p.add_argument('--execute',type=int,choices=(1,4),default=1)
    a=p.parse_args(); torch.set_num_threads(min(4,torch.get_num_threads()))
    model=load_model(a.checkpoint)
    for seed in range(a.seed,a.seed+1000):
        world=make_world(seed,a.kind)
        if expert(world) is not None:break
    else:raise RuntimeError('No solvable candidate in seed range')
    row,frames=rollout(model,world,a.controller,a.execute,record=True)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); path=out/f'{a.kind}_{a.controller}_{seed}.gif'
    animation(frames,row,path); print(row); print('Saved:',path.resolve())

if __name__=='__main__':main()
