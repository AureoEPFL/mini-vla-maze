import argparse, copy, json, platform, time
from pathlib import Path
import numpy as np
import torch
from .data import collect,save_dataset,load_dataset
from .model import MazeVLA,loss_fn

def batch(data,indices,device,augment=False):
    b={k:torch.as_tensor(v[indices].astype(np.int64) if k!='image' else v[indices],device=device) for k,v in data.items()}
    b['image']=b['image'].permute(0,3,1,2).float()/255
    if augment:
        b['image']=(b['image']*torch.empty(len(indices),1,1,1,device=device).uniform_(.8,1.1)+torch.randn_like(b['image'])*.01).clamp(0,1)
    return b

@torch.no_grad()
def validate(model,data,device):
    model.eval(); totals=np.zeros(4); count=0
    for start in range(0,len(data['image']),128):
        b=batch(data,np.arange(start,min(start+128,len(data['image']))),device)
        o=model(b['image'],b['tokens'],b['phase']); n=len(b['image'])
        ce=torch.nn.functional.cross_entropy(o['actions'].flatten(0,1),b['actions'].flatten()).item()
        vals=(ce,(o['actions'][:,0].argmax(-1)==b['actions'][:,0]).float().mean().item(),(o['target'].argmax(-1)==b['target']).float().mean().item(),(o['semantic'].argmax(1)==b['labels']).float().mean().item())
        totals+=np.array(vals)*n; count+=n
    return dict(zip(('action_ce','action_accuracy','target_accuracy','map_accuracy'),(totals/count).tolist()))

def run(out,episodes=320,steps=1800,seed=17,batch_size=64):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(4,torch.get_num_threads())); torch.manual_seed(seed); rng=np.random.default_rng(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    started=time.perf_counter()
    for split,n,s in [('train',episodes,10000),('val',40,100000)]:
        folder=out/split
        if not (folder/'data.npz').exists():
            a,m=collect(n,s); save_dataset(folder,a,m)
        manifest=json.loads((folder/'manifest.json').read_text())
        if len(manifest['accepted'])!=n: raise ValueError('Cached episode count differs; choose a new output directory.')
    train=load_dataset(out/'train'); val=load_dataset(out/'val')
    train_seeds={x['seed'] for x in json.loads((out/'train/manifest.json').read_text())['accepted']}
    val_seeds={x['seed'] for x in json.loads((out/'val/manifest.json').read_text())['accepted']}
    assert not train_seeds & val_seeds
    data_seconds=time.perf_counter()-started
    model=MazeVLA().to(device); opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4)
    before=validate(model,val,device); history=[]; best=float('inf'); best_state=copy.deepcopy(model.state_dict()); best_step=0
    started=time.perf_counter()
    for step in range(1,steps+1):
        model.train(); idx=rng.integers(len(train['image']),size=batch_size); b=batch(train,idx,device,True)
        opt.zero_grad(); o=model(b['image'],b['tokens'],b['phase']); loss,parts=loss_fn(o,b['labels'],b['tokens'],b['target'],b['actions'])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step()
        if step%100==0 or step==steps:
            v=validate(model,val,device)
            # Select on validation losses only. Closed-loop test results are never used.
            score=v['action_ce']+2*(1-v['target_accuracy'])+2*(1-v['map_accuracy'])
            if score<best: best=score; best_step=step; best_state=copy.deepcopy(model.state_dict())
            row={'step':step,'seconds':time.perf_counter()-started,**parts,**{'val_'+k:v for k,v in v.items()}}
            history.append(row); print(json.dumps(row),flush=True)
    model.load_state_dict(best_state)
    torch.save({k:v.cpu() for k,v in model.state_dict().items()},out/'weights.pt')
    report={'training_seed':seed,'device':str(device),'torch':torch.__version__,'python':platform.python_version(),'parameters':sum(p.numel() for p in model.parameters()),'episodes':episodes,'samples':len(train['image']),'steps':steps,'best_step':best_step,'batch_size':batch_size,'data_seconds':data_seconds,'train_seconds':time.perf_counter()-started,'validation_before':before,'validation_after':validate(model,val,device),'history':history}
    (out/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved',out,flush=True)
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='runs/maze'); p.add_argument('--episodes',type=int,default=320); p.add_argument('--steps',type=int,default=1800); p.add_argument('--seed',type=int,default=17)
    a=p.parse_args(); run(a.out,a.episodes,a.steps,a.seed)

if __name__=='__main__': main()
