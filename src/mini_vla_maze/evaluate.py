import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from .env import N,make_world,transition,rc,cell,hazards
from .model import MazeVLA
from .planner import plan,expert

class Policy:
    """Only RGB, grammar tokens, obstacle type and phase cross this boundary.

    Hybrid dynamics/schedule are known. Walls, objects, agent and goal are
    decoded from pixels. No ground-truth coordinates are supplied here.
    """
    def __init__(self,model,mode='direct',execute=1):
        self.model=model; self.mode=mode; self.execute=execute; self.goal=None
        self.calls=0; self.planner_calls=0; self.queue=[]; self.expected=None
        self.attention=None; self.map=None; self.last_path=[]

    @torch.no_grad()
    def act(self,image,tokens,phase,kind):
        if self.mode=='direct' and self.queue: return self.queue.pop(0)
        self.calls+=1
        x=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1)[None].float()/255
        o=self.model(x,torch.as_tensor(tokens)[None].long(),torch.tensor([phase]))
        self.attention=o['target'][0].softmax(0).numpy().reshape(N,N)
        self.map=o['semantic'][0].argmax(0).numpy()
        if self.mode=='direct':
            self.queue=o['actions'][0].argmax(-1).tolist()[:self.execute]
            return self.queue.pop(0)
        z=self.map.flatten(); walls=set(np.flatnonzero(z==1).tolist())
        objects=set(np.flatnonzero((z>=4)&(z<12)).tolist())
        pushers=np.flatnonzero(z==3); goals=np.flatnonzero(z==2)
        if len(goals)==1: self.goal=int(goals[0])
        target=int(np.argmax(self.attention))
        if len(pushers)!=1 or target not in objects or self.goal is None:
            self.queue=[]; self.expected=None; return 4
        state=(int(pushers[0]),target,int(phase)); fixed=objects-{target}
        if self.expected!=state or not self.queue:
            self.planner_calls+=1
            self.queue=plan(walls,kind,fixed,state,self.goal) or []
        self.last_path=self.queue.copy()
        action=self.queue.pop(0) if self.queue else 4
        self.expected=transition(walls,kind,fixed,state,action)
        return action

def load_model(path):
    model=MazeVLA(); model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)); model.eval(); return model

def rollout(model,world,mode='direct',execute=1,condition='clean',record=False,max_steps=100):
    w=world.clone(); policy=Policy(model,mode,execute); frames=[]; path=[]; original_text=w.text
    tokens=w.tokens.copy(); rng=np.random.default_rng(w.seed+9000000)
    if condition=='wrong_shape': tokens[1]=1-tokens[1]
    if condition=='wrong_color': tokens[0]=(tokens[0]+1)%4
    started=time.perf_counter(); visited=set(); repeats=0
    while w.steps<max_steps and not w.success() and not w.wrong_pushes:
        if condition=='displacement' and w.steps==6:
            r,c=rc(w.pusher)
            for dr,dc in ((0,1),(1,0),(0,-1),(-1,0)):
                p=cell(r+dr,c+dc)
                if p not in w.walls|set(w.objects)|hazards(w.kind,w.phase): w.pusher=p; path=[]; break
        rgb=w.render(); obs=rgb.copy()
        if condition=='blank': obs[:]=0
        elif condition=='dark': obs=(obs*.4).astype(np.uint8)
        elif condition=='noise': obs=np.clip(obs.astype(float)+rng.normal(0,25,obs.shape),0,255).astype(np.uint8)
        if mode=='oracle':
            if not path: path=expert(w) or []
            a=path.pop(0) if path else 4
        else: a=policy.act(obs,tokens,w.phase,w.kind)
        if record: frames.append({'rgb':rgb,'labels':w.labels(),'observation':obs,'attention':None if policy.attention is None else policy.attention.copy(),'phase':w.phase,'step':w.steps,'action':a,'pushes':w.pushes,'collisions':w.collisions,'calls':policy.calls,'pusher':w.pusher,'target':w.objects[w.target]})
        key=(w.state,tuple(w.objects)); repeats+=int(key in visited); visited.add(key)
        w.step(a)
    outcome='success' if w.success() else ('distractor_contact' if w.wrong_pushes else 'timeout')
    row={'seed':w.seed,'kind':w.kind,'mode':mode,'execute':execute,'condition':condition,'instruction':original_text,'success':w.success(),'outcome':outcome,'steps':w.steps,'pushes':w.pushes,'waits':w.waits,'collisions':w.collisions,'repeated_states':repeats,'model_calls':policy.calls,'planner_calls':policy.planner_calls,'seconds':time.perf_counter()-started}
    if record:
        frames.append({'rgb':w.render(),'labels':w.labels(),'observation':w.render(),'attention':policy.attention,'phase':w.phase,'step':w.steps,'action':4,'pushes':w.pushes,'collisions':w.collisions,'calls':policy.calls,'pusher':w.pusher,'target':w.objects[w.target]})
    return row,frames

def wilson(k,n):
    z=1.959964; p=k/n; den=1+z*z/n; mid=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]

def test_worlds(n,kind,start,heldout=False):
    accepted=[]; rejected=[]; seed=start
    while len(accepted)<n:
        w=make_world(seed,kind,heldout); seed+=1; p=expert(w)
        if p is None or not 4<=len(p)<=90: rejected.append(w.seed); continue
        accepted.append(w)
    return accepted,rejected

def run(checkpoint,out,n=16):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(4,torch.get_num_threads())); model=load_model(checkpoint)
    suites=[('open','open',False),('baffles','baffles',False),('gate','gate',False),('sweeper','sweeper',False),('unseen_rotor','rotor',False),('unseen_tower','tower',False),('heldout_blue_disk','baffles',True)]
    allrows=[]; manifest={}; worlds_by_suite={}
    for i,(name,kind,heldout) in enumerate(suites):
        worlds,rejected=test_worlds(n,kind,200000+i*10000,heldout); worlds_by_suite[name]=worlds
        manifest[name]={'accepted_seeds':[w.seed for w in worlds],'rejected_seeds':rejected,'kind':kind,'heldout':heldout}
        for mode,k in [('oracle',1),('direct',1),('direct',4),('hybrid',1)]:
            rows=[]
            for w in worlds:
                row,_=rollout(model,w,mode,k); row['suite']=name; rows.append(row)
            print(name,mode,k,sum(r['success'] for r in rows),'/',n,flush=True); allrows+=rows
    for cond in ('wrong_shape','wrong_color','blank','dark','noise','displacement'):
        for mode in ('direct','hybrid'):
            rows=[]
            for w in worlds_by_suite['baffles']:
                row,_=rollout(model,w,mode,1,cond); row['suite']=cond; rows.append(row)
            print(cond,mode,sum(r['success'] for r in rows),'/',n,flush=True); allrows+=rows
    # The oracle ceiling for displacement separates recovery difficulty from perception.
    for w in worlds_by_suite['baffles']:
        row,_=rollout(model,w,'oracle',1,'displacement'); row['suite']='displacement'; allrows.append(row)
    with (out/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0])); writer.writeheader(); writer.writerows(allrows)
    summary={}
    for row in allrows:
        key=f"{row['suite']}/{row['mode']}/k{row['execute']}"
        summary.setdefault(key,[]).append(row)
    summary={k:{'successes':sum(x['success'] for x in rows),'episodes':len(rows),'rate':np.mean([x['success'] for x in rows]).item(),'wilson_95':wilson(sum(x['success'] for x in rows),len(rows)),**{f'mean_{m}':float(np.mean([x[m] for x in rows])) for m in ('steps','collisions','model_calls','planner_calls','pushes','waits')}} for k,rows in summary.items()}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'test_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='runs/maze/weights.pt'); p.add_argument('--out',default='runs/maze/eval'); p.add_argument('--episodes',type=int,default=16)
    a=p.parse_args(); run(a.checkpoint,a.out,a.episodes)

if __name__=='__main__': main()
