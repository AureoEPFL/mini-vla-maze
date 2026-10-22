"""Larger, held-out arenas. Reuses the trained visual heads, never resizes the map.

The fixed-width neural action head is deliberately NOT used on these arenas.
An explicit A* controller uses the predicted entities and known dynamics.
"""
import argparse,copy,csv,heapq,json,time
from collections import deque
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from .env import RGB,COLORS,SHAPES,MOVES,CELL
from .evaluate import load_model,wilson

ARENAS=('switchback','catacombs','airlocks')
TITLES={'switchback':'Switchback / passages en zigzag','catacombs':'Catacombs / réseau de salles','airlocks':'Airlocks / portes successives'}

@dataclass
class Arena:
    n:int
    walls:set
    objects:list
    identities:list
    pusher:int
    goal:int
    target:int
    kind:str
    seed:int
    doors:list=field(default_factory=list)
    phase:int=0
    steps:int=0
    pushes:int=0
    waits:int=0
    collisions:int=0
    wrong_contacts:int=0

    def clone(self):return copy.deepcopy(self)
    @property
    def tokens(self):return np.array(divmod(self.identities[self.target],2),dtype=np.int64)
    @property
    def text(self):
        c,s=self.tokens;return f'push the {COLORS[c]} {SHAPES[s]} to the exit'
    @property
    def state(self):return self.pusher,self.objects[self.target],self.phase
    def active(self,phase):return door_cells(self.doors,phase)
    def success(self):return self.objects[self.target]==self.goal and self.wrong_contacts==0
    def labels(self):
        z=np.zeros((self.n,self.n),np.int64)
        for p in self.walls:z.flat[p]=1
        z.flat[self.goal]=2
        for p in self.active(self.phase):z.flat[p]=12
        for p,i in zip(self.objects,self.identities):z.flat[p]=4+i
        z.flat[self.pusher]=3
        return z
    def render(self):return render_labels(self.labels())
    def step(self,a):
        fixed=set(self.objects)-{self.objects[self.target]}
        nxt=step_state(self.n,self.walls,self.doors,fixed,self.state,a)
        if a==4:self.waits+=1
        if nxt is None:
            dr,dc=MOVES[a];r,c=divmod(self.pusher,self.n)
            if (r+dr)*self.n+c+dc in fixed:self.wrong_contacts+=1
            self.collisions+=1
            self.phase=next_phase(self.doors,self.phase,set(self.objects)|{self.pusher})
        else:
            self.pushes+=int(nxt[1]!=self.objects[self.target]);self.pusher,self.objects[self.target],self.phase=nxt
        self.steps+=1

def render_labels(labels):
    """Training tile vocabulary; raster order clips hazard strokes at tile edges."""
    n=labels.shape[0];im=Image.new('RGB',(n*CELL,n*CELL),(19,27,44));d=ImageDraw.Draw(im)
    for r in range(n):
        for c in range(n):
            z=int(labels[r,c]);x,y=c*CELL,r*CELL
            d.rectangle((x,y,x+7,y+7),outline=(25,35,53))
            if z==1:
                d.rectangle((x,y,x+7,y+7),fill=(66,81,103));d.line((x,y,x+7,y),fill=(91,109,134))
            elif z==2:
                d.rectangle((x,y,x+7,y+7),fill=(130,95,191));d.rectangle((x+2,y+2,x+5,y+5),outline=(230,209,255))
            elif z==3:
                d.ellipse((x+1,y+1,x+6,y+6),fill=(239,245,255));d.point((x+3,y+3),fill=(46,61,81))
            elif z==12:
                d.rectangle((x,y,x+7,y+7),fill=(224,107,46));d.line((x,y+7,x+7,y),fill=(255,209,128),width=2)
            elif 4<=z<12:
                i=z-4;box=(x+1,y+1,x+6,y+6)
                if i%2:d.ellipse(box,fill=RGB[i//2])
                else:d.rectangle(box,fill=RGB[i//2])
    return np.array(im)

def door_cells(doors,phase):
    return {p for cells,offset in doors if (phase+offset)%8<4 for p in cells}

def next_phase(doors,phase,occupied):
    if not doors:return 0
    nxt=(phase+1)%8
    return phase if door_cells(doors,nxt)&occupied else nxt

def step_state(n,walls,doors,fixed,state,a):
    p,b,phase=state;dr,dc=MOVES[a];r,c=divmod(p,n);nr,nc=r+dr,c+dc
    if not(0<=nr<n and 0<=nc<n):return None
    pp=nr*n+nc;bb=b;blocked=walls|fixed|door_cells(doors,phase)
    if a!=4:
        if pp in blocked:return None
        if pp==b:
            br,bc=divmod(b,n);br+=dr;bc+=dc
            if not(0<=br<n and 0<=bc<n):return None
            bb=br*n+bc
            if bb in blocked:return None
    return pp,bb,next_phase(doors,phase,fixed|{pp,bb})

def search(n,walls,doors,fixed,start,goal,max_nodes=350000):
    """A* with a relaxed object-distance heuristic. Unit cost per simulator action."""
    dist={goal:0};q=deque([goal]);blocked=walls|fixed
    while q:
        p=q.popleft();r,c=divmod(p,n)
        for dr,dc in MOVES[:4]:
            nr,nc=r+dr,c+dc;pp=nr*n+nc
            if 0<=nr<n and 0<=nc<n and pp not in blocked and pp not in dist:dist[pp]=dist[p]+1;q.append(pp)
    if start[1] not in dist:return None,0
    def h(s):
        pr,pc=divmod(s[0],n);br,bc=divmod(s[1],n)
        return dist.get(s[1],10**7)+max(0,abs(pr-br)+abs(pc-bc)-1)
    queue=[(h(start),0,start)];cost={start:0};parent={}
    while queue and len(cost)<max_nodes:
        _,g,s=heapq.heappop(queue)
        if g!=cost[s]:continue
        if s[1]==goal:
            path=[]
            while s!=start:s,a=parent[s];path.append(a)
            return path[::-1],len(cost)
        for a in range(5 if doors else 4):
            nxt=step_state(n,walls,doors,fixed,s,a)
            if nxt is None or nxt==s or nxt[1] not in dist:continue
            ng=g+1
            if ng<cost.get(nxt,10**9):
                cost[nxt]=ng;parent[nxt]=(s,a);heapq.heappush(queue,(ng+h(nxt),ng,nxt))
    return None,len(cost)

def make_arena(seed,kind='switchback'):
    if kind not in ARENAS:raise ValueError(kind)
    n=25;rng=np.random.default_rng(seed);walls={r*n+c for r in range(n) for c in range(n) if r in (0,n-1) or c in (0,n-1)};doors=[]
    if kind in ('switchback','airlocks'):
        for j,r in enumerate((6,12,18)):
            right=j%2==0
            wallcols=range(1,20) if right else range(5,24)
            walls|={r*n+c for c in wallcols}
            if kind=='airlocks':doors.append(([r*n+c for c in (range(20,24) if right else range(1,5))],j*2))
        positions=[22*n+int(c) for c in rng.choice(np.arange(4,20),4,replace=False)]
        pusher=23*n+int(rng.integers(3,21));goal=2*n+int(rng.integers(5,19))
    else:
        # Random spanning tree over nine 5x5 rooms; 3-cell-wide connecting passages.
        floor=set();centres={(r,c):(4+8*r,4+8*c) for r in range(3) for c in range(3)}
        for rr,cc in centres.values():floor|={r*n+c for r in range(rr-2,rr+3) for c in range(cc-2,cc+3)}
        seen={(2,0)};stack=[(2,0)]
        while stack:
            here=stack[-1];options=[(here[0]+dr,here[1]+dc) for dr,dc in MOVES[:4] if (here[0]+dr,here[1]+dc) in centres and (here[0]+dr,here[1]+dc) not in seen]
            if not options:stack.pop();continue
            there=options[int(rng.integers(len(options)))];seen.add(there);stack.append(there)
            ar,ac=centres[here];br,bc=centres[there]
            if ar==br:floor|={r*n+c for r in range(ar-1,ar+2) for c in range(min(ac,bc),max(ac,bc)+1)}
            else:floor|={r*n+c for r in range(min(ar,br),max(ar,br)+1) for c in range(ac-1,ac+2)}
        walls=set(range(n*n))-floor
        positions=[20*n+4,2*n+2,10*n+18,18*n+22];pusher=21*n+4;goal=4*n+20
    ids=list(map(int,rng.choice(8,4,replace=False)));target=0
    if kind!='catacombs':target=int(rng.integers(4))
    return Arena(n,walls,positions,ids,pusher,goal,target,kind,seed,doors,phase=int(rng.integers(8)) if doors else 0)

@torch.no_grad()
def perceive(model,image,tokens):
    x=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1)[None].float()/255
    f=model.patch(x);t=torch.as_tensor(tokens)[None].long()
    c=(model.ckey(f)*model.color(t[:,0])[:,:,None,None]).sum(1)/4
    s=(model.skey(f)*model.shape(t[:,1])[:,:,None,None]).sum(1)/4
    attention=(c+s).flatten(1).softmax(-1)[0].numpy().reshape(f.shape[-2:])
    return model.semantic(f).argmax(1)[0].numpy(),attention

def rollout(model,world,mode='hybrid',record=False,max_steps=450,condition='clean'):
    w=world.clone();frames=[];route=[];expected=None;goal=None;calls=0;searches=0;nodes=0;start=time.perf_counter();plan_seconds=0.;reason=None
    attention=None;pred=None
    while not w.success() and not w.wrong_contacts and w.steps<max_steps:
        image=w.render()
        if mode=='oracle':walls=w.walls;fixed=set(w.objects)-{w.objects[w.target]};state=w.state;goal=w.goal
        else:
            obs=(image*.4).astype(np.uint8) if condition=='dark' else image
            pred,attention=perceive(model,obs,w.tokens);calls+=1;z=pred.flatten();agents=np.flatnonzero(z==3);goals=np.flatnonzero(z==2);objects=set(np.flatnonzero((z>=4)&(z<12)).tolist());target=int(attention.argmax())
            if len(goals)==1:goal=int(goals[0])
            if len(agents)!=1 or target not in objects or goal is None:reason='invalid_perception';break
            walls=set(np.flatnonzero(z==1).tolist());fixed=objects-{target};state=(int(agents[0]),target,w.phase)
        if expected!=state or not route:
            t=time.perf_counter();route,expanded=search(w.n,walls,w.doors,fixed,state,goal);plan_seconds+=time.perf_counter()-t;nodes+=expanded;searches+=1
            if route is None:reason='no_plan';break
        if not route:reason='premature_goal';break
        a=route.pop(0)
        if record:frames.append({'rgb':image,'labels':w.labels(),'step':w.steps,'phase':w.phase,'pusher':w.pusher,'target':w.objects[w.target],'attention':attention,'action':a,'pushes':w.pushes,'remaining':len(route)+1})
        expected=step_state(w.n,walls,w.doors,fixed,state,a);w.step(a)
    if record:frames.append({'rgb':w.render(),'labels':w.labels(),'step':w.steps,'phase':w.phase,'pusher':w.pusher,'target':w.objects[w.target],'attention':attention,'action':4,'pushes':w.pushes,'remaining':0})
    row={'seed':w.seed,'arena':w.kind,'size':w.n,'mode':mode,'condition':condition,'instruction':w.text,'success':w.success(),'outcome':'success' if w.success() else (reason or ('distractor_contact' if w.wrong_contacts else 'timeout')),'steps':w.steps,'pushes':w.pushes,'waits':w.waits,'collisions':w.collisions,'model_calls':calls,'searches':searches,'search_states':nodes,'planning_seconds':plan_seconds,'seconds':time.perf_counter()-start}
    return row,frames

def run(checkpoint,out,n=6):
    torch.set_num_threads(min(4,torch.get_num_threads()));model=load_model(checkpoint);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];manifest={}
    for j,kind in enumerate(ARENAS):
        accepted=[];rejected=[];seed=800000+j*10000
        while len(accepted)<n and seed<800000+j*10000+200:
            w=make_arena(seed,kind);seed+=1;path,count=search(w.n,w.walls,w.doors,set(w.objects)-{w.objects[w.target]},w.state,w.goal)
            if path is None or not 40<=len(path)<=400:rejected.append({'seed':w.seed,'search_states':count});continue
            accepted.append(w.seed)
            for mode in ('oracle','hybrid'):
                row,_=rollout(model,w,mode);rows.append(row);print(json.dumps(row),flush=True)
        manifest[kind]={'accepted':accepted,'rejected':rejected,'requested':n}
    with (out/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary={}
    for kind in ARENAS:
        for mode in ('oracle','hybrid'):
            group=[r for r in rows if r['arena']==kind and r['mode']==mode];count=len(group);k=sum(r['success'] for r in group)
            summary[f'{kind}/{mode}']={'successes':k,'episodes':count,'wilson_95':wilson(k,count) if count else None,'mean_steps':float(np.mean([r['steps'] for r in group])) if count else None,'mean_pushes':float(np.mean([r['pushes'] for r in group])) if count else None}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='results/maze/weights.pt');p.add_argument('--out',default='runs/large');p.add_argument('--episodes',type=int,default=6);a=p.parse_args();run(a.checkpoint,a.out,a.episodes)
