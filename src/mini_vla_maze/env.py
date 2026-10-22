"""Discrete quasi-static pushing. This is not a continuous rigid-body engine."""
from dataclasses import dataclass
import copy
import numpy as np
from PIL import Image, ImageDraw

N, CELL, PERIOD = 11, 8, 6
COLORS = ('red', 'green', 'blue', 'amber')
SHAPES = ('square', 'disk')
RGB = ((236,83,99), (57,204,150), (73,143,247), (251,190,70))
MOVES = ((-1,0), (0,1), (1,0), (0,-1), (0,0))
ACTION_NAMES = ('up','right','down','left','wait')
KINDS = ('open','baffles','gate','sweeper','rotor','tower')

def parse_instruction(text):
    words=text.lower().strip().split()
    if len(words)!=7 or words[:2]!=['push','the'] or words[4:]!=['to','the','exit']:
        raise ValueError('Use: push the red/green/blue/amber square/disk to the exit')
    if words[2] not in COLORS or words[3] not in SHAPES:
        raise ValueError('Unknown colour or shape')
    return np.array([COLORS.index(words[2]),SHAPES.index(words[3])],dtype=np.int64)

def cell(r,c): return r*N+c
def rc(p): return divmod(int(p),N)

def hazards(kind, phase):
    if kind == 'gate': return {cell(5,5)} if phase < 3 else set()
    if kind == 'sweeper': return {cell(5,(3,4,5,6,5,4)[phase])}
    if kind == 'rotor':
        offsets = ((-1,0),(1,0),(0,-1),(0,1)) if phase < 3 else ((-1,-1),(-1,1),(1,-1),(1,1))
        return {cell(5+dr,5+dc) for dr,dc in offsets}
    return set()

def advance_phase(kind, phase, occupied):
    """Safety interlock: the obstacle waits when its next pose is occupied."""
    if kind in ('open','baffles','tower'): return 0
    nxt=(phase+1)%PERIOD
    return phase if hazards(kind,nxt) & occupied else nxt

def transition(walls, kind, fixed, state, action):
    """Shared exact dynamics for planner and simulator, fixed = other objects.

    A planner is forbidden to move distractors. The environment handles such
    contacts separately, and records them as task failures.
    """
    p,b,phase=state
    dr,dc=MOVES[int(action)]; r,c=rc(p); nr,nc=r+dr,c+dc
    if not (0<=nr<N and 0<=nc<N): return None
    np_=cell(nr,nc); nb=b
    blocked=walls | fixed | hazards(kind,phase)
    if action != 4:
        if np_ in blocked: return None
        if np_ == b:
            br,bc=rc(b); br+=dr; bc+=dc
            if not (0<=br<N and 0<=bc<N): return None
            nb=cell(br,bc)
            if nb in blocked: return None
    phase=advance_phase(kind,phase,set(fixed)|{np_,nb})
    return np_,nb,phase

@dataclass
class World:
    walls: set
    objects: list
    identities: list
    pusher: int
    goal: int
    target: int
    kind: str
    phase: int = 0
    seed: int = 0
    steps: int = 0
    collisions: int = 0
    wrong_pushes: int = 0
    pushes: int = 0
    waits: int = 0

    def clone(self): return copy.deepcopy(self)

    @property
    def tokens(self): return np.array(divmod(self.identities[self.target],2),dtype=np.int64)

    @property
    def text(self):
        c,s=self.tokens
        return f'push the {COLORS[c]} {SHAPES[s]} to the exit'

    @property
    def state(self): return self.pusher,self.objects[self.target],self.phase

    def success(self): return self.objects[self.target]==self.goal and self.wrong_pushes==0

    def step(self, action):
        action=int(action)
        fixed=set(self.objects)-{self.objects[self.target]}
        nxt=transition(self.walls,self.kind,fixed,self.state,action)
        if action==4: self.waits+=1
        if nxt is None:
            dr,dc=MOVES[action]; r,c=rc(self.pusher); dest=cell(r+dr,c+dc)
            if dest in fixed: self.wrong_pushes+=1
            self.collisions+=1
            self.phase=advance_phase(self.kind,self.phase,set(self.objects)|{self.pusher})
        else:
            p,b,phase=nxt
            self.pushes+=int(b!=self.objects[self.target])
            self.pusher=p; self.objects[self.target]=b; self.phase=phase
        self.steps+=1

    def labels(self):
        z=np.zeros((N,N),dtype=np.int64)
        for p in self.walls: z[rc(p)]=1
        z[rc(self.goal)]=2
        for p in hazards(self.kind,self.phase): z[rc(p)]=12
        for p,i in zip(self.objects,self.identities): z[rc(p)]=4+i
        z[rc(self.pusher)]=3
        return z

    def render(self, scale=1):
        im=Image.new('RGB',(N*CELL,N*CELL),(19,27,44)); d=ImageDraw.Draw(im)
        for r in range(N):
            for c in range(N):
                x,y=c*CELL,r*CELL
                d.rectangle((x,y,x+7,y+7),outline=(25,35,53))
        for p in self.walls:
            r,c=rc(p); x,y=c*CELL,r*CELL
            d.rectangle((x,y,x+7,y+7),fill=(66,81,103))
            d.line((x,y,x+7,y),fill=(91,109,134))
        r,c=rc(self.goal); x,y=c*CELL,r*CELL
        d.rectangle((x,y,x+7,y+7),fill=(130,95,191))
        d.rectangle((x+2,y+2,x+5,y+5),outline=(230,209,255))
        for p in hazards(self.kind,self.phase):
            r,c=rc(p); x,y=c*CELL,r*CELL
            d.rectangle((x,y,x+7,y+7),fill=(224,107,46))
            d.line((x,y+7,x+7,y),fill=(255,209,128),width=2)
        for p,i in zip(self.objects,self.identities):
            r,c=rc(p); x,y=c*CELL,r*CELL; color=RGB[i//2]
            box=(x+1,y+1,x+6,y+6)
            if i%2: d.ellipse(box,fill=color)
            else: d.rectangle(box,fill=color)
        r,c=rc(self.pusher); x,y=c*CELL,r*CELL
        d.ellipse((x+1,y+1,x+6,y+6),fill=(239,245,255))
        d.point((x+3,y+3),fill=(46,61,81))
        return np.array(im.resize((im.width*scale,im.height*scale),Image.Resampling.NEAREST)) if scale!=1 else np.array(im)

def make_world(seed,kind='baffles',heldout=False):
    """Pure seed function. Solvability is checked by the caller, never assumed."""
    rng=np.random.default_rng(seed)
    walls={cell(r,c) for r in range(N) for c in range(N) if r in (0,N-1) or c in (0,N-1)}
    if kind=='baffles':
        k=int(rng.integers(4,7))
        walls|={cell(4,c) for c in range(1,k)}|{cell(7,c) for c in range(k+1,10)}
    elif kind=='gate': walls|={cell(5,c) for c in range(1,10) if c!=5}
    elif kind=='rotor': walls.add(cell(5,5))
    elif kind=='tower':
        widths={1:1,2:3,3:5,4:5,5:7,6:7,7:9,8:9,9:9}
        walls|={cell(r,c) for r in range(1,10) for c in range(1,10) if abs(c-5)>widths[r]//2}
        walls|={cell(6,3),cell(6,4),cell(4,6)}
    goal=cell(2,int(rng.integers(4,7))) if kind!='tower' else cell(2,5)
    spots=[cell(r,c) for r in (8,9) for c in range(2,9) if cell(r,c) not in walls]
    positions=list(map(int,rng.choice(spots,4,replace=False)))
    # Include same-colour / different-shape and same-shape / different-colour distractors.
    target_identity=5 if heldout else int(rng.choice([0,1,2,3,4,6,7]))
    identities=[target_identity,target_identity^1,((target_identity//2+1)%4)*2+target_identity%2]
    remaining=[i for i in range(8) if i not in identities]
    identities.append(int(rng.choice(remaining)))
    order=rng.permutation(4); identities=[identities[i] for i in order]
    target=identities.index(target_identity)
    # Scene positions are independent of requested identity.
    free=[cell(r,c) for r in range(7,10) for c in range(1,10) if cell(r,c) not in walls|set(positions)]
    pusher=int(rng.choice(free))
    phase=int(rng.integers(PERIOD)) if kind in ('gate','sweeper','rotor') else 0
    return World(walls,positions,identities,pusher,goal,target,kind,phase,seed)
