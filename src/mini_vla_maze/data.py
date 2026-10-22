import json
import numpy as np
from pathlib import Path
from .env import make_world
from .planner import expert
from .model import HORIZON

TRAIN_KINDS=('open','baffles','gate','sweeper')

def collect(episodes=320,start_seed=10000,kinds=TRAIN_KINDS):
    records=[]; manifest=[]; rejected=[]
    candidate=start_seed
    while len(manifest)<episodes:
        kind=kinds[len(manifest)%len(kinds)]
        w=make_world(candidate,kind); candidate+=1
        path=expert(w)
        if path is None or not 4<=len(path)<=100:
            rejected.append(w.seed); continue
        manifest.append({'seed':w.seed,'kind':kind,'length':len(path),'text':w.text})
        for j,a in enumerate(path):
            chunk=(path[j:j+HORIZON]+[4]*HORIZON)[:HORIZON]
            records.append((w.render(),w.tokens,w.phase,w.labels(),w.objects[w.target],chunk))
            w.step(a)
        assert w.success(), (w.seed,kind)
        if len(manifest)%40==0: print('Demonstrations',len(manifest),'/',episodes,'samples',len(records),flush=True)
    columns=list(zip(*records))
    names=('image','tokens','phase','labels','target','actions')
    arrays={k:np.array(v,dtype=np.uint8) for k,v in zip(names,columns)}
    return arrays,{'accepted':manifest,'rejected_seeds':rejected,'start_seed':start_seed,'next_seed':candidate}

def save_dataset(path,arrays,manifest):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/'data.npz',**arrays)
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def load_dataset(path):
    with np.load(Path(path)/'data.npz') as data: return {k:data[k] for k in data.files}
