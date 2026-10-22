"""Build the paper figures and illustrated gallery from saved measurements."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Patch
from .evaluate import load_model,rollout
from .env import make_world
from .large import make_arena,rollout as large_rollout,ARENAS
from .presentation import animation,board,NAMES,DESCRIPTIONS

def paper_architecture(out):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10})
    fig,ax=plt.subplots(figsize=(15,7));ax.set(xlim=(0,14),ylim=(0,7));ax.axis('off')
    learned='#e8edf8';fixed='#f0f0ed';derived='#e8f2ed'
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=.05',facecolor=color,edgecolor='#465064',linewidth=.9))
        ax.text(x+w/2,y+h-.22,title,ha='center',va='top',fontsize=11,weight='bold')
        ax.text(x+w/2,y+h-.57,body,ha='center',va='top',fontsize=9,linespacing=1.4)
    def arrow(a,b,label=''):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,linewidth=1,color='#3e4654'))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=8)
    ax.text(.3,6.8,'(a)  Shared visual representation and supervised heads',fontsize=13)
    box(.3,5.0,2.4,1.25,'RGB observation',r'$I_t\in\mathbb{R}^{3\times 8N\times 8N}$'+'\nRegistered overhead image',fixed)
    box(3.1,5.0,2.7,1.25,'Patch encoder','Conv 8×8, stride 8 → ReLU\nConv 1×1 → ReLU',learned)
    box(6.3,5.0,2.7,1.25,'Visual features',r'$F_t\in\mathbb{R}^{32\times N\times N}$'+'\nOne feature vector per cell',derived)
    box(9.7,5.0,3.6,1.25,'Semantic head','Conv 1×1 → 13 class logits\nFloor, wall, exit, pusher, objects, hazard',learned)
    box(.3,2.9,2.4,1.25,'Instruction','Fixed grammar → colour, shape\nTwo discrete token IDs',fixed)
    box(3.1,2.9,2.7,1.25,'Attribute embeddings',r'$q_c,q_s\in\mathbb{R}^{16}$'+'\nTables: 4 colours, 2 shapes',learned)
    box(6.3,2.9,2.7,1.25,'Factorised grounding',r'$\alpha=\mathrm{softmax}((q_cK_c+q_sK_s)/4)$'+'\nSpatial distribution over N² cells',learned)
    box(9.7,2.9,3.6,1.25,'Action head  (N = 11)','CNN on [F, α] → flatten\n[1970] → MLP 192 → 4×5 logits',learned)
    arrow((2.7,5.62),(3.1,5.62));arrow((5.8,5.62),(6.3,5.62));arrow((9,5.62),(9.7,5.62))
    arrow((2.7,3.52),(3.1,3.52));arrow((5.8,3.52),(6.3,3.52));arrow((7.65,5.0),(7.65,4.15),'keys');arrow((9,3.52),(9.7,3.52));arrow((9,5.1),(9.7,4.1),'F')
    ax.text(.3,2.1,'(b)  Two control paths; identical trained checkpoint',fontsize=13)
    box(.3,.55,6.2,1.1,'Hybrid: learned map + target → A*','Known grid transitions + mechanism schedule + measured phase\nReplan on state mismatch; supports N = 11 and N = 25',fixed)
    box(7.1,.55,6.2,1.1,'Direct: action logits → execute k actions','Observed phase encoded as sin/cos; colour + shape embeddings\nk = 1 or 4, then observe again; fixed N = 11',learned)
    ax.legend(handles=[Patch(facecolor=learned,edgecolor='#465064',label='learned'),Patch(facecolor=fixed,edgecolor='#465064',label='input / programmed component'),Patch(facecolor=derived,edgecolor='#465064',label='intermediate tensor')],loc='lower center',bbox_to_anchor=(.5,-.05),ncol=3,frameon=False,fontsize=9)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(out/f'paper_architecture.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)

def benchmark_figure(root,out):
    metrics=json.loads((root/'results/maze/metrics.json').read_text());manifest=json.loads((root/'results/maze/test_manifest.json').read_text())
    kinds=['open','baffles','gate','sweeper','rotor','tower'];suites=['open','baffles','gate','sweeper','unseen_rotor','unseen_tower']
    fig=plt.figure(figsize=(15,10));outer=fig.add_gridspec(2,3,hspace=.42,wspace=.65)
    for index,(kind,suite) in enumerate(zip(kinds,suites)):
        inner=outer[index//3,index%3].subgridspec(2,1,height_ratios=[1.45,1],hspace=.08)
        ax=fig.add_subplot(inner[0]);w=make_world(manifest[suite]['accepted_seeds'][0],kind);ax.imshow(board(w.labels(),220));ax.axis('off');ax.set_title(NAMES[kind]+(' *' if index>3 else ''),fontsize=12,fontweight='bold')
        a=fig.add_subplot(inner[1]);colors=['#99a1ac','#4477aa','#aa77aa','#228877']
        for j,(mode,k) in enumerate([('oracle',1),('direct',1),('direct',4),('hybrid',1)]):
            m=metrics[f'{suite}/{mode}/k{k}'];v=m['rate'];lo,hi=m['wilson_95'];a.barh(j,v,color=colors[j],height=.62)
            a.errorbar(v,j,xerr=np.array([[max(0,v-lo)],[max(0,hi-v)]]),color='#252525',capsize=2,linewidth=.8)
            a.text(1.04,j,f"{m['successes']}/{m['episodes']}",va='center',fontsize=9)
        a.set_yticks(range(4),labels=['Oracle','Neural k=1','Neural k=4','Vision + A*'],fontsize=9);a.set_xlim(0,1.23);a.set_xticks([0,.5,1],labels=['0','50%','100%']);a.invert_yaxis();a.spines[['top','right']].set_visible(False);a.grid(axis='x',alpha=.18);a.set_axisbelow(True)
    fig.suptitle('Closed-loop success on the 11×11 benchmark',fontsize=16,y=.98)
    fig.text(.5,.025,'16 accepted scenes per condition; whiskers: Wilson 95% intervals.  * Layout absent from training.  One training seed.',ha='center',fontsize=10)
    fig.savefig(out/'benchmark_illustrated.png',dpi=180,bbox_inches='tight');fig.savefig(out/'benchmark_illustrated.svg',bbox_inches='tight');plt.close(fig)

def run(root):
    root=Path(root);out=root/'media';out.mkdir(exist_ok=True);torch.set_num_threads(min(4,torch.get_num_threads()));model=load_model(root/'results/maze/weights.pt')
    paper_architecture(out);benchmark_figure(root,out)
    manifest=json.loads((root/'results/maze/test_manifest.json').read_text());large_manifest=json.loads((root/'results/large/manifest.json').read_text());items=[]
    selections=[('baffles','direct','success'),('gate','hybrid','success'),('rotor','hybrid','success'),('tower','hybrid','success'),('baffles','direct','timeout')]
    with (root/'results/maze/episodes.csv').open() as f:rows=list(csv.DictReader(f))
    for kind,mode,outcome in selections:
        suite={'rotor':'unseen_rotor','tower':'unseen_tower'}.get(kind,kind)
        selected=next(r for r in rows if r['suite']==suite and r['mode']==mode and r['execute']=='1' and r['outcome']==outcome)
        w=make_world(int(selected['seed']),kind);row,frames=rollout(model,w,mode,record=True);name=f'viewer_{kind}_{mode}_{outcome}.gif'
        animation(frames,row,out/name);board(frames[0]['labels'],500).save(out/f'scene_{kind}.png');items.append({'file':name,**row});print(name,flush=True)
    for kind in ARENAS:
        seed=large_manifest[kind]['accepted'][0];w=make_arena(seed,kind);row,frames=large_rollout(model,w,record=True);name=f'viewer_{kind}.gif'
        animation(frames,row,out/name);board(frames[0]['labels'],500).save(out/f'scene_{kind}.png');items.append({'file':name,**row});print(name,flush=True)
    for kind in ('open','sweeper'):
        w=make_world(manifest[kind]['accepted_seeds'][0],kind);board(w.labels(),500).save(out/f'scene_{kind}.png')
    (out/'viewer_manifest.json').write_text(json.dumps(items,indent=2))
    # Simple portable viewer: no service, telemetry, fonts or build tool.
    payload=json.dumps(items).replace('</','<\\/')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mini-VLA Maze · experiments</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0d1421;color:#eef3fa;font:16px system-ui,sans-serif}main{max-width:1300px;margin:auto;padding:32px}header{border-bottom:1px solid #304056;padding-bottom:24px}h1{font-size:38px;letter-spacing:-1px;margin:8px 0}p{color:#aebdd2;line-height:1.6}.eyebrow{color:#68dfc3;font-size:12px;letter-spacing:2px}.layout{display:grid;grid-template-columns:240px 1fr;gap:24px;margin-top:28px}button{display:block;text-align:left;width:100%;padding:14px;margin:0 0 8px;background:#172334;border:1px solid #304056;color:#eef3fa;border-radius:8px;cursor:pointer}button.selected{border-color:#68dfc3;background:#1d3a3c}button small{display:block;color:#9cacc2;margin-top:5px}.stage img{width:100%;border-radius:12px}.note{font-size:13px}a{color:#68dfc3}.stats{display:flex;gap:20px;flex-wrap:wrap;margin:12px 0}.stat{background:#172334;padding:12px 22px;border-radius:8px}.stat strong{font-size:25px;display:block}.stat span{font-size:12px;color:#aebdd2}@media(max-width:800px){.layout{grid-template-columns:1fr}nav{display:flex;overflow:auto;gap:8px}button{min-width:160px}main{padding:16px}}</style>
<main><header><div class="eyebrow">VISION · LANGUAGE · ACTION</div><h1>Mini-VLA Maze</h1><p>A small learned policy, an explicit planner, and a set of increasingly demanding pushing tasks.</p></header><div class="layout"><nav id="cases"></nav><section class="stage"><img id="film" alt="Recorded experiment"><div class="stats" id="stats"></div><p id="caption"></p><p class="note">These are recorded executions. The styled view is for display; the neural model receives the raw RGB image shown in the animation. Large arenas use learned perception + A*. No neural action-head result is claimed on 25×25 maps.</p><p><a href="README.md">README</a> · <a href="RESULTS.md">Measured results</a> · <a href="media/benchmark_illustrated.png">Illustrated benchmark</a> · <a href="docs/METHODS.md">Methods</a></p></section></div></main><script>const cases=PAYLOAD;const names=NAMES;function select(i){const r=cases[i],kind=r.arena||r.kind;document.getElementById('film').src='media/'+r.file;document.getElementById('stats').innerHTML=[['steps',r.steps],['pushes',r.pushes],['waits',r.waits],['seed',r.seed]].map(([k,v])=>'<div class="stat"><strong>'+v+'</strong><span>'+k+'</span></div>').join('');document.getElementById('caption').textContent=r.instruction+' — '+r.outcome.replaceAll('_',' ')+'. '+(r.mode==='hybrid'?'Learned visual representation; programmed A* route.':'Actions predicted by the neural network.');document.querySelectorAll('button').forEach((b,j)=>b.classList.toggle('selected',j===i))}cases.forEach((r,i)=>{const b=document.createElement('button');b.innerHTML=names[r.arena||r.kind]+'<small>'+r.mode+' · '+r.outcome.replaceAll('_',' ')+'</small>';b.onclick=()=>select(i);document.getElementById('cases').appendChild(b)});select(cases.findIndex(r=>r.arena==='catacombs'));</script></html>'''.replace('PAYLOAD',payload).replace('NAMES',json.dumps(NAMES))
    (root/'gallery.html').write_text(html,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();run(a.root)
