"""Reproducible figures and labelled animations from measured rollouts."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import torch
from .env import N,CELL,ACTION_NAMES,make_world,rc
from .evaluate import load_model,rollout

BG='#101a2c'; FG='#edf4ff'; MUTED='#9cacc5'; TEAL='#43d6b3'; GOLD='#ffc663'

def font(size):
    for name in ('DejaVuSans.ttf','arial.ttf'):
        try: return ImageFont.truetype(name,size)
        except OSError: pass
    return ImageFont.load_default()

def animation(frames,row,path):
    rendered=[]; f18=font(18); f24=font(24); f14=font(14)
    for frame in frames:
        im=Image.new('RGB',(900,530),BG); d=ImageDraw.Draw(im)
        title={'direct':'Neural action policy','hybrid':'Learned vision + A* planner','oracle':'Privileged-state planner'}[row['mode']]
        d.text((26,18),'MINI-VLA  /  MAZE LAB',font=f14,fill=TEAL)
        d.text((26,44),title,font=f24,fill=FG)
        d.text((26,79),row['instruction'],font=f18,fill=FG)
        rgb=Image.fromarray(frame['rgb']).resize((352,352),Image.Resampling.NEAREST)
        im.paste(rgb,(26,119))
        d.text((26,482),f"{row['kind']}  |  seed {row['seed']}  |  {row['condition']}",font=f14,fill=MUTED)
        d.text((414,119),'LANGUAGE-CONDITIONED ATTENTION',font=f14,fill=TEAL)
        if frame['attention'] is not None:
            heat=frame['attention']; heat=heat/max(float(heat.max()),1e-9)
            color=(plt.get_cmap('magma')(heat)[:,:,:3]*255).astype(np.uint8)
            im.paste(Image.fromarray(color).resize((176,176),Image.Resampling.NEAREST),(414,148))
        else: d.text((414,175),'No learned perception',font=f14,fill=MUTED)
        fields=[('STEP',str(frame['step'])),('ACTION',ACTION_NAMES[frame['action']].upper()),('OBSTACLE PHASE',f"{frame['phase']} / 5"),('PUSHES',str(frame['pushes'])),('BLOCKED MOVES',str(frame['collisions'])),('MODEL CALLS',str(frame['calls']))]
        for j,(label,value) in enumerate(fields):
            y=142+j*47; d.text((628,y),label,font=f14,fill=MUTED); d.text((628,y+17),value,font=f18,fill=FG)
        d.text((414,352),'White: pusher   Purple: exit',font=f14,fill=MUTED)
        d.text((414,376),'Orange: moving obstacle',font=f14,fill=MUTED)
        d.text((414,407),f"Execute {row['execute']} action(s) per observation",font=f14,fill=FG)
        final=frame is frames[-1]
        status=row['outcome'].replace('_',' ').upper() if final else 'CLOSED-LOOP ROLLOUT'
        d.rounded_rectangle((414,443,866,487),radius=9,fill='#20344a')
        d.text((428,454),status,font=f18,fill=TEAL if row['success'] else GOLD)
        rendered.append(im)
    durations=[125]*len(rendered); durations[-1]=1700
    rendered[0].save(path,save_all=True,append_images=rendered[1:],duration=durations,loop=0,optimize=False)

def plots(training,metrics,rows,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    h=training['history']; x=[r['step'] for r in h]
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    axes[0].plot(x,[r['action_ce'] for r in h],label='train batch'); axes[0].plot(x,[r['val_action_ce'] for r in h],label='validation'); axes[0].set(ylabel='Action cross entropy'); axes[0].legend()
    for key,label in [('val_action_accuracy','next action'),('val_target_accuracy','target cell'),('val_map_accuracy','semantic cell')]: axes[1].plot(x,[r[key] for r in h],label=label)
    axes[1].set(ylabel='Validation accuracy',ylim=(0,1.03)); axes[1].legend()
    axes[2].plot(x,[r['seconds'] for r in h],color='#198b78'); axes[2].set(ylabel='Elapsed training seconds')
    for a in axes:a.set_xlabel('Gradient updates')
    fig.tight_layout(); fig.savefig(out/'learning.png',dpi=170); plt.close(fig)
    suites=['open','baffles','gate','sweeper','unseen_rotor','unseen_tower','heldout_blue_disk']
    fig,ax=plt.subplots(figsize=(12,4.8)); xx=np.arange(len(suites)); width=.2
    for j,(mode,k,label,color) in enumerate([('oracle',1,'Oracle','#b2bbc9'),('direct',1,'Neural k=1','#3d82d6'),('direct',4,'Neural k=4','#b077da'),('hybrid',1,'Vision + A*','#29aa8d')]):
        vals=[metrics[f'{s}/{mode}/k{k}'] for s in suites]; rates=np.array([v['rate'] for v in vals]); err=np.array([[max(0,rates[i]-v['wilson_95'][0]),max(0,v['wilson_95'][1]-rates[i])] for i,v in enumerate(vals)]).T
        ax.bar(xx+(j-1.5)*width,rates,width,yerr=err,capsize=2,label=label,color=color)
    ax.set_xticks(xx,labels=[s.replace('_','\n') for s in suites]); ax.set(ylabel='Episode success (Wilson 95% CI)',ylim=(0,1.12)); ax.legend(ncol=4,loc='upper center'); fig.tight_layout(); fig.savefig(out/'success.png',dpi=170); plt.close(fig)
    stresses=['wrong_shape','wrong_color','blank','dark','noise','displacement']; values=np.array([[metrics[f'{s}/{m}/k1']['rate'] for s in stresses] for m in ('direct','hybrid')])
    fig,ax=plt.subplots(figsize=(10,3)); ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto'); ax.set_xticks(range(6),labels=stresses); ax.set_yticks(range(2),labels=['Neural k=1','Vision + A*'])
    for i in range(2):
        for j in range(6):ax.text(j,i,f'{values[i,j]:.0%}',ha='center',va='center',color='white' if values[i,j]>.6 else 'black')
    ax.set_title('Stress tests: success against the ORIGINAL task'); fig.tight_layout(); fig.savefig(out/'stress.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    names=['direct/k1','direct/k4','hybrid/k1']; selected=[r for r in rows if r['suite'] in suites[:4]]
    for j,name in enumerate(names):
        m,k=name.split('/k'); group=[r for r in selected if r['mode']==m and int(r['execute'])==int(k)]
        outcomes=['success','timeout','distractor_contact']; bottom=0
        for outcome,color in zip(outcomes,['#29aa8d','#efbb63','#de647b']):
            v=sum(r['outcome']==outcome for r in group)/len(group); axes[0].bar(j,v,bottom=bottom,color=color,label=outcome if j==0 else None); bottom+=v
        axes[1].bar(j,np.mean([int(r['model_calls']) for r in group]),color=['#3d82d6','#b077da','#29aa8d'][j])
    axes[0].legend(fontsize=8); axes[0].set(ylabel='Fraction of episodes',title='Failure decomposition: familiar layouts')
    axes[1].set(ylabel='Mean network calls per episode',title='Observation cost (includes failed episodes)')
    for a in axes:a.set_xticks(range(3),labels=names)
    fig.tight_layout(); fig.savefig(out/'failures_cost.png',dpi=170); plt.close(fig)

def run(run_dir,out):
    run_dir=Path(run_dir); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    training=json.loads((run_dir/'training.json').read_text()); metrics=json.loads((run_dir/'eval/metrics.json').read_text())
    with (run_dir/'eval/episodes.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    plots(training,metrics,rows,out)
    torch.set_num_threads(min(4,torch.get_num_threads())); model=load_model(run_dir/'weights.pt'); manifest=[]
    for suite,mode in [('baffles','direct'),('gate','direct'),('gate','hybrid'),('sweeper','hybrid'),('unseen_rotor','hybrid'),('unseen_tower','hybrid'),('heldout_blue_disk','hybrid')]:
        group=[r for r in rows if r['suite']==suite and r['mode']==mode and int(r['execute'])==1]
        for outcome in ('success','timeout','distractor_contact'):
            candidates=[r for r in group if r['outcome']==outcome]
            if not candidates:continue
            r=candidates[0]; w=make_world(int(r['seed']),r['kind'],suite=='heldout_blue_disk'); row,frames=rollout(model,w,mode,1,record=True)
            filename=f'{suite}_{mode}_{outcome}.gif'; animation(frames,row,out/filename)
            manifest.append({'file':filename,'selection':'first episode with this outcome in declared test order',**row})
            print('Animation',filename,flush=True)
    # Same initial pixels, two different commands, both solvable. Separate illustration set.
    from .planner import expert
    found=False
    for seed in range(400000,400200):
        w=make_world(seed,'baffles'); targets=[]
        for target in range(4):
            q=w.clone(); q.target=target
            if expert(q) is not None: targets.append(target)
        if len(targets)<2:continue
        for j,target in enumerate(targets[:2]):
            q=w.clone(); q.target=target; row,frames=rollout(model,q,'hybrid',record=True)
            filename=f'counterfactual_{j}.gif'; animation(frames,row,out/filename)
            manifest.append({'file':filename,'selection':'first illustration seed with two oracle-solvable commands; excluded from benchmark',**row})
        found=True; break
    assert found
    (out/'gallery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Spatial trajectories and the actual gate clock expose more than success rate.
    fig,axes=plt.subplots(1,3,figsize=(13,4.6)); gate_frames=None
    for ax,suite,mode in zip(axes,['baffles','gate','unseen_tower'],['direct','hybrid','hybrid']):
        selected=next(r for r in manifest if r['file']==f'{suite}_{mode}_success.gif')
        w=make_world(selected['seed'],selected['kind']); row,frames=rollout(model,w,mode,record=True)
        ax.imshow(w.render(),alpha=.65)
        for key,color,label in [('pusher','#07a9a4','pusher'),('target','#ff4b70','object')]:
            xy=np.array([(rc(f[key])[1]*CELL+CELL/2,rc(f[key])[0]*CELL+CELL/2) for f in frames])
            ax.plot(xy[:,0],xy[:,1],'-o',color=color,markersize=3,linewidth=2,label=label)
            ax.scatter(*xy[0],marker='s',s=65,color=color,edgecolor='white',zorder=5)
            ax.scatter(*xy[-1],marker='*',s=130,color=color,edgecolor='white',zorder=5)
        ax.set_title(f"{selected['kind']} | {mode}\n{row['pushes']} pushes, {row['steps']} steps"); ax.axis('off')
        if suite=='gate':gate_frames=frames
    axes[0].legend(loc='lower left',fontsize=8);fig.suptitle('Recorded trajectories: square = start, star = finish');fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=170);plt.close(fig)
    if gate_frames:
        fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
        ts=[f['step'] for f in gate_frames]
        axes[0].step(ts,[f['phase'] for f in gate_frames],where='post',color='#298b85');axes[0].set(ylabel='Actual obstacle phase',yticks=range(6),title='Gate rollout: the safety interlock can pause its clock')
        waits=[f['step'] for f in gate_frames[:-1] if f['action']==4]
        for t in waits:axes[0].axvline(t,color='#e39b39',linestyle='--',label='wait command' if t==waits[0] else None)
        if waits:axes[0].legend()
        axes[1].step(ts,[rc(f['target'])[0] for f in gate_frames],where='post',label='object row',color='#da5070');axes[1].step(ts,[rc(f['pusher'])[0] for f in gate_frames],where='post',label='pusher row',color='#3c83c9');axes[1].set(xlabel='Simulator step',ylabel='Grid row');axes[1].legend()
        fig.tight_layout();fig.savefig(out/'gate_timeline.png',dpi=170);plt.close(fig)
    # Contact sheet of every animation's first frame, for quick verification.
    thumbs=[]
    for item in manifest:
        im=Image.open(out/item['file']).convert('RGB'); im.thumbnail((450,265)); thumbs.append(im.copy())
    sheet=Image.new('RGB',(900,265*((len(thumbs)+1)//2)),BG)
    for i,im in enumerate(thumbs): sheet.paste(im,((i%2)*450,(i//2)*265))
    sheet.save(out/'gallery_contact_sheet.png')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',default='runs/maze'); p.add_argument('--out',default='runs/maze/media'); a=p.parse_args(); run(a.run,a.out)

if __name__=='__main__': main()
