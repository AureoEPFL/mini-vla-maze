"""Display-only rendering. The model continues to receive the original RGB tiles."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .env import RGB,ACTION_NAMES

NAMES={'open':'Open floor','baffles':'Offset walls','gate':'Timed door','sweeper':'Moving barrier','rotor':'Alternating turnstile','tower':'Tower','switchback':'Switchback','catacombs':'Catacombs','airlocks':'Three airlocks'}
DESCRIPTIONS={'open':'Select the object, then approach its pushing side.','baffles':'Change sides to steer around offset walls.','gate':'Wait for the opening before crossing.','sweeper':'Avoid a blocker moving across the passage.','rotor':'Four arms alternate between two grid poses.','tower':'A narrow exit inside a tapered floor plan.','switchback':'Cross three offset passages and reverse direction.','catacombs':'Find the route through nine connected rooms.','airlocks':'Navigate the zigzag and three timed passages.'}
BG='#0d1421';FG='#eef3fa';MUTED='#9cacc2';ACCENT='#68dfc3'

def font(size):
    for name in ('DejaVuSans.ttf','arial.ttf'):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()

def board(labels,side=500,trail=()):
    n=len(labels);tile=side/n;im=Image.new('RGB',(side,side),'#131f30');d=ImageDraw.Draw(im)
    for r in range(n):
        for c in range(n):
            z=int(labels[r,c]);x,y=c*tile,r*tile;box=(x+1,y+1,x+tile-1,y+tile-1)
            if z==1:d.rounded_rectangle(box,radius=max(1,tile*.12),fill='#38485f')
            else:
                d.rectangle(box,fill='#182639')
                if z==2:
                    d.rounded_rectangle((x+2,y+2,x+tile-2,y+tile-2),radius=3,fill='#785fc5')
                    d.rectangle((x+tile*.3,y+tile*.3,x+tile*.7,y+tile*.7),outline='#ede6ff',width=max(1,int(tile*.06)))
                elif z==12:
                    d.rounded_rectangle(box,radius=2,fill='#ca7439')
                    d.line((x+3,y+tile-4,x+tile-4,y+3),fill='#ffe2ac',width=max(2,int(tile*.13)))
    if len(trail)>1:
        pts=[((int(p)%n+.5)*tile,(int(p)//n+.5)*tile) for p in trail]
        d.line(pts,fill='#587d80',width=max(1,int(tile*.11)))
    for r in range(n):
        for c in range(n):
            z=int(labels[r,c]);x,y=(c+.5)*tile,(r+.5)*tile;rad=tile*.32
            if z==3:
                d.ellipse((x-rad-2,y-rad-2,x+rad+2,y+rad+2),fill='#243a4b')
                d.ellipse((x-rad,y-rad,x+rad,y+rad),fill='#f3f7ff')
                d.ellipse((x-rad*.3,y-rad*.3,x+rad*.3,y+rad*.3),fill='#546b83')
            elif 4<=z<12:
                i=z-4;color=RGB[i//2];b=(x-rad,y-rad,x+rad,y+rad)
                if i%2:d.ellipse(b,fill=color)
                else:d.rounded_rectangle(b,radius=max(1,tile*.07),fill=color)
    return im

def animation(frames,row,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);output=[];kind=row.get('arena',row.get('kind'));mode=row['mode'];trail=[]
    for i,f in enumerate(frames):
        im=Image.new('RGB',(1000,680),BG);d=ImageDraw.Draw(im)
        d.text((30,23),'MINI-VLA / EXPERIMENT VIEWER',font=font(13),fill=ACCENT)
        d.text((30,51),NAMES[kind],font=font(30),fill=FG)
        d.text((30,94),DESCRIPTIONS[kind],font=font(15),fill=MUTED)
        trail.append(f['target']);im.paste(board(f['labels'],500,trail),(30,140))
        d.rounded_rectangle((560,140,970,640),radius=14,fill='#172334')
        controller={'hybrid':'LEARNED VISION + A*','direct':'NEURAL ACTION POLICY','oracle':'PRIVILEGED-STATE A*'}[mode]
        d.text((582,160),controller,font=font(14),fill=ACCENT)
        words=row['instruction'].replace('push the ','').replace(' to the exit','')
        d.text((582,190),'Push the '+words,font=font(23),fill=FG);d.text((582,220),'to the purple exit',font=font(17),fill=MUTED)
        action=ACTION_NAMES[f['action']].upper() if i<len(frames)-1 else ('DONE' if row['success'] else 'STOP')
        for x,y,label,value in [(582,264,'STEP',f"{f['step']} / {row['steps']}"),(777,264,'ACTION',action),(582,334,'PUSHES',str(f['pushes'])),(777,334,'CLOCK PHASE',str(f['phase']))]:
            d.text((x,y),label,font=font(12),fill=MUTED);d.text((x,y+22),value,font=font(24),fill=FG)
        d.line((582,405,948,405),fill='#344257')
        d.text((582,425),'MODEL INPUT',font=font(11),fill=MUTED)
        im.paste(Image.fromarray(f.get('observation',f['rgb'])).resize((112,112),Image.Resampling.NEAREST),(582,448))
        if f['attention'] is not None:
            a=f['attention'];a=a/max(float(a.max()),1e-9);rgb=np.stack((30+210*a,35+195*a,55+120*a),-1).clip(0,255).astype(np.uint8)
            im.paste(Image.fromarray(rgb).resize((112,112),Image.Resampling.NEAREST),(732,448))
            d.text((732,425),'TARGET ATTENTION',font=font(11),fill=MUTED)
        final=i==len(frames)-1;status=row['outcome'].replace('_',' ').upper() if final else 'OBSERVE  /  ACT  /  REPEAT'
        d.text((582,589),status,font=font(16),fill=ACCENT if not final or row['success'] else '#ffbf78')
        d.text((30,654),f"Seed {row['seed']}   |   {row.get('condition','clean')}   |   display styling only; raw input shown at right",font=font(12),fill=MUTED)
        d.rounded_rectangle((30,126,970,130),radius=2,fill='#243449')
        if i:d.rounded_rectangle((30,126,30+940*i/max(1,len(frames)-1),130),radius=2,fill=ACCENT)
        output.append(im)
    duration=[90]*len(output);duration[-1]=1800
    output[0].save(path,save_all=True,append_images=output[1:],duration=duration,loop=0,optimize=True)
