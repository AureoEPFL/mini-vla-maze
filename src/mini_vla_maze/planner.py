"""A* in (pusher cell, target cell, obstacle phase); unit-cost actions."""
import heapq
from .env import rc, transition

def plan(walls,kind,fixed,start,goal,max_nodes=45000):
    if start[1]==goal: return []
    def h(s):
        br,bc=rc(s[1]); gr,gc=rc(goal)
        return abs(br-gr)+abs(bc-gc)
    queue=[(h(start),0,start)]; cost={start:0}; parents={}
    while queue and len(cost)<max_nodes:
        _,g,s=heapq.heappop(queue)
        if g!=cost[s]: continue
        if s[1]==goal:
            path=[]
            while s!=start:
                prev,a=parents[s]; path.append(a); s=prev
            return path[::-1]
        for a in range(5):
            nxt=transition(walls,kind,fixed,s,a)
            if nxt is None or nxt==s: continue
            ng=g+1
            if ng<cost.get(nxt,10**9):
                cost[nxt]=ng; parents[nxt]=(s,a)
                heapq.heappush(queue,(ng+h(nxt),ng,nxt))
    return None

def expert(world):
    return plan(world.walls,world.kind,set(world.objects)-{world.objects[world.target]},world.state,world.goal)
