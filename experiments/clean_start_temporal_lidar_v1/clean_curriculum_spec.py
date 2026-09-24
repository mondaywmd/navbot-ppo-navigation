"""Independent fresh mirrored curriculum for one through five pillars."""
import heapq, math, random

ARENA_LIMIT=3.55; RADIUS_RANGE=(0.18,0.45); START_CLEARANCE=0.55
GOAL_CLEARANCE=0.55; INTER_PILLAR_CLEARANCE=0.25; ROBOT_RADIUS=0.105; GRID=0.10
LEVELS={
 1:{"obstacles":1,"target_x":(1.4,3.0),"target_abs_y":1.2,"path_lateral":0.75},
 2:{"obstacles":2,"target_x":(2.0,3.2),"target_abs_y":1.2,"path_lateral":0.85},
 3:{"obstacles":3,"target_x":(2.6,3.3),"target_abs_y":1.1,"path_lateral":0.95},
 4:{"obstacles":4,"target_x":(2.8,3.3),"target_abs_y":1.1,"path_lateral":1.25},
 5:{"obstacles":5,"target_x":(2.9,3.3),"target_abs_y":1.0,"path_lateral":1.45}}

def mirror(s):
 return {**s,"name":s["name"]+"_mirror","target_y":-s["target_y"],
  "robot_yaw_deg":-s["robot_yaw_deg"],"obstacles":[{"x":p["x"],"y":-p["y"],"radius":p["radius"]} for p in s["obstacles"]]}

def point_segment_distance(px,py,x1,y1):
 q=max(0.,min(1.,(px*x1+py*y1)/(x1*x1+y1*y1)));return math.hypot(px-q*x1,py-q*y1)

def direct_path_blocked(s):
 return any(point_segment_distance(p["x"],p["y"],s["target_x"],s["target_y"])<=p["radius"]+ROBOT_RADIUS for p in s["obstacles"])

def local_clearance(x,y,s):
 gd=math.hypot(s["target_x"]-x,s["target_y"]-y);return 0.25+0.15*min(1.,gd/0.50)

def has_clearance_route(s):
 start=(0,0);goal=(round(s["target_x"]/GRID),round(s["target_y"]/GRID))
 def blocked(n):
  x,y=n[0]*GRID,n[1]*GRID
  if not(-.45<=x<=3.65 and -3.45<=y<=3.45):return True
  c=local_clearance(x,y,s);return any(math.hypot(x-p["x"],y-p["y"])<=p["radius"]+c for p in s["obstacles"])
 queue=[(0.,start)];cost={start:0.};moves=((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
 while queue:
  _,node=heapq.heappop(queue)
  if node==goal:return True
  for dx,dy in moves:
   nxt=(node[0]+dx,node[1]+dy)
   if blocked(nxt):continue
   candidate=cost[node]+math.hypot(dx,dy)
   if candidate<cost.get(nxt,float("inf")):
    cost[nxt]=candidate;heapq.heappush(queue,(candidate+math.hypot(nxt[0]-goal[0],nxt[1]-goal[1]),nxt))
 return False

def legal(s):
 tx,ty=s["target_x"],s["target_y"];obs=s["obstacles"]
 if max(abs(tx),abs(ty))>ARENA_LIMIT or math.hypot(tx,ty)<1.2:return False
 for p in obs:
  if max(abs(p["x"]),abs(p["y"]))>ARENA_LIMIT or not RADIUS_RANGE[0]<=p["radius"]<=RADIUS_RANGE[1]:return False
  if math.hypot(p["x"],p["y"])<p["radius"]+START_CLEARANCE:return False
  if math.hypot(p["x"]-tx,p["y"]-ty)<p["radius"]+GOAL_CLEARANCE:return False
 if not all(math.hypot(a["x"]-b["x"],a["y"]-b["y"])>=a["radius"]+b["radius"]+INTER_PILLAR_CLEARANCE for i,a in enumerate(obs) for b in obs[i+1:]):return False
 return direct_path_blocked(s) and has_clearance_route(s)

def generate(level,count,seed):
 cfg=LEVELS[level];rng=random.Random(seed+level);out=[];attempts=0
 while len(out)<count:
  attempts+=1
  if attempts>count*10000:raise RuntimeError("could not generate legal level %d scenes"%level)
  tx=rng.uniform(*cfg["target_x"]);ty=rng.uniform(-cfg["target_abs_y"],cfg["target_abs_y"]);length=math.hypot(tx,ty);nx,ny=-ty/length,tx/length
  obs=[];blocker=rng.randrange(cfg["obstacles"]);fractions=[(i+1.)/(cfg["obstacles"]+1.) for i in range(cfg["obstacles"])]
  for i,f in enumerate(fractions):
   radius=rng.uniform(*RADIUS_RANGE);limit=radius+ROBOT_RADIUS-.02 if i==blocker else cfg["path_lateral"]
   lateral=rng.uniform(-limit,limit);along=f+rng.uniform(-.08,.08)
   obs.append({"x":tx*along+nx*lateral,"y":ty*along+ny*lateral,"radius":radius})
  s={"name":"level%d_%05d"%(level,len(out)+1),"level":level,"target_x":tx,"target_y":ty,"robot_yaw_deg":rng.uniform(-15,15),"obstacles":obs,"generation_seed":seed}
  if legal(s) and legal(mirror(s)):out.append(s)
 return out
