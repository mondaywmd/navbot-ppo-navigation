"""Online clean-start PPO with fresh uniformly random 1--5 pillar scenes."""
import argparse, json, math, os, random, time
import numpy as np
import rospy, torch
from torch import nn
from torch.distributions import Normal
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from clean_actor import CleanMirrorActor
from clean_curriculum_spec import generate
from clean_ppo_reward import transition_reward
from clean_wide_env import CleanWideStaticEnv
from observation78 import build_observation
from potential_escape import PotentialEscapeShield
from safety_shield import valid_scan
from temporal_lidar import sector_minima, signed_range_rate, temporal_features

STEP_LIMIT=200

class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(78,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
def decode(a):return -0.10+(float(np.clip(a[0],-1,1))+1)*.35/2,.8*float(np.clip(a[1],-1,1))
def normalize(v,w):return np.asarray([np.clip(2*(v+.10)/.35-1,-1,1),np.clip(w/.8,-1,1)],np.float32)
def pose(env,scene):
    s=env.get_model_state("turtlebot3_burger","world");x,y=s.pose.position.x,s.pose.position.y;q=s.pose.orientation
    yaw=math.atan2(2*q.w*q.z,1-2*q.z*q.z);d=math.hypot(scene["target_x"]-x,scene["target_y"]-y)
    return d,wrap(math.atan2(scene["target_y"]-y,scene["target_x"]-x)-yaw)

def collect_episode(env,actor,critic,scene,std,rng):
    env.reset_wide(scene);shield=PotentialEscapeShield();prev_sector=None;prev_action=np.zeros(2,np.float32)
    obsbuf=[];actbuf=[];logbuf=[];rewbuf=[];valbuf=[];minimum=float("inf");outcome="timeout"
    scan=rospy.wait_for_message("/scan",LaserScan,timeout=5);distance,heading=pose(env,scene)
    for step in range(1,STEP_LIMIT+1):
        current=sector_minima(scan.ranges,scan.range_min,scan.range_max)
        rates=np.zeros(36,np.float32) if prev_sector is None else signed_range_rate(prev_sector,current,.2)
        obs=build_observation(current,rates,prev_action,distance,heading);ot=torch.from_numpy(obs)
        with torch.no_grad():
            mean=actor(ot)[0];dist=Normal(mean,torch.full_like(mean,std));raw=dist.sample();logp=dist.log_prob(raw).sum();value=critic(ot.unsqueeze(0))[0]
        requested=decode(raw.numpy());front_ttc=float("inf")
        if prev_sector is not None and abs(requested[1])<.10:
            _,ttc=temporal_features(prev_sector,current,.2);front_ttc=float(min(ttc[17],ttc[18]))
        linear,angular,_=shield.apply(requested[0],requested[1],scan,front_ttc,float("inf"),goal_distance=distance)
        cmd=Twist();cmd.linear.x=linear;cmd.angular.z=angular;env.pub_cmd_vel.publish(cmd)
        next_scan=rospy.wait_for_message("/scan",LaserScan,timeout=5);next_distance,next_heading=pose(env,scene)
        scan_min=float(valid_scan(next_scan).min());minimum=min(minimum,scan_min)
        collision=scan_min<.20;success=next_distance<=.20;timeout=step==STEP_LIMIT and not(collision or success)
        reward=transition_reward(distance,next_distance,next_heading,linear,collision,success,timeout)
        obsbuf.append(obs);actbuf.append(raw.numpy());logbuf.append(float(logp));rewbuf.append(reward);valbuf.append(float(value))
        if collision or success or timeout:
            outcome="collision" if collision else ("success" if success else "timeout");break
        scan=next_scan;distance,heading=next_distance,next_heading;prev_sector=current;prev_action=normalize(linear,angular)
    env.pub_cmd_vel.publish(Twist())
    return {"obs":obsbuf,"acts":actbuf,"logs":logbuf,"rews":rewbuf,"vals":valbuf,
            "outcome":outcome,"steps":step,"minimum_lidar":minimum}

def advantages(rews,vals,gamma=.99,lam=.95):
    adv=np.zeros(len(rews),np.float32);last=0.
    for i in range(len(rews)-1,-1,-1):
        nxt=0. if i==len(rews)-1 else vals[i+1];delta=rews[i]+gamma*nxt-vals[i]
        last=delta+gamma*lam*last;adv[i]=last
    return adv,adv+np.asarray(vals,np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument("--bc-checkpoint",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--steps",type=int,default=10000)
    p.add_argument("--batch-steps",type=int,default=1000);p.add_argument("--update-epochs",type=int,default=8)
    p.add_argument("--smoke",action="store_true");a=p.parse_args()
    if os.path.exists(a.output_dir):raise RuntimeError("refusing overwrite: "+a.output_dir)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);rng=random.Random(a.seed)
    actor=CleanMirrorActor();actor.load_state_dict(torch.load(a.bc_checkpoint,map_location="cpu"));critic=Critic()
    ao=torch.optim.Adam(actor.parameters(),lr=1e-4);co=torch.optim.Adam(critic.parameters(),lr=3e-4);std=.12
    rospy.init_node("clean_start_online_ppo",anonymous=True);env=CleanWideStaticEnv();os.makedirs(a.output_dir)
    total=0;iteration=0;episodes=[]
    try:
        target=min(a.steps,64) if a.smoke else a.steps
        while total<target:
            bo=[];ba=[];bl=[];bad=[];br=[]
            while len(bo)<min(a.batch_steps,target-total):
                level=rng.randint(1,5);scene_seed=rng.randrange(1,2**31);scene=generate(level,1,scene_seed)[0]
                ep=collect_episode(env,actor,critic,scene,std,rng);ad,rt=advantages(ep["rews"],ep["vals"])
                bo+=ep["obs"];ba+=ep["acts"];bl+=ep["logs"];bad+=ad.tolist();br+=rt.tolist();total+=ep["steps"]
                episodes.append({"pillar_count":level,"scene_seed":scene_seed,"outcome":ep["outcome"],"steps":ep["steps"],"minimum_lidar":ep["minimum_lidar"]})
                print("CLEAN_PPO_EP total=%d pillars=%d outcome=%s steps=%d min=%.3f"%(total,level,ep["outcome"],ep["steps"],ep["minimum_lidar"]),flush=True)
                if total>=target or len(bo)>=a.batch_steps:break
            o=torch.tensor(np.asarray(bo),dtype=torch.float32);ac=torch.tensor(np.asarray(ba),dtype=torch.float32)
            old=torch.tensor(bl);ad=torch.tensor(bad);rt=torch.tensor(br);ad=(ad-ad.mean())/(ad.std()+1e-8)
            for _ in range(a.update_epochs):
                mean=actor(o);dist=Normal(mean,torch.full_like(mean,std));log=dist.log_prob(ac).sum(-1);ratio=(log-old).exp()
                aloss=-torch.min(ratio*ad,torch.clamp(ratio,.8,1.2)*ad).mean();ao.zero_grad();aloss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5);ao.step()
                closs=nn.functional.mse_loss(critic(o),rt);co.zero_grad();closs.backward();nn.utils.clip_grad_norm_(critic.parameters(),.5);co.step()
            iteration+=1;torch.save(actor.state_dict(),os.path.join(a.output_dir,"actor_latest.pth"));torch.save(critic.state_dict(),os.path.join(a.output_dir,"critic_latest.pth"))
            print("CLEAN_PPO_ITER %d total=%d actor_loss=%.6f critic_loss=%.6f"%(iteration,total,float(aloss),float(closs)),flush=True)
        summary={"seed":a.seed,"requested_steps":a.steps,"actual_steps":total,"iterations":iteration,"episodes":len(episodes),
                 "counts":{k:sum(e["outcome"]==k for e in episodes) for k in ("success","collision","timeout")},
                 "online_uniform_random_pillars_1_to_5":True,"step_limit":STEP_LIMIT,"bc_initialization":a.bc_checkpoint,
                 "exact_mirror_actor":True,"permanent_ood_accessed":False,"smoke":a.smoke}
        json.dump(episodes,open(os.path.join(a.output_dir,"episodes.json"),"w"),indent=2);json.dump(summary,open(os.path.join(a.output_dir,"summary.json"),"w"),indent=2,sort_keys=True)
        print("CLEAN_PPO_COMPLETE "+json.dumps(summary,sort_keys=True),flush=True)
    finally:env.restore_center();env.pub_cmd_vel.publish(Twist());print("CENTER_RESTORED_ZERO",flush=True)

if __name__=="__main__":main()
