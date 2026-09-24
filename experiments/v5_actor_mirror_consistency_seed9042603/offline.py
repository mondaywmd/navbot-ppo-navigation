"""Offline deterministic Actor equivariance test on non-OOD diagnostic logs."""
import argparse,csv,json,os
import numpy as np,torch
from mirror import mirror_observation,valid_observation
from residual_networks import ResidualActor
SEED=9042603
def read(path):
    with open(path,newline="")as f:return list(csv.DictReader(f))
def stats(v):
    a=np.asarray(v);return{"count":len(a),"mean":float(np.mean(a)),"median":float(np.median(a)),"p90":float(np.percentile(a,90)),"max":float(np.max(a))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--source-one",required=True);p.add_argument("--episodes-one",required=True);p.add_argument("--source-two",required=True);p.add_argument("--episodes-two",required=True);p.add_argument("--results-dir",required=True);a=p.parse_args()
    if os.path.exists(a.results_dir):raise RuntimeError("refusing overwrite: "+a.results_dir)
    os.makedirs(a.results_dir);rng=np.random.RandomState(SEED);all_rows=[]
    for trace_path,episode_path in((a.source_one,a.episodes_one),(a.source_two,a.episodes_two)):
        episodes=read(episode_path);labels={}
        for e in episodes:
            if e.get("controller","v5_actor")!="v5_actor":continue
            raw=e.get("outcome",e.get("classification"));label="early_collision"if raw in("collision","early_collision")and int(e["steps"])<=30 else("success"if raw=="success"else"timeout")
            labels[(e["scene"],int(e["episode"]))]=label
        for row in read(trace_path):
            if row.get("controller","v5_actor")!="v5_actor":continue
            key=(row["scene"],int(row["episode"]));
            if key in labels:all_rows.append((labels[key],row))
    selected=[]
    for label,limit in(("success",200),("early_collision",200),("timeout",200)):
        pool=[x for x in all_rows if x[0]==label];indices=rng.choice(len(pool),min(limit,len(pool)),replace=False)if pool else[];selected += [pool[int(i)]for i in indices]
    if len(selected)<100:raise RuntimeError("fewer than 100 legal observations")
    synthetic=np.asarray([.05,.15,.25,.35,.45,.55,.65,.75,.85,.95,.2,-.3,.4,.5,.866,.25],dtype=np.float32)
    assert valid_observation(synthetic);assert np.array_equal(mirror_observation(mirror_observation(synthetic)),synthetic);assert valid_observation(mirror_observation(synthetic))
    actor=ResidualActor(16,2);actor.load_state_dict(torch.load(a.checkpoint,map_location="cpu"));actor.eval();records=[];double_max=0.
    for index,(label,row) in enumerate(selected):
        obs=np.asarray([float(row["obs_%02d"%i])for i in range(16)],dtype=np.float32)
        if not valid_observation(obs):raise AssertionError("invalid logged observation")
        mirrored=mirror_observation(obs);double_max=max(double_max,float(np.max(np.abs(mirror_observation(mirrored)-obs))))
        with torch.inference_mode():original=actor(obs).squeeze(0).numpy();other=actor(mirrored).squeeze(0).numpy()
        records.append({"sample":index,"source_class":label,"linear_error":abs(float(other[0]-original[0])),"angular_error":abs(float(other[1]+original[1])),"original_linear":float(original[0]),"original_angular":float(original[1]),"mirrored_linear":float(other[0]),"mirrored_angular":float(other[1])})
    groups={}
    for label in("all","success","early_collision","timeout"):
        rows=records if label=="all"else[r for r in records if r["source_class"]==label]
        if rows:groups[label]={"linear_error":stats([r["linear_error"]for r in rows]),"angular_error":stats([r["angular_error"]for r in rows])}
    result={"seed":SEED,"unit_tests":{"shape":True,"finite_and_range":True,"double_mirror_max_error":double_max,"passed":double_max==0.},"sample_count":len(records),"groups":groups}
    with open(os.path.join(a.results_dir,"offline_summary.json"),"w")as f:json.dump(result,f,indent=2,sort_keys=True)
    with open(os.path.join(a.results_dir,"offline_samples.csv"),"w",newline="")as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print("OFFLINE_COMPLETE samples=%d angular_median=%.6f angular_p90=%.6f"%(len(records),groups["all"]["angular_error"]["median"],groups["all"]["angular_error"]["p90"]),flush=True)
if __name__=="__main__":main()
