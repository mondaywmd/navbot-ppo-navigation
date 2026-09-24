"""Post-obstacle analysis with thresholds fixed before execution."""
import argparse,csv,json,os
from collections import defaultdict
def load(p):
    with open(p,newline="") as f:return list(csv.DictReader(f))
def counts(r):return{k:sum(x["outcome"]==k for x in r)for k in("success","collision","timeout")}
def fmt(x):return"%d/%d/%d"%(x["success"],x["collision"],x["timeout"])
def metrics(rows):
    p=[r for r in rows if int(r["post_obstacle"])==1]
    if not p:return{"post_steps":0,"mean_abs_goal_angle_deg":None,"final_abs_goal_angle_deg":None,"mean_linear":None,"mean_abs_angular":None,"distance_progress":None,"angular_sign_persistence":None}
    angles=[abs(float(r["goal_relative_angle_deg"]))for r in p];angular=[float(r["cmd_angular"])for r in p];nz=[v for v in angular if abs(v)>.02]
    return{"post_steps":len(p),"mean_abs_goal_angle_deg":sum(angles)/len(angles),"final_abs_goal_angle_deg":angles[-1],"mean_linear":sum(float(r["cmd_linear"])for r in p)/len(p),"mean_abs_angular":sum(abs(v)for v in angular)/len(p),"distance_progress":float(p[0]["goal_distance"])-float(p[-1]["goal_distance"]),"angular_sign_persistence":max(sum(v>0 for v in nz),sum(v<0 for v in nz))/len(nz)if nz else 0.0}
def classify(m):
    if m["post_steps"]<20:return"evidence_insufficient"
    if m["mean_abs_angular"]>=.18 and m["angular_sign_persistence"]>=.8:return"persistent_turning"
    if m["mean_abs_goal_angle_deg"]>=20 and m["final_abs_goal_angle_deg"]>=20 and m["mean_abs_angular"]<.18:return"insufficient_heading_recovery"
    if m["mean_abs_goal_angle_deg"]<=15 and m["mean_linear"]<.08 and m["distance_progress"]<.30:return"forward_too_slow"
    return"evidence_insufficient"
def main():
    p=argparse.ArgumentParser();p.add_argument("--results-dir",required=True);a=p.parse_args();report=os.path.join(a.results_dir,"REPORT.md");js=os.path.join(a.results_dir,"analysis.json")
    if os.path.exists(report)or os.path.exists(js):raise RuntimeError("refusing to overwrite analysis")
    ep=load(os.path.join(a.results_dir,"episodes.csv"));tr=load(os.path.join(a.results_dir,"step_trace.csv"));actor=[r for r in ep if r["controller"]=="v5_actor"];rule=[r for r in ep if r["controller"]=="reactive_template"];g=defaultdict(list)
    for r in tr:g[(r["controller"],r["scene"],int(r["episode"]))].append(r)
    pairs=[]
    for n in range(1,7):
        r=[x for x in actor if int(x["pair"])==n];pairs.append({"pair":n,"focus":r[0]["focus"],"positive":counts([x for x in r if x["mirror_side"]=="positive"]),"negative":counts([x for x in r if x["mirror_side"]=="negative"])})
    timeouts=[]; comparisons=[]; representative=[]
    for e in actor:
        if e["outcome"]=="timeout"and float(e["min_lidar"])>=.2:
            timeout_rows=g[("v5_actor",e["scene"],int(e["episode"]))];m=metrics(timeout_rows);timeouts.append({"scene":e["scene"],"episode":int(e["episode"]),**m,"classification":classify(m)})
            successful=sorted([x for x in actor if x["scene"]==e["scene"]and x["outcome"]=="success"],key=lambda x:int(x["episode"]))
            success_metrics=metrics(g[("v5_actor",e["scene"],int(successful[0]["episode"]))])if successful else None
            comparisons.append({"scene":e["scene"],"timeout_episode":int(e["episode"]),"timeout":m,"classification":classify(m),"success_episode":int(successful[0]["episode"])if successful else None,"success":success_metrics})
            for label,rows in (("timeout",timeout_rows),("success",g[("v5_actor",e["scene"],int(successful[0]["episode"]))]if successful else [])):
                candidates=[r for r in rows if int(r["post_obstacle"])==1]or rows
                for i,r in enumerate(candidates):
                    if i%20==0 or i==len(candidates)-1: representative.append({"case":label,"scene":r["scene"],"episode":r["episode"],"step":r["step"],"robot_x":r["robot_x"],"robot_y":r["robot_y"],"goal_distance":r["goal_distance"],"goal_relative_angle_deg":r["goal_relative_angle_deg"],"min_lidar":r["min_lidar"],"residual_linear":r["residual_linear"],"residual_angular":r["residual_angular"],"cmd_linear":r["cmd_linear"],"cmd_angular":r["cmd_angular"],"gate_active":r["gate_active"],"terminal_reason":r["terminal_reason"]})
    early=[r for r in actor if r["outcome"]=="collision"and int(r["steps"])<=30];rs=sum(r["outcome"]=="success"for r in rule);evidence="positive_feasibility_evidence"if rs else("control_did_not_find_path"if rule else"no_early_collision_to_test")
    hypothesis="At least one early-collision scene is solvable through the unchanged bounded action architecture; test whether V5 selects the wrong early turn direction or residual magnitude."if rs else("The fixed reactive control did not find a path; this is not proof of physical infeasibility. More controller-level feasibility diagnosis is required."if rule else"Cannot decide: no early collision representative was produced.")
    for e in early[:1]:
        rows=g[("v5_actor",e["scene"],int(e["episode"]))]
        for i,r in enumerate(rows):
            if i%5==0 or i==len(rows)-1: representative.append({"case":"early_collision","scene":r["scene"],"episode":r["episode"],"step":r["step"],"robot_x":r["robot_x"],"robot_y":r["robot_y"],"goal_distance":r["goal_distance"],"goal_relative_angle_deg":r["goal_relative_angle_deg"],"min_lidar":r["min_lidar"],"residual_linear":r["residual_linear"],"residual_angular":r["residual_angular"],"cmd_linear":r["cmd_linear"],"cmd_angular":r["cmd_angular"],"gate_active":r["gate_active"],"terminal_reason":r["terminal_reason"]})
    result={"pair_comparison":pairs,"actor_overall":counts(actor),"early_collision_episodes":early,"rule_overall":counts(rule),"architecture_evidence":evidence,"safe_timeout_analysis":timeouts,"success_timeout_comparison":comparisons,"minimal_next_hypothesis":hypothesis}
    with open(js,"w")as f:json.dump(result,f,indent=2,sort_keys=True)
    with open(os.path.join(a.results_dir,"representative_trace.csv"),"w",newline="")as f:
        fields=list(representative[0].keys())if representative else["case"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(representative)
    lines=["# V5 failure-mechanism diagnosis","","Seed `9042601`; 36 deterministic Actor episodes; no training or permanent exam data.","","| Pair | Focus | Positive S/C/T | Negative S/C/T |","|---:|---|---:|---:|"]
    for x in pairs:lines.append("| %02d | %s | %s | %s |"%(x["pair"],x["focus"],fmt(x["positive"]),fmt(x["negative"])))
    lines += ["","Actor overall S/C/T: **%s**."%fmt(result["actor_overall"]),"","## Early-collision feasibility","","Early collisions: **%d**. Reactive control S/C/T: **%s**. Evidence: **%s**."%(len(early),fmt(result["rule_overall"]),evidence),"","## Safe-timeout post-obstacle diagnosis","","| Scene | Ep | Post steps | Mean abs angle | Mean linear | Mean abs angular | Progress | Class |","|---|---:|---:|---:|---:|---:|---:|---|"]
    val=lambda v:"n/a"if v is None else"%.3f"%v
    for x in timeouts:lines.append("| %s | %d | %d | %s | %s | %s | %s | %s |"%(x["scene"],x["episode"],x["post_steps"],val(x["mean_abs_goal_angle_deg"]),val(x["mean_linear"]),val(x["mean_abs_angular"]),val(x["distance_progress"]),x["classification"]))
    if not timeouts:lines.append("| none | - | - | - | - | - | - | evidence_insufficient |")
    lines += ["","### Same-scene successful versus timeout trajectory","","| Scene | Case | Post steps | Mean abs angle | Final abs angle | Mean linear | Mean abs angular | Progress |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in comparisons:
        for label,m in (("timeout ep %d"%x["timeout_episode"],x["timeout"]),("success ep %d"%x["success_episode"],x["success"])):
            if m:lines.append("| %s | %s | %d | %s | %s | %s | %s | %s |"%(x["scene"],label,m["post_steps"],val(m["mean_abs_goal_angle_deg"]),val(m["final_abs_goal_angle_deg"]),val(m["mean_linear"]),val(m["mean_abs_angular"]),val(m["distance_progress"])))
    lines += ["","## Minimal next hypothesis","",hypothesis,"","Diagnostic evidence only; this does not authorize training."]
    with open(report,"w")as f:f.write("\n".join(lines)+"\n")
    print("ANALYSIS_COMPLETE",flush=True)
if __name__=="__main__":main()
