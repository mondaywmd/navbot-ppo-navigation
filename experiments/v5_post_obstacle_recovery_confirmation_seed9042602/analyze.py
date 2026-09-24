"""Predeclared association test; reports support or evidence insufficient."""
import argparse,csv,json,os,statistics
from collections import Counter,defaultdict
def load(p):
    with open(p,newline="")as f:return list(csv.DictReader(f))
def sgn(v):return 1 if v>.001 else(-1 if v<-.001 else 0)
def window_metrics(rows):
    start040=next((i for i,r in enumerate(rows)if float(r["goal_distance"])<.40),len(rows));start=max(start040,max(0,len(rows)-100));w=rows[start:]
    gates=[int(r["gate_active"])for r in w];cmd=[float(r["cmd_angular"])for r in w];res=[float(r["residual_angular"])for r in w];dist=[float(r["goal_distance"])for r in w];deltas=[b-a for a,b in zip(dist,dist[1:])];nz=[sgn(x)for x in deltas if sgn(x)]
    return{"window_steps":len(w),"gate_switches":sum(a!=b for a,b in zip(gates,gates[1:])),"gate_active_fraction":sum(gates)/len(gates),"mean_abs_angular_residual":sum(abs(x)for x in res)/len(res),"cmd_angular_sign_switches":sum(sgn(a)and sgn(b)and sgn(a)!=sgn(b)for a,b in zip(cmd,cmd[1:])),"distance_monotonic_decrease_fraction":sum(x<=.001 for x in deltas)/len(deltas)if deltas else 1.,"distance_trend_reversals":sum(a!=b for a,b in zip(nz,nz[1:])),"minimum_distance":min(dist),"final_distance":dist[-1],"final_minus_minimum_distance":dist[-1]-min(dist)}
def median(rows,key):return statistics.median(r[key]for r in rows)if rows else None
def main():
    p=argparse.ArgumentParser();p.add_argument("--results-dir",required=True);a=p.parse_args();jp=os.path.join(a.results_dir,"analysis.json");rp=os.path.join(a.results_dir,"REPORT.md")
    if os.path.exists(jp)or os.path.exists(rp):raise RuntimeError("refusing overwrite analysis")
    ep=load(os.path.join(a.results_dir,"episodes.csv"));trace=load(os.path.join(a.results_dir,"step_trace.csv"));groups=defaultdict(list)
    for r in trace:groups[(r["scene"],int(r["episode"]))].append(r)
    metrics=[]
    for e in ep:
        if e["classification"]in("success","near_goal_safe_timeout"):
            metrics.append({"scene":e["scene"],"episode":int(e["episode"]),"classification":e["classification"],**window_metrics(groups[(e["scene"],int(e["episode"]))])})
    success=[r for r in metrics if r["classification"]=="success"];timeouts=[r for r in metrics if r["classification"]=="near_goal_safe_timeout"]
    success_gate=median(success,"gate_switches");success_turn=median(success,"cmd_angular_sign_switches")
    gate_higher=bool(timeouts)and median(timeouts,"gate_switches")>success_gate and sum(r["gate_switches"]>success_gate for r in timeouts)/len(timeouts)>=.75
    turn_higher=bool(timeouts)and median(timeouts,"cmd_angular_sign_switches")>success_turn and sum(r["cmd_angular_sign_switches"]>success_turn for r in timeouts)/len(timeouts)>=.75
    nonconvergent=bool(timeouts)and sum(r["distance_trend_reversals"]>=4 and r["final_minus_minimum_distance"]>=.02 for r in timeouts)/len(timeouts)>=2/3
    supported=len(timeouts)>=3 and gate_higher and turn_higher and nonconvergent;verdict="oscillation_hypothesis_supported"if supported else"evidence_insufficient"
    result={"classification_counts":dict(Counter(r["classification"]for r in ep)),"comparison_window_metrics":metrics,"group_medians":{"success":{k:median(success,k)for k in("gate_switches","gate_active_fraction","mean_abs_angular_residual","cmd_angular_sign_switches","distance_monotonic_decrease_fraction","distance_trend_reversals","final_minus_minimum_distance")},"near_goal_safe_timeout":{k:median(timeouts,k)for k in("gate_switches","gate_active_fraction","mean_abs_angular_residual","cmd_angular_sign_switches","distance_monotonic_decrease_fraction","distance_trend_reversals","final_minus_minimum_distance")}},"support_checks":{"near_goal_timeout_count_at_least_3":len(timeouts)>=3,"gate_switches_systematically_higher":gate_higher,"angular_reversals_systematically_higher":turn_higher,"repeated_nonconvergence":nonconvergent},"verdict":verdict,"causality_claim":False,"minimal_control_ablation_worth_designing":supported}
    with open(jp,"w")as f:json.dump(result,f,indent=2,sort_keys=True)
    def val(v):return"n/a"if v is None else"%.3f"%v
    lines=["# V5 post-obstacle recovery confirmation"," Context","Seed `9042602`; 12 new scenes; 36 deterministic episodes; no training or permanent OOD access.","","## Classification","",str(result["classification_counts"]),"","## Final-window comparison","","| Group | N | Gate switches | Gate active | Mean abs angular residual | Angular sign switches | Monotonic-distance fraction | Trend reversals | Final-min rebound |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,rows in(("success",success),("near-goal safe timeout",timeouts)):
        m=result["group_medians"]["success"if name=="success"else"near_goal_safe_timeout"];lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |"%(name,len(rows),val(m["gate_switches"]),val(m["gate_active_fraction"]),val(m["mean_abs_angular_residual"]),val(m["cmd_angular_sign_switches"]),val(m["distance_monotonic_decrease_fraction"]),val(m["distance_trend_reversals"]),val(m["final_minus_minimum_distance"])))
    lines += ["","## Predeclared support checks","",str(result["support_checks"]),"","Verdict: **%s**."%verdict,"","This tests association only and makes no causal claim.","","Worth designing a separate minimal control ablation next: **%s**."%("yes"if supported else"no; evidence is insufficient")] 
    with open(rp,"w")as f:f.write("\n".join(lines)+"\n")
    print("ANALYSIS_COMPLETE verdict="+verdict,flush=True)
if __name__=="__main__":main()
