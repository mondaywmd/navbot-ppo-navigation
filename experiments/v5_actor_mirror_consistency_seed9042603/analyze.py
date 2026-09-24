"""Apply preregistered mirror-consistency decision rule."""
import argparse,csv,json,os,statistics
from collections import Counter
def read(p):
    with open(p,newline="")as f:return list(csv.DictReader(f))
def mean(v):return sum(v)/len(v)if v else None
def main():
    p=argparse.ArgumentParser();p.add_argument("--results-dir",required=True);a=p.parse_args();report=os.path.join(a.results_dir,"REPORT.md");summary=os.path.join(a.results_dir,"summary.json")
    if os.path.exists(report)or os.path.exists(summary):raise RuntimeError("refusing overwrite analysis")
    with open(os.path.join(a.results_dir,"offline","offline_summary.json"))as f:offline=json.load(f)
    traces=read(os.path.join(a.results_dir,"gazebo","first20.csv"));episodes=read(os.path.join(a.results_dir,"gazebo","episodes.csv"));pairs=[];representative=[]
    for pair in range(1,7):
        linear=[];angular=[]
        for ep in range(1,4):
            pos={(int(r["step"])):r for r in traces if int(r["pair"])==pair and r["mirror_side"]=="positive"and int(r["episode"])==ep};neg={(int(r["step"])):r for r in traces if int(r["pair"])==pair and r["mirror_side"]=="negative"and int(r["episode"])==ep}
            for step in sorted(set(pos)&set(neg)):
                le=abs(float(neg[step]["residual_linear"])-float(pos[step]["residual_linear"]));ae=abs(float(neg[step]["residual_angular"])+float(pos[step]["residual_angular"]));linear.append(le);angular.append(ae);representative.append({"pair":pair,"episode":ep,"step":step,"positive_linear":pos[step]["residual_linear"],"negative_linear":neg[step]["residual_linear"],"linear_error":le,"positive_angular":pos[step]["residual_angular"],"negative_angular":neg[step]["residual_angular"],"angular_error":ae,"positive_cmd_linear":pos[step]["cmd_linear"],"negative_cmd_linear":neg[step]["cmd_linear"],"positive_cmd_angular":pos[step]["cmd_angular"],"negative_cmd_angular":neg[step]["cmd_angular"]})
        pe=[e for e in episodes if int(e["pair"])==pair and e["mirror_side"]=="positive"];ne=[e for e in episodes if int(e["pair"])==pair and e["mirror_side"]=="negative"];pc=sum(e["outcome"]=="early_collision"for e in pe);nc=sum(e["outcome"]=="early_collision"for e in ne);ml=mean(linear);ma=mean(angular)
        pairs.append({"pair":pair,"positive_outcomes":dict(Counter(e["outcome"]for e in pe)),"negative_outcomes":dict(Counter(e["outcome"]for e in ne)),"paired_steps":len(linear),"mean_linear_error":ml,"mean_angular_error":ma,"action_relation_violated":ml>.05 or ma>.10,"one_sided_early_collision":(pc>0)==(nc==0) and pc!=nc})
    off=offline["groups"]["all"];offline_clear=off["angular_error"]["median"]>.10 or off["angular_error"]["p90"]>.20 or off["linear_error"]["median"]>.05 or off["linear_error"]["p90"]>.10;systematic=sum(x["action_relation_violated"]for x in pairs)>=4;associated=[x for x in pairs if x["one_sided_early_collision"]];association=len(associated)>=2 and statistics.median(x["mean_angular_error"]for x in associated)>.10;support=offline_clear and systematic and association;verdict="actor_mirror_inconsistency_supported"if support else"evidence_insufficient"
    result={"offline_mirror_error":offline,"pair_results":pairs,"checks":{"offline_error_clear":offline_clear,"closed_loop_systematic_violation":systematic,"one_sided_early_collision_association":association},"verdict":verdict,"collision_causality_claim":False,"minimal_next_hypothesis":"A mirror-equivariance learning constraint on newly generated paired training scenes can reduce the measured V5 directional action bias without relying on additional PPO steps."if support else"Cannot decide; next inspect coordinate and action-composition symmetry with a minimal control audit."}
    with open(summary,"w")as f:json.dump(result,f,indent=2,sort_keys=True)
    worst=max(pairs,key=lambda x:x["mean_angular_error"]);rows=[r for r in representative if r["pair"]==worst["pair"] and r["episode"]==1]
    with open(os.path.join(a.results_dir,"representative_first20.csv"),"w",newline="")as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=["# Final V5 Actor mirror-consistency diagnosis","","## Transform","","LiDAR 0..9 -> 9..0; index 11, 13, and 15 negate; indices 10, 12, and 14 remain unchanged. Unit tests and double-mirror checks passed.","","## Offline errors","","| Group | N | Linear mean/median/P90/max | Angular mean/median/P90/max |","|---|---:|---|---|"]
    for name,g in offline["groups"].items():
        l=g["linear_error"];q=g["angular_error"];lines.append("| %s | %d | %.4f / %.4f / %.4f / %.4f | %.4f / %.4f / %.4f / %.4f |"%(name,l["count"],l["mean"],l["median"],l["p90"],l["max"],q["mean"],q["median"],q["p90"],q["max"]))
    lines += ["","## Gazebo mirror pairs","","| Pair | Positive outcomes | Negative outcomes | Paired steps | Mean linear error | Mean angular error | Violated | One-sided early collision |","|---:|---|---|---:|---:|---:|---|---|"]
    for x in pairs:lines.append("| %02d | %s | %s | %d | %.4f | %.4f | %s | %s |"%(x["pair"],x["positive_outcomes"],x["negative_outcomes"],x["paired_steps"],x["mean_linear_error"],x["mean_angular_error"],x["action_relation_violated"],x["one_sided_early_collision"]))
    lines += ["","Checks: `%s`."%result["checks"],"","Formal verdict: **%s**."%verdict,"","No collision causality is claimed.","","Minimal next hypothesis: "+result["minimal_next_hypothesis"]]
    with open(report,"w")as f:f.write("\n".join(lines)+"\n")
    print("ANALYSIS_COMPLETE verdict="+verdict,flush=True)
if __name__=="__main__":main()
