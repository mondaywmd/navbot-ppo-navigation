"""Aggregate completed clean-start acceptance runs without touching OOD data."""
import argparse,csv,json,os,statistics

def main():
 p=argparse.ArgumentParser();p.add_argument("--inputs",nargs="+",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 if os.path.exists(a.output):raise RuntimeError("refusing overwrite: "+a.output)
 rows=[]
 for folder in a.inputs:
  with open(os.path.join(folder,"episodes.csv")) as f:rows.extend(list(csv.DictReader(f)))
 groups={}
 for r in rows:
  level=int(r["level"]);g=groups.setdefault(str(level),[]);g.append(r)
 def stats(items):
  steps=[int(x["steps"]) for x in items];clear=[float(x["minimum_lidar"]) for x in items]
  return {"episodes":len(items),"success":sum(x["outcome"]=="success" for x in items),
   "collision":sum(x["outcome"]=="collision" for x in items),"timeout":sum("timeout" in x["outcome"] for x in items),
   "steps_mean":statistics.mean(steps),"steps_median":statistics.median(steps),"steps_max":max(steps),
   "minimum_lidar_min":min(clear),"minimum_lidar_mean":statistics.mean(clear)}
 result={"overall":stats(rows),"by_level":{k:stats(v) for k,v in sorted(groups.items())},
  "training":False,"permanent_ood_accessed":False,"source_dirs":a.inputs}
 os.makedirs(os.path.dirname(a.output),exist_ok=True)
 with open(a.output,"w") as f:json.dump(result,f,indent=2,sort_keys=True)
 print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
