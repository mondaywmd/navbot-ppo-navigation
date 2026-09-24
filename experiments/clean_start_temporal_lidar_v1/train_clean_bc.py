"""Pair-held-out BC for the clean 78-D exactly symmetric Actor."""
import argparse,hashlib,json,os
import numpy as np
import torch
from torch import nn
from clean_actor import CleanMirrorActor

SEED=9051002
def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--epochs",type=int,default=400);p.add_argument("--smoke-only",action="store_true");a=p.parse_args()
 if not a.smoke_only and os.path.exists(a.output_dir):raise RuntimeError("refusing overwrite: "+a.output_dir)
 torch.manual_seed(SEED);np.random.seed(SEED);d=np.load(a.dataset)
 levels=d["levels"];pairs=d["pair_ids"];validation_pairs=[]
 for level in sorted(set(levels.tolist())):validation_pairs.append(int(sorted(set(pairs[levels==level].tolist()))[-1]))
 val=np.isin(pairs,validation_pairs);train=~val
 x=torch.from_numpy(d["observations"]);y=torch.from_numpy(d["actions"]);tm=torch.from_numpy(train);vm=torch.from_numpy(val)
 actor=CleanMirrorActor();initial=digest(actor);initial_val=float(nn.functional.mse_loss(actor(x[vm]),y[vm]))
 loss=nn.functional.mse_loss(actor(x[tm]),y[tm]);loss.backward()
 assert digest(actor)==initial
 if a.smoke_only:
  print("CLEAN_BC_ZERO_UPDATE_PASS train=%d val=%d loss=%.6f digest=%s"%(int(tm.sum()),int(vm.sum()),float(loss),initial));return
 os.makedirs(a.output_dir);opt=torch.optim.Adam(actor.parameters(),lr=1e-3);best=float("inf");best_epoch=0;state=None;history=[]
 for epoch in range(1,a.epochs+1):
  actor.train();pred=actor(x[tm]);tl=nn.functional.mse_loss(pred,y[tm]);opt.zero_grad();tl.backward();opt.step()
  actor.eval()
  with torch.no_grad():vl=nn.functional.mse_loss(actor(x[vm]),y[vm])
  if float(vl)<best:best=float(vl);best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
  if epoch==1 or epoch%40==0:
   row={"epoch":epoch,"train_mse":float(tl),"validation_mse":float(vl)};history.append(row);print("CLEAN_BC "+json.dumps(row),flush=True)
 actor.load_state_dict(state);checkpoint=os.path.join(a.output_dir,"actor_bc_best.pth");torch.save(actor.state_dict(),checkpoint)
 summary={"seed":SEED,"initialization":"random","epochs":a.epochs,"best_epoch":best_epoch,"initial_validation_mse":initial_val,"best_validation_mse":best,"train_samples":int(tm.sum()),"validation_samples":int(vm.sum()),"validation_pairs":validation_pairs,"exact_mirror_architecture":True,"old_weights_used":False,"ppo_started":False,"permanent_ood_accessed":False,"checkpoint":checkpoint,"actor_digest":digest(actor),"history":history}
 with open(os.path.join(a.output_dir,"summary.json"),"w") as f:json.dump(summary,f,indent=2,sort_keys=True)
 print("CLEAN_BC_COMPLETE "+json.dumps(summary,sort_keys=True),flush=True)
if __name__=="__main__":main()
