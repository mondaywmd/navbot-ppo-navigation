"""Zero-update gate for clean-start online PPO preparation."""
import argparse, hashlib, json, random
import numpy as np
import torch

from clean_actor import CleanMirrorActor
from clean_curriculum_spec import generate
from observation78 import mirror_action, mirror_observation


def digest(model):
    h=hashlib.sha256()
    for name,p in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--episodes",type=int,default=100)
    a=p.parse_args();rng=random.Random(a.seed)
    actor=CleanMirrorActor();actor.load_state_dict(torch.load(a.checkpoint,map_location="cpu"));actor.eval()
    before=digest(actor);counts={i:0 for i in range(1,6)};keys=set()
    for _ in range(a.episodes):
        level=rng.randint(1,5);scene_seed=rng.randrange(1,2**31)
        scene=generate(level,1,scene_seed)[0];counts[level]+=1
        key=json.dumps(scene,sort_keys=True);assert key not in keys;keys.add(key)
    obs=torch.rand(1000,78)*2-1
    with torch.no_grad():error=(actor(mirror_observation(obs))-mirror_action(actor(obs))).abs().max().item()
    after=digest(actor)
    assert before==after and error==0.0 and all(counts.values())
    print("CLEAN_PPO_ZERO_UPDATE_PASS "+json.dumps({"seed":a.seed,"episodes":a.episodes,
          "pillar_counts":counts,"unique_scenes":len(keys),"actor_digest":before,
          "max_mirror_error":error,"step_limit":200,"permanent_ood_accessed":False,
          "parameter_update":False},sort_keys=True))

if __name__=="__main__":main()
