import numpy as np
import torch
from observation78 import build_observation, mirror_action, mirror_observation
from temporal_lidar import signed_range_rate

previous=np.linspace(0.5,3.0,36,dtype=np.float32);current=previous.copy()
current[2]-=0.10;current[30]+=0.10
rates=signed_range_rate(previous,current,0.2)
assert rates[2]>0 and rates[30]<0
observation=build_observation(current,rates,[0.2,-0.4],2.3,0.7)
mirrored=mirror_observation(observation)
assert observation.shape==mirrored.shape==(78,)
assert np.allclose(mirror_observation(mirrored),observation)
batch=torch.from_numpy(np.stack([observation]*8))
assert torch.equal(mirror_observation(mirror_observation(batch)),batch)
action=torch.tensor([[0.2,-0.4]])
assert torch.allclose(mirror_action(action),torch.tensor([[0.2,0.4]]))
print("OBSERVATION78_TEST_PASS approaching=%.3f receding=%.3f"%(rates[2],rates[30]))
