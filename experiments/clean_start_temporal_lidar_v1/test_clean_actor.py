import torch
from clean_actor import CleanMirrorActor
from observation78 import mirror_action,mirror_observation
torch.manual_seed(9051002);actor=CleanMirrorActor();x=torch.randn(1000,78)
with torch.no_grad():error=(actor(mirror_observation(x))-mirror_action(actor(x))).abs().max().item()
assert error<1e-7,error
assert actor(x).shape==(1000,2)
print("CLEAN_ACTOR_TEST_PASS max_mirror_error=%.9g"%error)
