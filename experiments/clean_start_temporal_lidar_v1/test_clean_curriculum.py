import math
from clean_curriculum_spec import LEVELS,generate,has_clearance_route,legal,mirror
for level in range(1,6):
 for scene in generate(level,2,seed=9050000+level):
  mirrored=mirror(scene)
  assert len(scene["obstacles"])==LEVELS[level]["obstacles"]
  assert legal(scene) and legal(mirrored) and has_clearance_route(scene) and has_clearance_route(mirrored)
  assert mirrored["target_y"]==-scene["target_y"]
  for a,b in zip(scene["obstacles"],mirrored["obstacles"]):
   assert a["x"]==b["x"] and a["y"]==-b["y"] and a["radius"]==b["radius"]
   assert math.hypot(a["x"]-scene["target_x"],a["y"]-scene["target_y"])-a["radius"]-0.5>=0.05-1e-9
print("CLEAN_CURRICULUM_TEST_PASS levels=1..5")
