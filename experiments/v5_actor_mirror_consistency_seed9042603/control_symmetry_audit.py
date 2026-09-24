"""Closed-form symmetry audit of unchanged control composition."""
import argparse,json,os
import numpy as np
from control_gate import compose_gated_command
from residual_action import goal_seeking_command
SEED=9042603
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
    if os.path.exists(a.output):raise RuntimeError("refusing overwrite")
    rng=np.random.RandomState(SEED);errors=[]
    for _ in range(1000):
        distance=float(rng.uniform(.2,3.));heading=float(rng.uniform(-np.pi,np.pi));residual=np.asarray([rng.uniform(-1,1),rng.uniform(-1,1)],dtype=np.float32);mirrored=np.asarray([residual[0],-residual[1]],dtype=np.float32)
        nominal=goal_seeking_command(distance,heading);nominal_m=goal_seeking_command(distance,-heading)
        errors.append({"nominal_linear":abs(float(nominal_m[0]-nominal[0])),"nominal_angular":abs(float(nominal_m[1]+nominal[1]))})
        for gate in(False,True):
            command=compose_gated_command(distance,heading,residual,gate);command_m=compose_gated_command(distance,-heading,mirrored,gate)
            errors[-1]["gate_%d_linear"%gate]=abs(float(command_m[0]-command[0]));errors[-1]["gate_%d_angular"%gate]=abs(float(command_m[1]+command[1]))
    maxima={key:max(row[key]for row in errors)for key in errors[0]};passed=all(value<=1e-6 for value in maxima.values());result={"seed":SEED,"samples":1000,"tolerance":1e-6,"maximum_errors":maxima,"passed":passed,"interpretation":"unchanged nominal and gated residual composition are mirror symmetric"if passed else"control composition symmetry violation detected"}
    with open(a.output,"w")as f:json.dump(result,f,indent=2,sort_keys=True)
    print("CONTROL_SYMMETRY_AUDIT passed=%s max=%.9f"%(passed,max(maxima.values())),flush=True)
if __name__=="__main__":main()
