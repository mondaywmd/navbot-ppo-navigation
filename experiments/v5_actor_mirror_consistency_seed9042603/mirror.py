"""Mirror transform derived from the existing 16-D observation construction."""
import numpy as np
MIRROR_INDEX=np.asarray([9,8,7,6,5,4,3,2,1,0,10,11,12,13,14,15])
NEGATE=np.asarray([1,1,1,1,1,1,1,1,1,1,1,-1,1,-1,1,-1],dtype=np.float32)
def mirror_observation(observation):
    value=np.asarray(observation,dtype=np.float32)
    if value.shape!=(16,)or not np.all(np.isfinite(value)):raise ValueError("expected finite shape-(16,) observation")
    return value[MIRROR_INDEX]*NEGATE
def valid_observation(value):
    x=np.asarray(value,dtype=np.float32)
    return x.shape==(16,)and np.all(np.isfinite(x))and np.all((-1e-6<=x[:10])&(x[:10]<=1.000001))and -1.000001<=x[10]<=1.000001 and -1.000001<=x[11]<=1.000001 and 0<=x[12]<=1.5 and np.all(np.abs(x[13:])<=1.000001)
