#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as mp
SIZE=np.array([1280, 720])

if __name__ == '__main__':
    mon = np.load("smonitor.npy")

    stamp = mon[:,0]
    device = mon[:,1]
    free = mon[:,2]
    total = mon[:,3]
    used = mon[:,4]
    pid = mon[:,5]
    usedGpuMemory = mon[:,6]
    proc_count = mon[:,7] 

    t = (stamp - stamp[0])/1e6 
    GB = usedGpuMemory/1e9 

    title = "smonitor.sh"
    fig, ax = mp.subplots(figsize=SIZE/100.)
    fig.suptitle(title)

    label = "usedGpuMemory[GB] vs t[s]"
    ax.plot( t, GB , label=label )
    ax.scatter( t, GB , label=None )

    ax.legend()    
    fig.show()    
          


     
     




