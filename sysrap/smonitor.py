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
    usedGpuMemory_GB = usedGpuMemory/1e9 
    total_GB = total/1e9  
    free_GB = free/1e9  
    used_GB = used/1e9

    u_device = np.unique(device)
    u_total = np.unique(total)
    u_pid = np.unique(pid)
    u_proc_count = np.unique(proc_count)

    assert len(u_device) == 1 
    assert len(u_total) == 1    
    assert len(u_pid) == 1    

    _device = u_device[0]
    _total = u_total[0]/1e9
    _pid = u_pid[0]
    _proc_count = u_proc_count[0]

    assert _proc_count == 1 

 
    title = "smonitor.sh device %(_device)s total_GB %(_total)4.1f pid %(_pid)s " % locals()
    fig, ax = mp.subplots(figsize=SIZE/100.)
    fig.suptitle(title)

    ax.set_yscale('log') 

    ax.plot( t, total_GB , label="total_GB" )
    ax.plot( t, free_GB , label="free_GB" )
    ax.plot( t, used_GB , label="used_GB" )
    ax.plot( t, usedGpuMemory_GB , label="proc.usedGpuMemory_GB"  )
    ax.scatter( t, usedGpuMemory_GB )



    ax.legend()    
    fig.show()    
          

