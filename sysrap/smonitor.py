#!/usr/bin/env python
"""
smonitor.py 
============

~/o/sysrap/smonitor.sh ana

"""

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

    delta_usedGpuMemory_GB = np.diff( usedGpuMemory_GB ) 
    w_delta_usedGpuMemory_GB = np.where( delta_usedGpuMemory_GB > 0.001 )[0]

    start = w_delta_usedGpuMemory_GB[2] if len(w_delta_usedGpuMemory_GB) > 2 else None   
    sel = slice(start, None) 

    expr = "np.c_[t[sel], usedGpuMemory_GB[sel]]"
    print(expr)
    print(eval(expr))

    _dmem = "(usedGpuMemory_GB[sel][-1]-usedGpuMemory_GB[sel][0])"
    dmem = eval(_dmem)
    print("dmem %10.3f  %s " % ( dmem, _dmem )) 

    _dt = "(t[sel][-1]-t[sel][0])"
    dt = eval(_dt)
    print("dt   %10.3f  %s " % ( dt, _dt )) 
    print("dmem/dt  %10.3f  " % (dmem/dt))

    
    deg = 1  # linear   
    pfit = np.polyfit(t[sel], usedGpuMemory_GB[sel], deg)
    linefit = np.poly1d(pfit)
    linefit_label = "line fit:  slope %10.3f [GB/s] intercept %10.3f " % (linefit.coef[0], linefit.coef[1])
 
    headline = "smonitor.sh device %(_device)s total_GB %(_total)4.1f pid %(_pid)s " % locals()
    title = "\n".join([headline, linefit_label])
    print(title)

    fig, ax = mp.subplots(figsize=SIZE/100.)
    fig.suptitle(title)

    ax.set_yscale('log') 

    ax.plot( t, total_GB , label="total_GB" )
    ax.plot( t, free_GB , label="free_GB" )
    ax.plot( t, used_GB , label="used_GB" )

    #ax.plot( t, usedGpuMemory_GB , label="proc.usedGpuMemory_GB"  )
    ax.scatter( t, usedGpuMemory_GB, label="proc.usedGpuMemory_GB"  )
    ax.plot( t[sel], linefit(t[sel]), label=linefit_label )


    ax.legend()    
    fig.show()    
          

