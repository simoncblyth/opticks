#!/usr/bin/env python


import os, numpy as np, logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 
from opticks.ana.main import opticks_main 
from opticks.ana.key import keydir

if __name__ == '__main__':

    ok = opticks_main()
    kd = keydir(os.environ["OPTICKS_KEY"])
 
    rindex = np.load(os.path.join(kd, "GScintillatorLib/LS_ori/RINDEX.npy"))
    rindex[:,0] *= 1e6  

    wave = False
    if wave: 
        rindex[:,0] = 1240./rindex[:,0]
        rindex = rindex[::-1]       
    pass

    plt.ion()
    fig, ax = plt.subplots(figsize=ok.figsize); 

    ax.plot( rindex[:,0], rindex[:,1], drawstyle="steps" )

    xlim = ax.get_xlim()

    ax.plot( xlim, [1.5,1.5], linestyle="dotted", color="r" )


    #for i in range(len(rindex)):
    #    ax.plot( [rindex[i,0], rindex[i,0]], ylim , linestyle="dotted", color="b" )
    #pass

    #axr = ax.twinx() 
    #axr.set_ylabel("rindex")
    #axr.spines['right'].set_position(('outward', 0))

    fig.show()


