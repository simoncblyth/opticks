#!/usr/bin/env python
"""
Seconds for rainbow 1M 
-----------------------

CAVEAT: using laptop GPU with only 384 cores, desktop GPUs expected x20-30 faster

Disabling step-by-step recording has large improvement
factor for Opticks of about x3 but not much impact on cfg4-.
The result match between G4 and Op remains unchanged.

Seems like reducing the number and size of 
buffers in context is a big win for Opticks.

With step by step and sequence recording::

   Op   4.6    5.8      # Opticks timings rather fickle to slight code changes, maybe stack 
   G4   56.8  55.9

Just final photon recording:: 

   Op    1.8 
   G4   47.9


Matching curand buffer to requirement
---------------------------------------

* tried using 1M cuRAND buffer matching the requirement rather than using default 3M all the time,
  saw no change in propagation time 

::

    # change ggeoview-rng-max value down to 1M

    ggeoview-rng-prep  # create the states cache 
 
    #  opticks-/OpticksCfg.hh accordingly 


Compute Mode, ie no OpenGL
-----------------------------

Revived "--compute" mode of ggv binary which uses OptiX owned buffers
as opposed to the usual interop approach of using OpenGL buffers.
Both with and without step recording is giving similar times in 
compute mode. This is very different from interop mode where 
cutting down on buffers gives big wins.

::

    Op   0.75  0.65 
    G4   57.   56.

A related cmp mode controlled by "--cmp" option uses different computeTest binary, 
is not operational and little motivation now that "--compute" mode works.
Could create package without OpenGL dependencies if there is a need.

::

   ggv-;ggv-rainbow --compute 
   ggv-;ggv-rainbow --compute --nostep 
   ggv-;ggv-rainbow --compute --nostep --dbg


* look at how time scales with photon count  


Split the prelaunch from launch timings
-----------------------------------------

Kernel validation, compilation and prelaunch does not 
need to be done for each event so can exclude it from 
timings. 

Doing this get::

    Op (interop mode)         1.509 
    Op (--compute)            0.290
    Op (--compute --nostep)   0.294     # skipping step recording not advantageous   
    Op (--compute)            0.1416    # hmm some performance instability

In ">console" login mode "ggv-rainbow" gives error that no GPU available

Immediately after login getting::

    Op (--compute)            0.148


Testing in Console Mode
-------------------------

::

    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_170136/t_delta.ini:propagate=0.14798854396212846

    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171121/t_delta.ini:propagate=0.44531063502654433  # try >console mode 
    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171142/t_delta.ini:propagate=0.45501201006118208
    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171156/t_delta.ini:propagate=0.33855076995678246 
    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171213/t_delta.ini:propagate=0.46851423906628042
    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171226/t_delta.ini:propagate=0.33861030195839703

    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171527/t_delta.ini:propagate=1.5933509200112894   # GUI interop mode
    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171548/t_delta.ini:propagate=0.27229616406839341  # GUI --compute mode

Immediately after switching back to automatic graphics switching, then shortly after that::

    0.142      
    0.293



To do the nostep check
------------------------

After standard comparison::

   ggv-;ggv-rainbow 
   ggv-;ggv-rainbow --cfg4 

* recompile optixrap- without RECORD define 
* run with --nostep option::

   ggv-;ggv-rainbow --nostep 
   ggv-;ggv-rainbow --cfg4 --nostep 



"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

from env.numerics.npy.evt import Evt, costheta_, cross_
from env.numerics.npy.history import History, AbbFlags
from env.numerics.npy.geometry import Boundary   
from env.numerics.npy.droplet import Droplet
from env.numerics.npy.fresnel import fresnel_factor
from env.numerics.npy.nbase import chi2

X,Y,Z,W = 0,1,2,3

deg = np.pi/180.
n2ref = 1.33257


def a_scatter_plot_cf(ax, a_evt, b_evt, log_=False):
    db = np.arange(0,360,1)

    incident = np.array([0,0,-1])
    cnt = {}
    bns = {}
    ptc = {}
    j = -1
    for i,evt in enumerate([a_evt, b_evt]):
        dv = evt.a_deviation_angle(axis=X, incident=incident)/deg
        ax.set_xlim(0,360)
        if len(dv) > 0:
            cnt[i], bns[i], ptc[i] = ax.hist(dv, bins=db,  log=log_, histtype='step', label=evt.label)
            j = i 
    pass
    if len(bns) == 2:
        assert np.all( bns[0] == bns[1] )

    if j == -1:
        bns = None
    else:
        bns = bns[j] 

    return cnt, bns


def cf_plot(evt_a, evt_b, label="", log_=False, ylim=[1,1e5], ylim2=[0,10], sli=slice(0,10)):

    tim_a = " ".join(map(lambda f:"%5.2f" % f, map(float, filter(None, evt_a.tdii['propagate']) )[sli]))
    tim_b = " ".join(map(lambda f:"%5.2f" % f, map(float, filter(None, evt_b.tdii['propagate']) )[sli]))

    fig = plt.figure()
    fig.suptitle("Rainbow cfg4 " + label + "[" + tim_a + "] [" + tim_b + "]"  )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    ax = fig.add_subplot(gs[0])

    c, bns = a_scatter_plot_cf(ax, evt_a, evt_b, log_=log_)
    droplet.bow_angle_rectangles()
    ax.set_ylim(ylim)
    ax.legend()

    xlim = ax.get_xlim()

    ax = fig.add_subplot(gs[1])

    if len(c) == 2:
        a,b = c[0],c[1]

        c2, c2n = chi2(a, b, cut=30)
        c2p = c2.sum()/c2n
        
        plt.plot( bns[:-1], c2, drawstyle='steps', label="chi2/ndf %4.2f" % c2p )
        ax.set_xlim(xlim) 
        ax.legend()

        ax.set_ylim(ylim2) 

        droplet.bow_angle_rectangles()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    boundary = Boundary("Vacuum///MainH2OHale")
    droplet = Droplet(boundary)
    af = AbbFlags()

    plt.ion()
    plt.close()

    rec = False
    tag = "6"
    src = "torch"
    det = "rainbow"
    log_ = True
    not_ = False

    if det == "rainbow":
       if tag == "5":
           label = "S-Pol"
       elif tag == "6":
           label = "P-Pol"
       else:
           label = "no label"


    if rec:
        seqs = Droplet.seqhis([0,1,2,3,4,5,6,7],src="TO")

        his_a = History.for_evt(tag="%s" % tag, src=src, det=det)
        his_b = History.for_evt(tag="-%s" % tag, src=src, det=det)

        cf = his_a.table.compare(his_b.table)
        print cf

        sa = set(his_a.table.labels)
        sb = set(his_b.table.labels)
        sc = sorted(list(sa & sb), key=lambda _:his_a.table.label2count.get(_, None)) 

        print "Opticks but not G4, his_a.table(sa-sb)\n", his_a.table(sa - sb)
        print "G4 but not Opticks, his_b.table(sb-sa)\n", his_b.table(sb - sa)
        sq = [None]

    else:
        sq = [None]

    for seq in sq:

        seqs = [] if seq is None else [seq]
        evt_a =  Evt(tag=tag, src=src, det=det, label="%s Op" % label, seqs=seqs, not_=not_, rec=rec)
        evt_b =  Evt(tag="-%s" % tag, src=src, det=det, label="%s G4" % label, seqs=seqs, not_=not_, rec=rec)

        #sli = slice(0,15)
        #sli = slice(None)
        #evt_a.history_table(sli)
        #evt_b.history_table(sli)


    if 1:
        cf_plot(evt_a, evt_b, label=label, log_=log_, ylim=[0.8,4e4],ylim2=None)



