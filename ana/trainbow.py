#!/usr/bin/env python
"""
trainbow.py : Rainbow deviation angle comparison between Opticks and Geant4 
====================================================================================

To simulate the rainbow events::

   trainbow-

   trainbow-- --spol
   trainbow-- --ppol

   trainbow-- --spol --tcfg4
   trainbow-- --ppol --tcfg4

 

Expected output, is a scattering angle plot and history comparison table:

.. code-block:: py

    In [28]: run rainbow_cfg4.py
    WARNING:opticks.ana.evt:init_index S-Pol G4 finds too few (ps)phosel uniques : 1
    WARNING:opticks.ana.evt:init_index S-Pol G4 finds too few (rs)recsel uniques : 1
    WARNING:opticks.ana.evt:init_index S-Pol G4 finds too few (rsr)reshaped-recsel uniques : 1
                           5:rainbow   -5:rainbow           c2 
                    8ccd        819160       819654             0.15  [4 ] TO BT BT SA
                     8bd        102089       101615             1.10  [3 ] TO BR SA
                   8cbcd         61869        61890             0.00  [5 ] TO BT BR BT SA
                  8cbbcd          9617         9577             0.08  [6 ] TO BT BR BR BT SA
                 8cbbbcd          2604         2687             1.30  [7 ] TO BT BR BR BR BT SA
                8cbbbbcd          1056         1030             0.32  [8 ] TO BT BR BR BR BR BT SA
                   86ccd          1014         1000             0.10  [5 ] TO BT BT SC SA
               8cbbbbbcd           472          516             1.96  [9 ] TO BT BR BR BR BR BR BT SA
                     86d           498          473             0.64  [3 ] TO SC SA
              bbbbbbbbcd           304          294             0.17  [10] TO BT BR BR BR BR BR BR BR BR
              8cbbbbbbcd           272          247             1.20  [10] TO BT BR BR BR BR BR BR BT SA
              cbbbbbbbcd           183          161             1.41  [10] TO BT BR BR BR BR BR BR BR BT
                     4cd           161          139             1.61  [3 ] TO BT AB
                    86bd           138          142             0.06  [4 ] TO BR SC SA
                   8c6cd           126          106             1.72  [5 ] TO BT SC BT SA
                    4ccd           100          117             1.33  [4 ] TO BT BT AB
                  86cbcd            88          110             2.44  [6 ] TO BT BR BT SC SA
                      4d            51           54             0.09  [2 ] TO AB
                   8cc6d            38           40             0.05  [5 ] TO SC BT BT SA
                 8cc6ccd            19           33             3.77  [7 ] TO BT BT SC BT BT SA
                             1000000      1000000         1.09 



* see also: :doc:`rainbow_cfg4_notes`



"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle


from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt, costheta_, cross_
from opticks.ana.boundary import Boundary   
from opticks.ana.droplet import Droplet
from opticks.ana.nbase import chi2

X,Y,Z,W = 0,1,2,3

deg = np.pi/180.
n2ref = 1.33257


def a_scatter_plot_cf(ax, a_evt, b_evt, log_=False):
    """
    :param ax: mpl axis
    :param a_evt:  
    :param b_evt:  
    :param log_: log scale 

    :return: cnt, bns dicts keyed with 0,1

    Scattering angle in degrees 0 to 360  

    Histograms result of a_deviation_angle for each evt, 
    storing counts, bins and patches in dicts 
    with keys 0, 1 

    """ 
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
    suptitle = "Rainbow cfg4 " + label + "[" + tim_a + "] [" + tim_b + "]"  
    log.info("plotting %s " % suptitle)
    fig.suptitle(suptitle  )

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

        c2, c2n, c2nn = chi2(a, b, cut=30)
        c2p = c2.sum()/c2n
        
        plt.plot( bns[:-1], c2, drawstyle='steps', label="chi2/ndf %4.2f" % c2p )
        ax.set_xlim(xlim) 
        ax.legend()

        ax.set_ylim(ylim2) 

        droplet.bow_angle_rectangles()



if __name__ == '__main__':
    args = opticks_main(tag="5",src="torch",det="rainbow",doc=__doc__)


    boundary = Boundary("Vacuum///MainH2OHale")
    droplet = Droplet(boundary)

    plt.ion()
    plt.close()

    tag = args.tag
    src = args.src
    det = args.det
    rec = True
    log_ = True
    not_ = False

    if det == "rainbow":
       if tag == "5":
           label = "S-Pol"
       elif tag == "6":
           label = "P-Pol"
       else:
           label = "no label"


    seqs = []

    try:
        a = Evt(tag=tag, src=src, det=det, label="%s Op" % label, seqs=seqs, not_=not_, rec=rec, args=args)
        b = Evt(tag="-%s" % tag, src=src, det=det, label="%s G4" % label, seqs=seqs, not_=not_, rec=rec, args=args)
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)    

    print a.brief
    print b.brief

    if not (a.valid and b.valid):
        log.fatal("need two valid events to compare ")
        sys.exit(1)


    lmx = 20  
    #hcf = a.history.table.compare(b.history.table)
    hcf = a.his.compare(b.his)
    if len(hcf.lines) > lmx:
        hcf.sli = slice(0,lmx)
    print hcf 

    cf_plot(a, b, label=label, log_=log_, ylim=[0.8,4e4],ylim2=None)


## EXERCISE : ENABLE THE BELOW PLOTS AND INTERPRET WHAT YOU GET 

if 0:
    #seqs = a.history.table.labels[:5] 
    seqs = a.his.labels[:5] 
    for seq in seqs:
        qa =  Evt(tag=tag, src=src, det=det, label="%s Op" % label, seqs=[seq], not_=not_, rec=rec, args=args)
        qb =  Evt(tag="-%s" % tag, src=src, det=det, label="%s G4" % label, seqs=[seq], not_=not_, rec=rec, args=args)

        cf_plot(qa, qb, label=label + " " + seq, log_=log_, ylim=[0.8,4e4],ylim2=None)



