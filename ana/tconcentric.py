#!/usr/bin/env python
"""
tconcentric.py 
=============================================

Loads test events from Opticks and Geant4 and 
created by OKG4Test and 
compares their bounce histories.

"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nbase import vnorm, costheta_
from opticks.ana.cf   import CF

STEP = 4

X,Y,Z=0,1,2


def butterfly_yz(plt, a, b, pt):
    plt.subplot(1, 2, 1)
    plt.scatter(a[:,pt,Y],a[:,pt,Z])

    plt.subplot(1, 2, 2)
    plt.scatter(b[:,pt,Y],b[:,pt,Z])


def butterfly(plt, scf):
    """
    Expecting 8cc6ccd 

              TO BT BT SC BT BT AB
              p0 p1 p2 p3 p4 p5 p6
                       
    p3: scatter occurs at point on X axis
    p4: first intersection point after the scatter  
    """
    a,b = scf.rpost()
    butterfly_yz(plt, a, b, pt=4)


def isolated_scatter():
    """
    Without selection scatter distrib plots 
    """ 
    OLDMOM,OLDPOL,NEWMOM,NEWPOL = 0,1,2,3
    aa = np.load(os.path.expandvars("$TMP/RayleighTest/ok.npy"))
    bb = np.load(os.path.expandvars("$TMP/RayleighTest/cfg4.npy"))

    bins = 100 
    nx = 6 
    ny = 2 



    qwns = [ 
         (1,aa[:,NEWMOM,X],"Amomx"), 
         (2,aa[:,NEWMOM,Y],"Amomy"), 
         (3,aa[:,NEWMOM,Z],"Amomz"), 
         (4,aa[:,NEWPOL,X],"Apolx"), 
         (5,aa[:,NEWPOL,Y],"Apoly"), 
         (6,aa[:,NEWPOL,Z],"Apolz"), 

         (7,bb[:,NEWMOM,X],"Bmomx"), 
         (8,bb[:,NEWMOM,Y],"Bmomy"), 
         (9,bb[:,NEWMOM,Z],"Bmomz"), 
         (10,bb[:,NEWPOL,X],"Bpolx"), 
         (11,bb[:,NEWPOL,Y],"Bpoly"), 
         (12,bb[:,NEWPOL,Z],"Bpolz"), 

           ]

    for i,q,label in qwns:
        plt.subplot(ny, nx, i)
        plt.hist(q, bins=bins, histtype="step", label=label)
    pass
    plt.show()


def dirpol(scf):
    assert len(scf.seqs) == 1
    seq_ = scf.seqs[0]

    fr0 = seq_.split().index("SC") - 1
    #fr1 = seq_.split().index("SA") 
    fr1 = len(seq_.split()) - 1 

    bins = 100 
    nx = 12 
    ny = fr1 - fr0 
    offset = 0

    log.info(" seq_ %s fr0 %d fr1 %d ny %d " % (seq_, fr0, fr1, ny))

    for fr in range(fr0,fr1):
        to = fr + 1

        adir, bdir = scf.rdir(fr, to)
        apol, bpol = scf.rpol_(fr)

        qwns = [ 
            (adir[:,X],"adirx"),
            (adir[:,Y],"adiry"),
            (adir[:,Z],"adirz"),
            (apol[:,X],"apolx"),
            (apol[:,Y],"apoly"),
            (apol[:,Z],"apolz"),

            (bdir[:,X],"bdirx"),
            (bdir[:,Y],"bdiry"),
            (bdir[:,Z],"bdirz"),
            (bpol[:,X],"bpolx"),
            (bpol[:,Y],"bpoly"),
            (bpol[:,Z],"bpolz"),
        ]

        for i,(q,label) in enumerate(qwns):
            plt.subplot(ny, nx, offset+i+1)
            plt.hist(q, bins=bins, histtype="step", label=label)
        pass
        offset += nx
    pass
    plt.show()



def abplt(a,b, bins=100,nx=2,ny=1,offset=0, title=""):

    ax = a[:,0]
    ay = a[:,1]
    az = a[:,2]

    nax = np.where(np.isnan(ax))[0] 
    nay = np.where(np.isnan(ay))[0] 
    naz = np.where(np.isnan(az))[0] 

    if nax+nay+naz > 0:
       log.warning("A: nan found in %s nax %d nay %d naz %d " % (title, nax, nay, naz)) 

    ax = ax[~np.isnan(ax)]
    ay = ay[~np.isnan(ay)]
    az = az[~np.isnan(az)]



    bx = b[:,0]
    by = b[:,1]
    bz = b[:,2]

    nbx = np.where(np.isnan(bx))[0] 
    nby = np.where(np.isnan(by))[0] 
    nbz = np.where(np.isnan(bz))[0] 

    if nbx+nby+nbz > 0:
       log.warning("B: nan found in %s nbx %d nby %d nbz %d " % (title, nbx, nby, nbz)) 

    bx = bx[~np.isnan(bx)]
    by = by[~np.isnan(by)]
    bz = bz[~np.isnan(bz)]


    plt.subplot(ny,nx,1+offset+0)
    plt.hist(ax,bins=bins,histtype="step", label="ax")
    plt.hist(ay,bins=bins,histtype="step", label="ay")
    plt.hist(az,bins=bins,histtype="step", label="az")

    plt.subplot(ny,nx,1+offset+1)
    plt.hist(bx,bins=bins,histtype="step", label="bx")
    plt.hist(by,bins=bins,histtype="step", label="by")
    plt.hist(bz,bins=bins,histtype="step", label="bz")


def dirpol(scf, fr, to):
    nx = 2
    ny = 2

    a,b = scf.rdir(fr=fr,to=to)
    abplt(a,b, bins=100, nx=nx, ny=ny, offset=0, title="dirpol/rdir")

    a,b = scf.rpol_(fr=fr)
    abplt(a,b, bins=100, nx=nx, ny=ny, offset=ny, title="dirpol/rpol")

    plt.show()


def poldot(scf, fr, oldpol=[0,1,0], bins=100):
    """
    dot product between old and new polarization
    """
    a,b = scf.rpol_(fr=fr)
    act = costheta_( np.tile( oldpol, len(a) ).reshape(-1,3), a)
    bct = costheta_( np.tile( oldpol, len(b) ).reshape(-1,3), b)

    plt.hist(act, bins=bins, histtype="step", label="act")
    plt.hist(bct, bins=bins, histtype="step", label="bct")
  
    plt.show()




if __name__ == '__main__':
    ok = opticks_main(doc=__doc__, tag="1", src="torch", det="concentric")  

    log.info(ok.brief)

    cf = CF(ok)

    if not ok.ipython:
        log.info("early exit as non-interactive")
        sys.exit(0)

    sa = cf.a.all_seqhis_ana
    sb = cf.b.all_seqhis_ana

    pfxseqhis = ok.pfxseqhis   ## eg ".6ccd" standing for "TO BT BT SC .."
    
    if len(pfxseqhis) > 0:
        log.info(" pfxseqhis [%s] label [%s] " % (pfxseqhis, sa.af.label(pfxseqhis)))
        cf.init_spawn([pfxseqhis]) 
        scf = cf.ss[0]

        if pfxseqhis[0] == ".":
            to = pfxseqhis[::-1].index(".")
            fr = to - 1

            dirpol(scf, fr, to)
            #poldot(scf, fr )   
        pass 
    else:
        scf = None
    pass






    

