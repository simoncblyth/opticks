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
from opticks.ana.ab   import AB
from opticks.ana.cfh  import CFH

STEP = 4

X,Y,Z=0,1,2


def butterfly_yz(plt, a, b, pt):
    plt.subplot(1, 2, 1)
    plt.scatter(a[:,pt,Y],a[:,pt,Z])

    plt.subplot(1, 2, 2)
    plt.scatter(b[:,pt,Y],b[:,pt,Z])


def butterfly(plt, ab):
    """
    Expecting 8cc6ccd 

              TO BT BT SC BT BT AB
              p0 p1 p2 p3 p4 p5 p6
                       
    p3: scatter occurs at point on X axis
    p4: first intersection point after the scatter  
    """
    a,b = ab.rpost()
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




def dirpol(ab, flg0="SC"):

    iflg = ab.iflg(flg0)
    nrec = ab.nrec()

    fr0 = iflg - 1
    fr1 = nrec - 1 

    bins = 100 
    nx = 12 
    ny = fr1 - fr0 
    offset = 0

    log.info(" lab0 %s fr0 %d fr1 %d ny %d " % (lab0, fr0, fr1, ny))

    for fr in range(fr0,fr1):
        to = fr + 1

        adir, bdir = ab.rdir(fr, to)
        apol, bpol = ab.rpol_(fr)

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
    """
    Two subplots containg x,y,z coordinates of a and b 
    """
    ax = a[:,0]
    ay = a[:,1]
    az = a[:,2]

    nax = np.count_nonzero(np.isnan(ax))
    nay = np.count_nonzero(np.isnan(ay))
    naz = np.count_nonzero(np.isnan(az))

    if nax+nay+naz > 0:
       log.warning("A: nan found in %s nax %d nay %d naz %d " % (title, nax, nay, naz)) 

    ax = ax[~np.isnan(ax)]
    ay = ay[~np.isnan(ay)]
    az = az[~np.isnan(az)]

    bx = b[:,0]
    by = b[:,1]
    bz = b[:,2]

    nbx = np.count_nonzero(np.isnan(bx))
    nby = np.count_nonzero(np.isnan(by))
    nbz = np.count_nonzero(np.isnan(bz))

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


def dirpol(ab, fr, to):
    """
    """

    log.info("dirpol fr %d to %d " % (fr, to ))

    nx = 2
    ny = 2

    a,b = ab.rdir(fr=fr,to=to)
    abplt(a,b, bins=100, nx=nx, ny=ny, offset=0, title="dirpol/rdir")

    a,b = ab.rpol_(fr=fr)
    abplt(a,b, bins=100, nx=nx, ny=ny, offset=ny, title="dirpol/rpol")

    plt.show()



def scatter(ab):
    """
    """
    ab.sel = "TO SC BT BT BT BT SA"

    fr = ab.iflg("SC")

    a_opol, b_opol = ab.rpol_(fr=fr-1)
    a_npol, b_npol = ab.rpol_(fr=fr)

    a_odir, b_odir = ab.rdir(fr=fr-1,to=fr)
    a_ndir, b_ndir = ab.rdir(fr=fr,to=fr+1)

    a_ct = costheta_( np.cross(a_opol, a_ndir ), a_opol )
    b_ct = costheta_( np.cross(b_opol, b_ndir) , b_opol )

    return a_ct, b_ct




def poldot(ab, fr, oldpol=[0,1,0], bins=100):
    """
    dot product between old and new polarization


  
        o_pol,o_dir  / n_pol,n_dir
          |         /
          |--------/


    * https://bugzilla-geant4.kek.jp/show_bug.cgi?id=207
 
    New pol needs to be

    * Perpendicular to the new momentum vector
    * Same plane as the new momentum vector and initial polarization vector


    OpRayleigh.cc::

        164            // calculate the new polarization direction
        165            // The new polarization needs to be in the same plane as the new
        166            // momentum direction and the old polarization direction
        167            OldPolarization = aParticle->GetPolarization();
        168            constant = -NewMomentumDirection.dot(OldPolarization);
        ...
        170            NewPolarization = OldPolarization + constant*NewMomentumDirection;
        ///
        ///          linear combination of oldpol and newdir is in same plane as these
        ///
        171            NewPolarization = NewPolarization.unit();


    ::

             constant = -n_dir.o_pol

             n_pol = o_pol + (-n_dir.o_pol )n_dir
          
             n_dir.n_pol = n_dir.o_pol + (-n_dir.o_pol) n_dir.n_dir = 0


             o_pol ^ n_dir : is normal to the plane 

             n_pol.( o_pol ^ n_dir ) == 0 



    """
    a,b = ab.rpol_(fr=fr)  # new pol
    act = costheta_( np.tile( oldpol, len(a) ).reshape(-1,3), a)
    bct = costheta_( np.tile( oldpol, len(b) ).reshape(-1,3), b)

    plt.hist(act, bins=bins, histtype="step", label="act")
    plt.hist(bct, bins=bins, histtype="step", label="bct")
  
    plt.show()




def debug_plotting(ok, ab):
    pfxseqhis = ok.pfxseqhis   ## eg ".6ccd" standing for "TO BT BT SC .."
    pfxseqmat = ok.pfxseqmat 
    
    if len(pfxseqhis) > 0:
        log.info(" pfxseqhis [%s]  " % (pfxseqhis))

        ab.sel = pfxseqhis

        fr = ab.iflg("SC")

        if fr is None:
            log.fatal("expecting one line selection including SC")
        else:
            dirpol(ab, fr, fr+1)
            poldot(ab, fr )   
        pass 
    elif len(pfxseqmat) > 0:
        log.info(" pfxseqmat [%s] " % (pfxseqmat))
        ab.flv = "seqmat"
        ab.sel = pfxseqmat 
    else:
        pass




if __name__ == '__main__':
    ok = opticks_main(doc=__doc__, tag="1", src="torch", det="concentric", smry=True)  

    print "ok.smry %d " % ok.smry 
    log.info(ok.brief)



    ab = AB(ok)
    print ab

    if not ok.ipython:
        log.info("early exit as non-interactive")
        sys.exit(0)


    debug_plotting(ok, ab)

    #ab.sel = "[TO] BT BT BT BT SA" 
    #
    #hh = ab.hh

    

    

