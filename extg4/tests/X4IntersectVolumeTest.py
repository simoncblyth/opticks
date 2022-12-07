#!/usr/bin/env python
"""
X4IntersectVolumeTest.py : 2D scatter plots of geometry intersect positions
============================================================================

* typically used from xxv.sh 

"""

import os, logging, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import mpplt_add_contiguous_line_segments, mpplt_add_line


log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

try:
    import matplotlib.pyplot as mp 
except ImportError:
    mp = None
pass

try:
    import pyvista as pv
    from pyvista.plotting.colors import hexcolors  
    themes = "default dark paraview document".split()
    pv.set_plot_theme(themes[1])
except ImportError:
    pv = None
    hexcolors = None
pass

#mp = None
pv = None

X,Y,Z = 0,1,2

efloatlist_ = lambda ekey:list(map(float, filter(None, os.environ.get(ekey,"").split(","))))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    log.info("X4IntersectVolumeTest.py")

    geom = os.environ.get("X4IntersectVolumeTest_GEOM", "body_phys")

    #reldir = os.environ.get("CXS_RELDIR", "extg4/X4IntersectVolumeTest" ) 
    #basedir = os.path.expandvars(os.path.join("/tmp/$USER/opticks",reldir, geom ))

    basedir = os.path.expandvars("$FOLD")

    transforms = np.load(os.path.join(basedir, "transforms.npy"))
    transforms_names = np.loadtxt( os.path.join(basedir, "transforms_names.txt"), dtype=np.object ) 
    soname_prefix = os.path.commonprefix(list(map(str, transforms_names))) 

    figsdir = os.path.join(basedir, "figs")
    if not os.path.isdir(figsdir):
        os.makedirs(figsdir)
    pass

    colors = "red green blue cyan magenta yellow pink orange purple lightgreen".split()
    gcol = "grey"


    savefig = True
    figname = "isect"

    log.info("figsdir %s " % figsdir)
    log.info("basedir %s" % basedir)
    log.info("transforms.shape %s" % str(transforms.shape))
    log.info("transforms_names %s" % ",".join(transforms_names))
    log.info("soname_prefix %s" % soname_prefix)

    topline = "X4IntersectVolumeTest.py"
    botline = "%s" % (geom)
    thirdline = soname_prefix

    isects = {}
    for soname in transforms_names:
        isects[soname] = Fold.Load(basedir, soname, "X4Intersect")
    pass
    LABELS = list(filter(None,os.environ.get("LABELS","").split(",")))
    SZ = float(os.environ.get("SZ",3))

     

    XFOLD = os.environ.get("XFOLD", "/tmp/U4PMTFastSimTest")
    XPID = int(os.environ.get("XPID", -1 ))
    EXTRA = os.environ.get("EXTRA", "/tmp/U4PMTFastSimTest/pid726.npy")

    xfold = None
    extra = None
    extra_type = None
    spud = None 

    if os.path.isdir(XFOLD) and XPID > -1:
        xfold = Fold.Load(XFOLD, symbol="xfold")
        extra = xfold.record[XPID]
        extra_type = "record"
        a_spud = getattr(xfold, "A_SPhoton_Debug", None)
        b_spud = getattr(xfold, "B_SPhoton_Debug", None)
        if not a_spud is None: spud = a_spud
        if not b_spud is None: spud = b_spud
        if not a_spud is None and not b_spud is None:
            print("HMM BOTH A AND B SPUDS : a_spud : %s b_spud: %s " % (str(a_spud.shape), str(b_spud.shape)))
        pass
        if not spud is None:
            print("spud %s " % (str(spud.shape)))  
        pass

        print("loaded extra %s from EXTRA.xfold %s XPID %d extra_type %s  " % (str(extra.shape), EXTRA, XPID, extra_type ))
        extra_wl = extra[:,2,3] 
        extra = extra[extra_wl > 0]
        print("after restrict to extra_wl > 0 : to remove tail zeros : extra %s " % (str(extra.shape)))
        # TODO: assert that all the wl zero skipped are at the tail 
    elif os.path.exists(EXTRA):
        extra = np.load(EXTRA)
        extra_type = "ModelTrigger"
        print("loaded extra %s from EXTRA %s extra_type %s " % (str(extra.shape), EXTRA, extra_type ))
    pass     


     
    REVERSE = int(os.environ.get("REVERSE","0")) == 1
    print("REVERSE : %d " % REVERSE)

    size = np.array([1280, 720])

    axes = X,Z
    H,V = axes


    if mp: 
        fig, ax = mp.subplots(figsize=size/100.) # 100 dpi 
        ax.set_aspect('equal')
       
        soname0 = transforms_names[0]
        isect0 = isects[soname0]
        gpos = isect0.gs[:,5,:3]    # last line of the transform is translation

        hlim = gpos[:,H].min(), gpos[:,H].max()
        vlim = gpos[:,V].min(), gpos[:,V].max()
        ax.set_xlim( hlim )
        ax.set_ylim( vlim )

        if "GS" in os.environ:
            ax.scatter( gpos[:,H], gpos[:,V], s=SZ, color=gcol ) 
        pass
        num = len(transforms_names)
        for j in range(num):
            i = num - 1 - j  if REVERSE else j              
            soname = transforms_names[i] 
            isect = isects[soname]
            tran = np.float32(transforms[i,0])
            ipos = isect.isect[:,0,:3] + tran[3,:3]
            color = colors[ i % len(colors)]

            label = str(soname)[len(soname_prefix):]
            if label[0] == "_": label = label[1:]   # seems labels starting "_" have special meaning to mpl, causing problems
            label = label.replace("solid","s")

            select = len(LABELS) == 0 or label in LABELS
            print(" %2d : %30s : %15s : %s " % (i, soname, label, select ))
            if select:
                ax.scatter( ipos[:,H], ipos[:,V], s=SZ, color=color, label=label ) 
            pass
        pass

        if not spud is None:
            """                       
            Illustrate the normal vector::

 
                       b    b:intersect+NRM_SCALE*normal 
                       :
                  -----i--- i: spud_pos
                       :
                       a    a:intersect-NRM_SCALE*normal 

            """
            nrm_scale = float(os.environ.get("NRM_SCALE","50"))
            spud_pos = spud[:,0,:3]
            spud_nrm = spud[:,3,:3]

            nrm_b = spud_pos + nrm_scale*spud_nrm
            nrm_i = spud_pos
            nrm_a = spud_pos - nrm_scale*spud_nrm

            for i in range(len(spud)):
                mpplt_add_line(ax, nrm_a[i], nrm_b[i], axes )   
                ax.arrow( nrm_b[i,H], nrm_b[i,V], 10*spud_nrm[i,H], 10*spud_nrm[i,V], head_width=10, head_length=10, fc='k', ec='k' )
            pass
            #ax.scatter( nrm_b[:,H], nrm_b[:,V], s=10*SZ )
            ax.scatter( nrm_i[:,H], nrm_i[:,V], s=100*SZ )  
        pass

        if not extra is None:
            print("extra %s EXTRA %s extra_type %s " % ( str(extra.shape), EXTRA, extra_type ))
            if extra_type == "ModelTrigger":
                ModelTriggerYES = extra[np.where(extra[:,3,0].astype(np.int64) == 1)]  
                ModelTriggerNO = extra[np.where(extra[:,3,0].astype(np.int64) == 0)]  

                if "ModelTriggerYES" in os.environ:
                    ax.scatter( ModelTriggerYES[:,0,H], ModelTriggerYES[:,0,V], s=50, color="red"   )
                    thirdline += " ModelTriggerYES" 
                pass
                if "ModelTriggerNO" in os.environ:
                    ax.scatter( ModelTriggerNO[:,0,H], ModelTriggerNO[:,0,V], s=50, color="blue"  )
                    thirdline += " ModelTriggerNO" 
                pass
            pass
            mpplt_add_contiguous_line_segments(ax, extra[:,0,:3], axes=(H,V), label=None )

            if extra_type == "record":
                hv = extra[:,0,(H,V)]   
                tweak = False
                
                for i in range(len(extra)):
                    dx,dy = 0,0
                    if tweak:
                        if i==2: dx,dy=-10,0  
                        if i==3: dx,dy=0,-10  
                        if i==4: dx,dy=-10,0  
                        if i==16: dx,dy=10,-10  
                        backgroundcolor="yellow"
                    else:
                        backgroundcolor="yellow"
                        #backgroundcolor=None 
                    pass
                    if backgroundcolor is None:
                        ax.text(dx+hv[i,0],dy+hv[i,1], str(i), fontsize=15 )
                    else:
                        ax.text(dx+hv[i,0],dy+hv[i,1], str(i), fontsize=15, backgroundcolor=backgroundcolor )
                    pass
                pass
            pass
        pass
        pass

        #loc = "lower left"
        #loc = "upper right"
        loc = "upper left"
        LOC = os.environ.get("LOC",loc)
        if LOC != "skip":
            ax.legend(loc=LOC,  markerscale=4)
        pass
        fig.suptitle("\n".join([topline,botline,thirdline]))
        fig.show()

        if savefig:
            figpath = os.path.join(figsdir,figname+"_mpplt.png")
            log.info("saving figpath %s " % figpath)
            fig.savefig(figpath)
        pass 
    pass


