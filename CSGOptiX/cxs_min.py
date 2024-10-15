#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

print("[from opticks.sysrap.sevt import SEvt, SAB")
from opticks.sysrap.sevt import SEvt, SAB
print("]from opticks.sysrap.sevt import SEvt, SAB")




PLOT = os.environ.get("PLOT","scatter")
GLOBAL = int(os.environ.get("GLOBAL","0")) == 1
MODE = int(os.environ.get("MODE","3")) 
SEL = int(os.environ.get("SEL","0")) 


if MODE in [2,3]:
    from opticks.ana.pvplt import *   
    # HMM this import overrides MODE, so need to keep defaults the same 
pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("GLOBAL:%d MODE:%d SEL:%d" % (GLOBAL,MODE, SEL))

    a = SEvt.Load("$AFOLD", symbol="a")
    print(repr(a))

    if "BFOLD" in os.environ:   
        b = SEvt.Load("$BFOLD", symbol="b") 
        print(repr(b))
        ab = SAB(a,b) 
        print(repr(ab))
    pass


    e = a 

    qtab = e.minimal_qtab()
    print("qtab")
    print(qtab)


    if hasattr(e.f, 'record'):
        pp = e.f.record[:,:,0,:3].reshape(-1,3)
        found = "record"
    elif hasattr(e.f, 'hit'):
        pp = e.f.hit[:,0,:3]
        found = "hit"
    elif hasattr(e.f, 'photon'):
        pp = e.f.photon[:,0,:3]
        found = "photon"
    elif hasattr(e.f, 'inphoton'):
        pp = e.f.inphoton[:,0,:3]
        found = "inphoton"
    elif hasattr(e.f, 'genstep'):
        pp = e.f.genstep[:,1,:3]
        found = "genstep"
    else:
        pp = None
        found = "NONE" 
        pass
    pass

    assert not pp is None

    gpos = np.ones( [len(pp), 4 ] )
    gpos[:,:3] = pp
    lpos = np.dot( gpos, e.f.sframe.w2m )
    upos = gpos if GLOBAL else lpos

    if MODE in [0,1]:
        print("not plotting as MODE %d in environ" % MODE )
    elif MODE == 2:
        label = "%s : " % ( e.f.base.replace("/data/blyth/opticks/","") )
        label +=  " : %s " % PLOT

        if PLOT.startswith("seqnib") and hasattr(e.f, "seqnib"):
            expl = "Photon History Step Counts Occurrence in single 1M photon event"
        elif PLOT.startswith("thit") and hasattr(e.f, "hit"):
            expl = "Histogram hit times[ns] of all(and step tail) : from  1M photon TORCH event"       
        pass 
        title = "\n".join([label, expl])
        pl = mpplt_plotter(label=title)
        fig, axs = pl
        assert len(axs) == 1
        ax = axs[0]
    elif MODE == 3:

        label = "%s : found %s " % ( e.f.base, found )
        print(label)
        pl = pvplt_plotter(label)
        pvplt_viewpoint(pl)   # sensitive to EYE, LOOK, UP envvars
        pvplt_frame(pl, e.f.sframe, local=not GLOBAL )
    pass
    

    H,V = 0,2  # X, Z

    if SEL == 1:
        sel = np.logical_and( np.abs(upos[:,H]) < 500, np.abs(upos[:,V]) < 500 )
        spos = upos[sel]
    else:
        spos = upos 
    pass

    if MODE == 2:

        if PLOT.startswith("scatter"):
            ax.scatter( spos[:,H], spos[:,V], s=0.1 )
        elif PLOT.startswith("seqnib") and hasattr(e.f, "seqnib"):
            seqnib = e.f.seqnib  

            bounce = np.arange(len(seqnib))
            ax.set_aspect('auto')
            ax.plot( bounce, seqnib )
            ax.scatter( bounce, seqnib )
            ax.set_ylabel("Photon counts for step points (0->31)", fontsize=20 )

            cs_seqnib = np.cumsum(seqnib)   
            ax2 = ax.twinx() 
            ax2.plot( bounce, cs_seqnib, linestyle="dashed" )
            ax2.set_ylabel( "Cumulative counts rising to total of 1M photons", fontsize=20 )
        elif PLOT.startswith("thit") and hasattr(e.f, "hit"):
            hit_time = e.f.hit[:,0,3] 
            mn_hit_time = hit_time.min() 
            mx_hit_time = hit_time.max() 
            bins = np.linspace( mn_hit_time, mx_hit_time, 100 )

            hit_time_n,_  = np.histogram(hit_time, bins=bins ) 
            ax.set_aspect('auto')
            ax.plot( bins[:-1], hit_time_n, drawstyle="steps-mid", label="Simulated Hit time [ns]" )

            hit_idx = e.f.hit.view(np.uint32)[:,3,2] & 0x7fffffff    
            hit_nib = a.f.seqnib[hit_idx]   # nibbles of the hits  

            cut = 23
            hit_time_sel = hit_time[hit_nib>cut]
            sel_time_n, _ = np.histogram(hit_time_sel, bins=bins) 

            ax.plot( bins[:-1], sel_time_n, drawstyle="steps-mid", label="Simulated Hit time [ns] for step points > %d " % cut  )

            ax.set_ylabel("Photon counts in simulated time[ns] bins", fontsize=20)
            ax.set_xlabel("Simulated time[ns]", fontsize=20)

            ax.set_yscale('log')
            ax.legend() 
            # need the seqnib of each photon so can check times of the big seqnib 
            # PLOT=thit MODE=2 ~/o/cxs_min.sh  
        else:
            print("PLOT:%s unhandled" % PLOT)
        pass 
    elif MODE == 3:
        if PLOT.startswith("scatter"):
            pl.add_points(spos[:,:3])
        else:
            pass
        pass 
    else:
        pass
    pass

    if MODE == 2:
        fig.show()
    elif MODE == 3:
        pl.show()
    pass
pass


 

