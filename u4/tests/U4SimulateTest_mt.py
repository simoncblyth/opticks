#!/usr/bin/env python
"""
U4SimulateTest_mt.py
========================

::

    u4t
    ./U4SimulateTest.sh mt

"""
import os, textwrap, numpy as np
from opticks.ana.fold import Fold, AttrBase
from opticks.ana.p import * 

# TODO: these should be coming from standard place, not duplicate
BULK_ABSORB = 0x1 <<  3
SURFACE_DETECT = 0x1 << 6 
SURFACE_ABSORB = 0x1 << 7
SURFACE_DREFLECT  = 0x1 << 8
SURFACE_SREFLECT = 0x1 <<  9   
BOUNDARY_REFLECT  = 0x1 << 10
BOUNDARY_TRANSMIT = 0x1 << 11
TORCH = 0x1 << 12

AB = BULK_ABSORB.bit_length()
SD = SURFACE_DETECT.bit_length()
SA = SURFACE_ABSORB.bit_length()
DR = SURFACE_DREFLECT.bit_length()
SR = SURFACE_SREFLECT.bit_length()   
BR = BOUNDARY_REFLECT.bit_length()
BT = BOUNDARY_TRANSMIT.bit_length()
TO = TORCH.bit_length()

N = int(os.environ.get("VERSION", "-1"))
CMDLINE = "N=%d ./U4SimulateTest.sh mt" % N

TEST = os.environ.get("TEST", "Manual")
GEOM = os.environ.get("GEOM", "DummyGEOM")
GEOMList = os.environ.get("%s_GEOMList" % GEOM, "DummyGEOMList") 

VERSION = N  
MODE =  int(os.environ.get("MODE", "2"))
assert MODE in [0,2,3]
PIDX = int(os.environ.get("PIDX", "123")) 


if MODE > 0:
    from opticks.ana.pvplt import * 
pass

from opticks.u4.tests.ModelTrigger_Debug import ModelTrigger_Debug       

axes = 0, 2  # X,Z
H,V = axes 

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    end = t.photon[:,0,:3]
    q_ = t.seq[:,0]                   #  t.seq shape eg (1000, 2, 2)  
    q = ht.seqhis(q_)                 # history label eg b'TO BT BT SA ... lots of blankspace...'  
    qq = ht.Convert(q_)               # array of step point flags 
    n = np.sum( seqnib_(q_), axis=1 ) # nibble count

    w_sr = np.where( qq == SR )       # selecting step points with SR 
    w_to = np.where( qq == TO ) 
    w_br = np.where( qq == BR ) 
   
    sr_pos = t.record[tuple(w_sr[0]), tuple(w_sr[1]),0,:3]   # step point positions of SR
    to_pos = t.record[tuple(w_to[0]), tuple(w_to[1]),0,:3]   
    br_pos = t.record[tuple(w_br[0]), tuple(w_br[1]),0,:3]   


    #LAYOUT = os.environ.get("LAYOUT", "")   
    ## ana env may not match run env, making it fragile to assume so 
    LAYOUT = t.photon_meta.LAYOUT[0]  
    CHECK = t.photon_meta.CHECK[0]  

    print("TEST:%s" % (TEST) )
    print("CMDLINE:%s" % (CMDLINE) )
    print("LAYOUT:%s" % (LAYOUT) )
    print("CHECK:%s" % (CHECK) )
    print("MODE:%d" % (MODE) )
    print("PIDX:%d" % (PIDX) )
    print("N:%d" % (N) )

    mtd = ModelTrigger_Debug(t, symbol="mtd", publish=False)  # publish:True crashing 
    print(mtd)

    if 'SPECS' in os.environ:
        SPECS = np.array(t.U4R_names.lines)     # step specification, for skip fake debugging 
        st_ = t.aux[:,:,2,3].view(np.int32)
        st = SPECS[st_]
        st_dump = True
    else:
        SPEC, st_, st = None, None, None
        st_dump = False
    pass  

    if st_dump:
        u_st, n_st = np.unique(st, return_counts=True)
        expr = "np.c_[n_st,u_st][np.argsort(n_st)[::-1]]"
        print(expr)
        print(eval(expr))
    pass

    exprs = r"""
    q[PIDX] 
    t.record[PIDX,:n[PIDX],0] 
    mtd.pv[mtd.index==PIDX]
    np.unique(mtd.whereAmI[mtd.trig==1],return_counts=True)
    """    
    exprs_ = list(filter(None,textwrap.dedent(exprs).split("\n")))
    for expr in exprs_:
        print("\n%s ## " % expr)
        print(eval(expr))
    pass


    mtd_outside = np.logical_and(mtd.trig == 1, mtd.EInside1 == 0 )
    
    mtd_trig = mtd.trig == 1 
    mtd_upper = mtd.pos[:,2] > 1e-4   
    mtd_mid   = np.abs( mtd.pos[:,2]) < 1e-4
    mtd_lower = mtd.pos[:,2] < -1e-4   
    mtd_pyrex  = mtd.whereAmI_ == 1 
    mtd_vacuum = mtd.whereAmI_ == 2 

    mtd_trig_vacuum = np.logical_and(mtd_trig, mtd_vacuum)
    mtd_trig_vacuum_upper = np.logical_and(mtd_trig_vacuum, mtd_upper )
    mtd_trig_vacuum_mid   = np.logical_and(mtd_trig_vacuum, mtd_mid )
    mtd_trig_vacuum_lower = np.logical_and(mtd_trig_vacuum, mtd_lower )

    mtd_trig_pyrex  = np.logical_and(mtd_trig, mtd_pyrex )
    mtd_trig_pyrex_upper = np.logical_and(mtd_trig_pyrex, mtd_upper )
    mtd_trig_pyrex_mid   = np.logical_and(mtd_trig_pyrex, mtd_mid )
    mtd_trig_pyrex_lower = np.logical_and(mtd_trig_pyrex, mtd_lower )

    

    rqwns = textwrap.dedent("""
    GEOM
    GEOMList    
    LAYOUT
    CHECK
    TEST 
    """) 

    lqwns = textwrap.dedent("""
    mtd.IMPL
    t.photon.shape
    mtd.pos.shape

    mtd_trig
    mtd_upper
    mtd_mid
    mtd_lower

    mtd_pyrex
    mtd_vacuum



    mtd_trig_pyrex
    mtd_trig_pyrex_upper
    mtd_trig_pyrex_mid
    mtd_trig_pyrex_lower

    mtd_trig_vacuum
    mtd_trig_vacuum_upper
    mtd_trig_vacuum_mid
    mtd_trig_vacuum_lower
    """)


    llines = []
    for qwn in lqwns.split("\n"): 
        if len(qwn) == 0:
            llines.append("")
        elif qwn.find(".") > -1:
            llines.append("%30s : %s" % ( qwn, eval(qwn)))
        else:
            num = np.count_nonzero(eval(qwn))  
            llines.append("%30s : %d " % ( qwn, num ))
        pass
    pass
    lanno = "\n".join(llines)
    print(lanno)

    if not "NANNO" in os.environ:
        os.environ["LHSANNO"] = lanno 
    pass

    rlines = []
    for qwn in rqwns.split("\n"): 
        line = "" if len(qwn) == 0 else "%s : %s" % ( qwn, eval(qwn))
        rlines.append(line)
    pass
    ranno = "\n".join(rlines)
    print(ranno)
    if not "NANNO" in os.environ:
        os.environ["RHSANNO"] = ranno 
    pass






    idxs = np.unique(mtd.index[mtd_trig_pyrex_lower])   # photon indices 

    flagmask = t.photon[:,3,3].view(np.int32) 
    sd = flagmask & SURFACE_DETECT != 0 
    sa = flagmask & SURFACE_ABSORB != 0 

    ## branch on layout as coordinates are specific to each (eg midline of the left hand PMT in two_pmt)
    if LAYOUT == "two_pmt":
        x_midline = np.logical_and( end[:,0] > -251, end[:,0] < -249 )    
        z_midline = np.logical_and( end[:,2] > -250, end[:,2] <  250 )    
        xz_midline = np.logical_and( x_midline, z_midline )
    elif LAYOUT == "one_pmt":
        x_midline = np.logical_and( end[:,0] > -245, end[:,0] < 245 )    
        z_midline = np.logical_and( end[:,2] > -1,   end[:,2] <  1 )    
        xz_midline = np.logical_and( x_midline, z_midline )
    else:
        xz_midline = None 
    pass
    w_midline = np.where(xz_midline)[0]  


    nn = n[w_midline] 

    end2 = t.record[w_midline, tuple(nn-1), 0, :3]   # recreate photon endpoint pos from record array 
    assert(np.all( end[w_midline] == end2 ))

    penultimate =  t.record[w_midline, tuple(nn-2), 0, :3]  
    prior =  t.record[w_midline, tuple(nn-3), 0, :3]          


    #ppos0_ = "None"
    #ppos0_ = "end"
    #ppos0_ = "end[sd] # photon SD endpoints around the upper hemi"
    #ppos0_ = "end[sa] # photon SA endpoints around the upper hemi and elsewhere"
    #ppos0_ = "end[w_midline]  # photons ending on midline " 
    #ppos0_ = "mtd.pos[mtd_outside] # just around upper hemi "
    ppos0_  = "mtd.pos[mtd_trig]" 
    #ppos0_ = "mtd.pos[mtd_trig_pyrex]  # Simple:just around upper hemi, Buggy:also dynode/MCP sprinkle "
    #ppos0_ = "mtd.pos[mtd_trig_pyrex_lower] # "
    #ppos0_ = "mtd.pos[mtd_trig_vacuum] # mostly on midline, sprinkle of obliques around upper hemi "
    #ppos0_ = "mtd.next_pos[mtd_trig] # just around upper hemi"
    #ppos0_ = "sr_pos # SR positions  "
    #ppos0_ = "to_pos # TO positions  "
    #ppos0_ = "br_pos # BR positions  "
    #ppos0_ = "mtd.pos[mtd_pyrex] #   "
    #ppos0_ = "mtd.pos[mtd_trig_pyrex] #   "

    #ppos1_ = "None" 
    #ppos1_ = "mtd.pos[mtd_trig_vacuum_upper]"
    #ppos1_ = "mtd.next_pos[mtd_trig_vacuum_upper]"
    ppos1_ = "mtd.pos[mtd_trig_pyrex_lower] # BUG : Pyrex triggers inside inner2 : UNPHYSICAL  "
    #ppos1_  = "end[xz_midline]"
    #ppos1_  = "penultimate  # photon position prior to terminal one"
    #ppos1_  = "prior  # two positions before last"

    ppos2_ = "None"


    if TEST == "quiver":
        qsel = "mtd_trig_vacuum_upper"
        #qsel = "np.where(mtd_trig_vacuum_upper)[0][:10]"   
        #qsel = "mtd_trig"
        ppos0_ = "mtd.pos[%(qsel)s] #QUIVER "  % locals()
        ppos1_ = "mtd.dir[%(qsel)s] #QUIVER "  % locals()
        ppos2_ = "mtd.next_pos[%(qsel)s] #QUIVER " % locals()
    else:
        print("using default ppos0_ ppos1_ ppos2_ exprs ")
    pass 


    ppos0  = eval(ppos0_)
    ppos1  = eval(ppos1_) 
    ppos2  = eval(ppos2_) 

    elem = []
    elem.append(CMDLINE)
    if not ppos0 is None: elem.append("blue:%s" % ppos0_)
    if not ppos1 is None: elem.append("red:%s" % ppos1_)
    if not ppos2 is None: elem.append("green:%s" % ppos2_)
    label = "\n".join(elem)


    exprs = textwrap.dedent("""
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==2),mtd.EInside1==0))
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==2),mtd.EInside1==1))
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==2),mtd.EInside1==2))
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==1),mtd.EInside1==0))
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==1),mtd.EInside1==1))
    np.count_nonzero(np.logical_and(np.logical_and(mtd.trig==1,mtd.whereAmI_==1),mtd.EInside1==2))
    """)

    for expr in exprs.split():
        print(expr)
        print(eval(expr))
    pass


    if MODE == 0:
        print("not plotting as MODE 0  in environ")
    elif MODE == 2:
        fig, axs = mpplt_plotter(label=label)
        ax = axs[0]
        ax.set_ylim(-250,250)
        ax.set_xlim(-500,500)

        if "#QUIVER" in ppos0_ and "#QUIVER" in ppos1_ and "#QUIVER" in ppos2_:
            assert not ppos0 is None
            assert not ppos1 is None
            assert not ppos2 is None
            ax.quiver( ppos0[:,H], ppos0[:,V],  ppos1[:,H], ppos1[:,V], units="width", width=0.0002, scale=10.0 )
            ax.scatter( ppos0[:,H], ppos0[:,V], s=1, c="r") 
            ax.scatter( ppos2[:,H], ppos2[:,V], s=1, c="g") 
        else:
            if not ppos0 is None: ax.scatter( ppos0[:,H], ppos0[:,V], s=1 )  
            if not ppos1 is None: ax.scatter( ppos1[:,H], ppos1[:,V], s=1, c="r" )  
        pass
        fig.show()

    elif MODE == 3:
        pl = pvplt_plotter(label)
        os.environ["EYE"] = "0,100,165"
        os.environ["LOOK"] = "0,0,165"
        pvplt_viewpoint(pl)
        pl.add_points(ppos0)
        pl.show()
    pass
pass
