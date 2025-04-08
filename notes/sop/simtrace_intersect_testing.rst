simtrace_intersect_testing
============================

Simtracing Overview
---------------------

For intersect testing the string to search for is “Simtrace”.

The idea of “Simtracing” is to generate radial rays within a plane
from source points within a grid of points also arranged within
the same plane. The density of grid points can be
arranged to be more within regions of interest.
Ray origin, direction and intersects when they happen
are collected into (n,4,4) arrays which are persisted to NumPy
arrays.

Plotting these intersect positions allows creation of
cross sections through geometry with different volumes identified
with different colors.

This is done both with Opticks and Geant4 geometries with NumPy
arrays saved and comparisons done in python.

With Opticks this can be done globally across entire geometries,
with Geant4 the current implementation is limited to small test
geometries.

I have done lots of intersect testing across many of generations of Opticks
code but the last time I worked on that was quite a while ago.
So trying to get things going will likely not be smooth
and there is too much choice of code/scripts to use.
But I can give my best guess at which scripts to start with.



Selected Scripts with "Simtrace" code
----------------------------------------

::

    ./CSGOptiX/cxt_min.sh
    ./cxt_min.sh      ## symbolic link to the above
        ## revived April 2025, needs pyvista for 3D plotting

    ./CSG/ct.sh
        ## revived 2025 April 3 : querying uses SOPR=0:0 selecting CSGSolid:CSGPrim by index
        ## uses tests/CSG/CSGSimtraceTest.{py,cc} using CSGSimtrace.cc CSGQuery.cc
        ##
        ## SOPR=2:1 LOG=1 ~/o/CSG/ct.sh    ## uses matplotlib plotting
        ##  HMM: this was an early one later used pyvista with 3D option


    ./u4/tests/U4SimtracePlot.sh
    ./u4/tests/U4SimtraceTest.sh
        ## U4VolumeMaker::PV creates Geant4 geometry, so mainly simple geometry
        ## U4SimtraceTest intersects the geometry and saves to FOLD
        ##
        ## has PMTSim_standalone complications (juno only lib giving access to JUNO geometry),
        ## maybe easier to make another executable without that dependency to avoids the complication


    ./u4/tests/U4SimtraceSimpleTest.{sh,cc,py}
        ## to avoid complications with the above start this one 
        ## without the PMTSim_standalone dep for non-JUNO tests



Scripts with "Simtrace"
-------------------------------

::

    P[blyth@localhost opticks]$ find . -name '*.sh' -exec grep -l imtrace {} \;
    ./CSG/CSGSimtraceRerunTest.sh

        ## CSGSimtraceRerunTest.cc : repeating simtrace GPU intersects on the CPU with the csg_intersect CUDA code
        ## uses CSGQuery::intersect_again : looks worthy of revival

    ./CSG/CSGSimtraceSampleTest.sh
    ./CSG/Values.sh
    ./CSG/cf_ct.sh

    ./CSG/ct.sh
        ## revived 2025 April 3 : querying uses SOPR=0:0 selecting CSGSolid:CSGPrim by index
        ## uses tests/CSG/CSGSimtraceTest.{py,cc} using CSGSimtrace.cc CSGQuery.cc
        ##
        ## SOPR=2:1 LOG=1 ~/o/CSG/ct.sh    ## uses matplotlib plotting
        ##  HMM: this was an early one later used pyvista with 3D option

    ./CSG/ct_chk.sh
    ./CSG/mct.sh
    ./CSG/nmskMaskOut.sh
    ./CSG/nmskSolidMask.sh
    ./CSG/nmskSolidMaskVirtual.sh
    ./CSG/tests/CSGSimtraceSampleTest.sh

    ./CSGOptiX/cachegrab.sh
    ./CSGOptiX/cxs.sh
    ./CSGOptiX/cxs_debug.sh
    ./CSGOptiX/cxs_geochain.sh
    ./CSGOptiX/cxs_grab.sh
    ./CSGOptiX/cxs_pub.sh
    ./CSGOptiX/cxs_solidXJfixture.sh
    ./CSGOptiX/cxsim.sh
    ./CSGOptiX/grab.sh
    ./CSGOptiX/pub.sh
    ./CSGOptiX/tmp_grab.sh

    ./CSGOptiX/cxt_min.sh
    ./cxt_min.sh
        ## revived April 2025, needs pyvista for 3D plotting

    ./GeoChain/grab.sh

    ./bin/geomlist.sh
    ./bin/geomlist_test.sh
    ./bin/log.sh


    ./extg4/cf_x4t.sh
    ./extg4/ct_vs_x4t.sh
    ./extg4/mx4t.sh
    ./extg4/x4t.sh
    ./extg4/xxs.sh
          ## extg4 is DEAD CODE


    ./g4cx/cf_gxt.sh
    ./g4cx/gxt.sh
          ## using G4CXSimtraceTest.cc G4CXOpticks::simtrace QSim::simtrace  CSGOptiX::simtrace_launch
          ## LOTS OF WORK NEEDED TO REVIVE

    ./g4cx/tests/G4CXTest.sh
          ## glance suggests simtrace is here just used for scenery against which to show photon histories

    ./sysrap/tests/SSimtrace_check.sh

    ./u4/tests/FewPMT.sh
    ./u4/tests/U4SimtracePlot.sh
    ./u4/tests/U4SimtraceTest.sh
          ## U4VolumeMaker::PV creates Geant4 geometry, so mainly simple geometry
          ## U4SimtraceTest intersects the geometry and saves to FOLD

    ./u4/tests/U4SimtraceTest_one_pmt.sh
    ./u4/tests/U4SimtraceTest_two_pmt.sh
    ./u4/tests/U4SimtraceTest_two_pmt_cf.sh


    ./u4/tests/U4SimulateTest.sh
    ./u4/tests/viz.sh



Converting geometry
---------------------

::

    There are too many scripts in Opticks and I am not able to find a
    straightforward way to do the conversion. I run G4CXOpticks_setGeometry_Test.sh
    to do it. Let me know if there is a better way.

Doing it from the top like you did is best if you want to 
avoid spending time to understand the lower level
CSG and sn.h code and lower level scripts.


Opticks has too many scripts : I cannot find what I want
-----------------------------------------------------------

The problem with scripts is that rolling another one for a 
specific task is faster and simpler than generalizing other 
scripts. I agree there are too many scripts, but finding time to
tidy or document them all is difficult. 

I suggest you use the bash and python scripts as examples
of how to do things rather than trying to debug them. It is
easier to learn by writing your own bash+python, 
especially with ipython.




3D ray trace vizualize geometry
---------------------------------

Converting from the top (ie starting from gdml) eg with g4cx/tests/G4CXOpticks_setGeometry_Test.sh
will also enables you to 3D ray trace visualize your geometry with:: 

    epsilon:opticks blyth$ cat CSGOptiX/cxr_minimal.sh 
    #!/bin/bash
    usage(){ cat << EOU
    cxr_minimal.sh
    ===============

    Does minimal env setup to visualize the last persisted geometry 
    using CSGOptiXRenderInteractiveTest.

    See also cxr_min.sh which does similar but is a lot less minimal.

    EOU
    }

    geombase=$HOME/.opticks/GEOM
    last_CSGFoundry=$(cd $geombase && ls -1dt */CSGFoundry | head -1)
    last_GEOM=$(dirname $last_CSGFoundry)

    export GEOM=$last_GEOM
    export ${GEOM}_CFBaseFromGEOM=$geombase/$GEOM

    bin=CSGOptiXRenderInteractiveTest
    tbin=$(which $bin)

    vv="0 geombase last_CSGFoundry last_GEOM GEOM ${GEOM}_CFBaseFromGEOM vv bin tbin"
    vvp(){ for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done ; }

    vvp 
    $bin
    vvp


For interactive keys see sysrap/SGLFW.h SGLFW::HELP



Pattern of bash+python comparison scripts
-------------------------------------------

Comparing SEvt event info, including simtrace intersects, between folders 
is the basis for many validations so there are a great many examples across 
both live and dead pkgs. 

The general pattern is for the bash script to define envvars
identifying the folders to compare, eg A_FOLD B_FOLD etc.. 
Usually a “run” argument to the bash script does the intersects
and “ana” 

Which are loaded in python with Fold.Load (or sometimes with sevt.Load)


::

    epsilon:opticks blyth$ opticks-f _FOLD | grep Fold.Load 
    ./CSG/tests/SimtraceRerunTest.py:    t = Fold.Load("$T_FOLD", symbol="t")
    ./sysrap/tests/sreport.py:    fold = Fold.Load("$SREPORT_FOLD", symbol="fold")
    ./sysrap/tests/storch_test_cf.py:    a = Fold.Load("$A_FOLD", symbol="a")
    ./sysrap/tests/storch_test_cf.py:    b = Fold.Load("$B_FOLD", symbol="b")
    ./sysrap/tests/sleak.py:    fold = Fold.Load("$SLEAK_FOLD", symbol="fold")
    ./sysrap/tests/sprof.py:    fold = Fold.Load("$SPROF_FOLD", symbol="fold")
    ./sysrap/tests/sreport_ab.py:    print("[sreport_ab.py:fold = Fold.Load A_SREPORT_FOLD [%s]" % asym ) 
    ./sysrap/tests/sreport_ab.py:    a = Fold.Load("$A_SREPORT_FOLD", symbol=asym )
    ./sysrap/tests/sreport_ab.py:    print("[sreport_ab.py:fold = Fold.Load B_SREPORT_FOLD [%s]" % bsym ) 
    ./sysrap/tests/sreport_ab.py:    b = Fold.Load("$B_SREPORT_FOLD", symbol=bsym )
    ./sysrap/dv.py:    a = Fold.Load("$A_FOLD", symbol="a") if "A_FOLD" in os.environ else None
    ./sysrap/dv.py:    b = Fold.Load("$B_FOLD", symbol="b") if "B_FOLD" in os.environ else None
    ./sysrap/xfold.py:    a = Fold.Load("$A_FOLD", symbol="a")
    ./sysrap/xfold.py:    b = Fold.Load("$B_FOLD", symbol="b")
    ./qudarap/tests/rayleigh_scatter_align.py:    t = Fold.Load(G4_FOLD)
    ./u4/tests/U4RecorderTest_ab.py:    a = Fold.Load("$A_FOLD", symbol="a", quiet=quiet) if "A_FOLD" in os.environ else None
    ./u4/tests/U4RecorderTest_ab.py:    b = Fold.Load("$B_FOLD", symbol="b", quiet=quiet) if "B_FOLD" in os.environ else None
    ./examples/UseGeometryShader/UseGeometryShader.py:    f = Fold.Load("$RECORD_FOLD",symbol="f")
    ./g4cx/tests/ab.py:    a = Fold.Load("$A_FOLD", symbol="a", quiet=quiet) if "A_FOLD" in os.environ else None
    ./g4cx/tests/ab.py:    b = Fold.Load("$B_FOLD", symbol="b", quiet=quiet) if "B_FOLD" in os.environ else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    s = Fold.Load("$S_FOLD", symbol="s") if not s_geom is None else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    t = Fold.Load("$T_FOLD", symbol="t") if not t_geom is None else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    u = Fold.Load("$U_FOLD", symbol="u") if not u_geom is None else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    v = Fold.Load("$V_FOLD", symbol="v") if not v_geom is None else None
    ./g4cx/tests/recplot.py:    a = Fold.Load("$A_FOLD", symbol="a")
    ./g4cx/tests/recplot.py:    b = Fold.Load("$B_FOLD", symbol="b")
    ./g4cx/tests/G4CXSimtraceTest.py:    a = Fold.Load("$A_FOLD", symbol="a")
    ./g4cx/tests/G4CXSimtraceTest.py:    b = Fold.Load("$B_FOLD", symbol="b")
    epsilon:opticks blyth$ 


The Fold object will have a simtrace member giving you the intersects::

    epsilon:opticks blyth$ opticks-py simtrace
    ./ana/framegensteps.py:        The way *local* is used implies that the simtrace genstep in q1 contains global centers, 
    ./ana/feature.py:        HMM: identity access is only fully applicable to simtrace, not photons
    ./ana/feature.py:        * TODO: accomodate the photon layout as well as the simtrace one by using 
    ./ana/feature.py:          * OR: standardize the flag/identity layout between photons and simtrace ?
    ./ana/feature.py:            232 QEVENT_METHOD void qevent::add_simtrace( unsigned idx, const quad4& p, const quad2* prd, float tmin )
    ./ana/feature.py:            254     simtrace[idx] = a ;
    ./ana/feature.py:        p = pos.simtrace  ## CAUTION : changed simtrace layout might not be accomodated yet 
    ./ana/feature.py:            ids = p[:,3,3].view(np.int32)   ## see sevent::add_simtrace a.q3.u.w = prd->identity() 
    ./ana/pvplt.py:def get_from_simtrace_isect( isect, mode="o2i"):
    ./ana/pvplt.py:    :param isect: (4,4) simtrace array item
    ./ana/pvplt.py:    dist = isect[0,3] # simtrace layout assumed, see CSG/tests/SimtraceRerunTest.cc
    ./ana/pvplt.py:def mpplt_simtrace_selection_line(ax, sts, axes, linewidths=2):
    ./ana/pvplt.py:    :param sts: simtrace_selection array of shape (n,2,4,4)  where n is small eg < 10
    ./ana/pvplt.py:    The simtrace_selection created in CSG/tests/SimtraceRerunTest.cc
    ./ana/pvplt.py:    contains pairs of isect the first from normal GPU simtrace and
    ./ana/pvplt.py:    log.info("mpplt_simtrace_selection_line sts\n%s\n" % repr(sts))
    ./ana/pvplt.py:                    o2i = get_from_simtrace_isect(isect, "o2i")
    ./ana/pvplt.py:                    o2i_XDIST = get_from_simtrace_isect(isect, "o2i_XDIST")
    ./ana/pvplt.py:                    nrm10 = get_from_simtrace_isect(isect, "nrm10")
    ./ana/pvplt.py:def pvplt_simtrace_selection_line(pl, sts):
    ./ana/pvplt.py:    print("pvplt_simtrace_selection_line sts\n", sts)
    ./ana/pvplt.py:                o2i = get_from_simtrace_isect(isect, "o2i")
    ./ana/pvplt.py:                nrm10 = get_from_simtrace_isect(isect, "nrm10")
    ./ana/simtrace_plot.py:        if hasattr(self, 'simtrace_selection'):
    ./ana/simtrace_plot.py:            sts = self.simtrace_selection
    ./ana/simtrace_plot.py:            mpplt_simtrace_selection_line(ax, sts, axes=self.frame.axes, linewidths=2)
    ./ana/simtrace_plot.py:        if hasattr(self, 'simtrace_selection'):
    ./ana/simtrace_plot.py:            sts = self.simtrace_selection
    ./ana/simtrace_plot.py:            pvplt_simtrace_selection_line(pl, sts)
    ./ana/simtrace_positions.py:    The isect, gpos used here come from qevent::add_simtrace
    ./ana/simtrace_positions.py:    def __init__(self, simtrace, gs, frame, local=True, mask="pos", symbol="t_pos" ):
    ./ana/simtrace_positions.py:        :param simtrace: "photons" array  
    ./ana/simtrace_positions.py:        The simtrace array is populated by:
    ./ana/simtrace_positions.py:        1. cx/CSGOptiX7.cu:simtrace 
    ./ana/simtrace_positions.py:        271 static __forceinline__ __device__ void simtrace( const uint3& launch_idx, const uint3& dim, quad2* prd )
    ./ana/simtrace_positions.py:        286     sim->generate_photon_simtrace(p, rng, gs, idx, genstep_id );
    ./ana/simtrace_positions.py:        300     evt->add_simtrace( idx, p, prd, params.tmin );
    ./ana/simtrace_positions.py:        410 SEVENT_METHOD void sevent::add_simtrace( unsigned idx, const quad4& p, const quad2* prd, float tmin )
    ./ana/simtrace_positions.py:        432     simtrace[idx] = a ;
    ./ana/simtrace_positions.py:        isect = simtrace[:,0]
    ./ana/simtrace_positions.py:        gpos = simtrace[:,1].copy()              # global frame intersect positions
    ./ana/simtrace_positions.py:        ##      np.all( t_pos.simtrace[:,:3] == t.simtrace[:,:3] ) 
    ./ana/simtrace_positions.py:        self.simtrace = simtrace 
    ./ana/simtrace_positions.py:                   "%s.simtrace %s " % (symbol, str(self.simtrace.shape)),
    ./ana/simtrace_positions.py:        self.simtrace = self.simtrace[mask]
    ./ana/simtrace_positions.py:        self.upos2simtrace = np.where(mask)[0]   # map upos indices back to simtrace indices before mask applied
    ./ana/simtrace_positions.py:        #t = self.simtrace[:,2,2]
    ./ana/simtrace_positions.py:        t = self.simtrace[:,0,3]
    ./CSGOptiX/cxt_min.py:for minimal simtrace plotting 
    ./CSGOptiX/cxt_min.py:    st = e.f.simtrace
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.py:* see notes/issues/simtrace-shakedown.rst
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.py:ISEL envvar selects simtrace geometry intersects by their features, according to frequency order
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.py:from opticks.ana.simtrace_positions import SimtracePositions
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.py:from opticks.ana.simtrace_plot import SimtracePlot
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.py:    SimtracePositions.Check(t.simtrace)
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.py:    t_pos = SimtracePositions(t.simtrace, gs, t.sframe, local=local, mask=MASK, symbol="t_pos" )
    ./CSG/tests/CSGNodeScanTest.py:from opticks.ana.pvplt import mpplt_simtrace_selection_line, mpplt_hist
    ./CSG/tests/CSGNodeScanTest.py:    s_simtrace = s.simtrace.reshape(-1,4,4)
    ./CSG/tests/CSGNodeScanTest.py:        s_hit = s_simtrace[:,0,3]>0 
    ./CSG/tests/CSGNodeScanTest.py:        s_pos = s_simtrace[s_hit][:,1,:3]
    ./CSG/tests/CSGSimtraceTest.py:from opticks.ana.pvplt import mpplt_simtrace_selection_line, mpplt_hist, mpplt_parallel_lines_auto, mpplt_add_shapes
    ./CSG/tests/CSGSimtraceTest.py:        s_hit = s.simtrace[:,0,3]>0 
    ./CSG/tests/CSGSimtraceTest.py:        s_pos = s.simtrace[s_hit][:,1,:3]
    ./CSG/tests/CSGSimtraceTest.py:        ## e = np.logical_and( s.simtrace[:,2,0] > 100., np.logical_and( s.simtrace[:,1,0] > 120. , s.simtrace[:,0,3]>0 )) 
    ./CSG/tests/CSGSimtraceTest.py:        ## e_ori = s.simtrace[e][:100,2,:3]
    ./CSG/tests/CSGSimtraceTest.py:        ## e_dir = s.simtrace[e][:100,3,:3]
    ./CSG/tests/CSGSimtraceTest.py:            w_label, w = "apex glancers",  np.logical_and( np.abs(s.simtrace[:,1,0]) < 220, np.abs(s.simtrace[:,1,2]-98) < 1 ) 
    ./CSG/tests/CSGSimtraceTest.py:            #w_label, w = "quadratic precision loss", np.logical_and( np.abs(s.simtrace[:,1,0] - (-214)) < 5, np.abs(s.simtrace[:,1,2] - (115)) < 5 )
    ./CSG/tests/CSGSimtraceTest.py:            w_simtrace = s.simtrace[w]
    ./CSG/tests/CSGSimtraceTest.py:            w_path = "/tmp/simtrace_sample.npy"
    ./CSG/tests/CSGSimtraceTest.py:            np.save(w_path, w_simtrace)
    ./CSG/tests/CSGSimtraceTest.py:            w = np.logical_and( np.abs(s.simtrace[:,1,2]) > 0.20 , s.simtrace[:,0,3]>0 )  
    ./CSG/tests/CSGSimtraceTest.py:            w_simtrace = s.simtrace[w][::10]
    ./CSG/tests/CSGSimtraceTest.py:        log.info("UNEXPECTED w_simtrace : %s " % str(w_simtrace.shape))
    ./CSG/tests/CSGSimtraceTest.py:        w_ori = w_simtrace[:,2,:3]
    ./CSG/tests/CSGSimtraceTest.py:        mpplt_simtrace_selection_line(ax, w_simtrace, axes=fr.axes, linewidths=2)
    ./CSG/tests/CSGSimtraceTest.py:    if not s is None and hasattr(s,"simtrace_selection"):
    ./CSG/tests/CSGSimtraceTest.py:       sts = s.simtrace_selection 
    ./CSG/tests/CSGSimtraceTest.py:        #w = np.logical_and( s.simtrace[:,0,3]>0, np.logical_and( s.simtrace[:,1,Z] > -38.9, s.simtrace[:,1,Z] < -20. ))
    ./CSG/tests/CSGSimtraceTest.py:        w = s.simtrace[:,1,X] > 264.5
    ./CSG/tests/CSGSimtraceTest.py:        sts = s.simtrace[w][:50]
    ./CSG/tests/CSGSimtraceTest.py:        mpplt_simtrace_selection_line(ax, sts, axes=fr.axes, linewidths=2)
    ./CSG/tests/CSGSimtraceTest.py:            a_hit = fold.simtrace[:,0,3]>0 
    ./CSG/tests/CSGSimtraceTest.py:            a_pos = a_offset + fold.simtrace[a_hit][:,1,:3]
    ./CSG/tests/CSGIntersectComparisonTest.py:from opticks.ana.pvplt import mpplt_simtrace_selection_line, mpplt_hist
    ./CSG/tests/CSGIntersectComparisonTest.py:    ab = s.a_simtrace - s.b_simtrace 
    ./CSG/tests/CSGIntersectComparisonTest.py:    a_simtrace = s.a_simtrace.reshape(-1,4,4)
    ./CSG/tests/CSGIntersectComparisonTest.py:    b_simtrace = s.b_simtrace.reshape(-1,4,4)
    ./CSG/tests/CSGIntersectComparisonTest.py:        a_hit = a_simtrace[:,0,3]>0 
    ./CSG/tests/CSGIntersectComparisonTest.py:        a_pos = a_simtrace[a_hit][:,1,:3]
    ./CSG/tests/CSGIntersectComparisonTest.py:        b_hit = b_simtrace[:,0,3]>0 
    ./CSG/tests/CSGIntersectComparisonTest.py:        b_pos = b_simtrace[b_hit][:,1,:3]
    ./CSG/tests/CSGIntersectComparisonTest.py:        #w = np.logical_and( np.abs(s.b_simtrace[:,1,0]) < 10. , np.abs(s.b_simtrace[:,1,2]) < 10. )  
    ./extg4/tests/X4IntersectVolumeTest.py:#from opticks.ana.simtrace_positions import SimtracePositions
    ./extg4/tests/X4IntersectVolumeTest.py:#from opticks.ana.simtrace_plot import SimtracePlot, pv, mp
    ./extg4/tests/X4IntersectVolumeTest.py:                ipos = isect.simtrace[:,1,:3] + tran[3,:3]     ## OOPS : ONLY TRANSLATES 
    ./extg4/tests/X4SimtraceTest.py:            a_hit = fold.simtrace[:,0,3]>0
    ./extg4/tests/X4SimtraceTest.py:            a_pos = a_offset + fold.simtrace[a_hit][:,1,:3]  ## no rotation, just translation
    ./sysrap/tests/SSimtrace_check.py:s = np.load("simtrace.npy")
    ./u4/tests/U4SimtraceSimpleTest.py:    The more general GPU simtrace deserves more attention than this one.
    ./u4/tests/U4SimtraceSimpleTest.py:            if not getattr(sf, 'simtrace', None) is None:
    ./u4/tests/U4SimtraceSimpleTest.py:                _lpos = sf.simtrace[:,1].copy()
    ./u4/tests/U4SimtraceSimpleTest.py:                 print("missing simtrace for soname:%s " % soname)
    ./u4/tests/U4SimtraceTest.py:    The more general GPU simtrace deserves more attention than this one.
    ./u4/tests/U4SimtraceTest.py:            if not getattr(sf, 'simtrace', None) is None:
    ./u4/tests/U4SimtraceTest.py:                _lpos = sf.simtrace[:,1].copy()
    ./u4/tests/U4SimtraceTest.py:                 print("missing simtrace for soname:%s " % soname)
    ./u4/tests/U4SimtracePlot.py:    inrm = t.simtrace[:,0].copy()  # normal at intersect (not implemented)
    ./u4/tests/U4SimtracePlot.py:    lpos = t.simtrace[:,1].copy()  # intersect position 
    ./u4/tests/U4SimtracePlot.py:    tpos = t.simtrace[:,2].copy()  # trace origin
    ./u4/tests/U4SimtracePlot.py:    tdir = t.simtrace[:,3].copy()  # trace direction
    ./g4cx/tests/G4CXSimtraceMinTest.py:G4CXSimtraceMinTest.py : simtrace plot backdrop with APID, BPID onephotonplot on top 
    ./g4cx/tests/G4CXSimtraceMinTest.py:    pp =  e.f.simtrace[:,1,:3]  
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    s_hit = s.simtrace[:,0,3]>0 if not s is None else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    t_hit = t.simtrace[:,0,3]>0 if not t is None else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    u_hit = u.simtrace[:,0,3]>0 if not u is None else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    v_hit = v.simtrace[:,0,3]>0 if not v is None else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    s_pos = s_offset + s.simtrace[s_hit][:,1,:3] if not s is None else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    t_pos = t_offset + t.simtrace[t_hit][:,1,:3] if not t is None else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    u_pos = u_offset + u.simtrace[u_hit][:,1,:3] if not u is None else None
    ./g4cx/tests/cf_G4CXSimtraceTest.py:    v_pos = v_offset + v.simtrace[v_hit][:,1,:3] if not v is None else None
    ./g4cx/tests/G4CXSimtraceTest.py:from opticks.ana.simtrace_positions import SimtracePositions
    ./g4cx/tests/G4CXSimtraceTest.py:from opticks.ana.simtrace_plot import SimtracePlot, pv, mp
    ./g4cx/tests/G4CXSimtraceTest.py:        log.info("RERUN envvar switched on use of simtrace_rerun from CSG/SimtraceRerunTest.sh " ) 
    ./g4cx/tests/G4CXSimtraceTest.py:        simtrace = t.simtrace_rerun
    ./g4cx/tests/G4CXSimtraceTest.py:        simtrace = t.simtrace
    ./g4cx/tests/G4CXSimtraceTest.py:    t_pos = SimtracePositions(simtrace, t_gs, t.sframe, local=local, mask=MASK, symbol="t_pos" )
    ./g4cx/tests/G4CXSimtraceTest.py:        j_kpos = t_pos.upos2simtrace[i_kpos]
    ./g4cx/tests/G4CXSimtraceTest.py:        log.info("j_kpos = t_pos.upos2simtrace[i_kpos]\n%s" % str(t_pos.upos2simtrace[i_kpos]) )
    ./g4cx/tests/G4CXSimtraceTest.py:        log.info("simtrace[j_kpos]\n%s" % str(simtrace[j_kpos]) )
    ./g4cx/tests/G4CXSimtraceTest.py:        simtrace_spurious = j_kpos
    ./g4cx/tests/G4CXSimtraceTest.py:        simtrace_spurious = []
    ./g4cx/tests/G4CXSimtraceTest.py:    ## created by CSG/SimtraceRerunTest.sh with SELECTION envvar picking simtrace indices to highlight 
    ./g4cx/tests/G4CXSimtraceTest.py:    if hasattr(t, "simtrace_selection") and SELECTION:  
    ./g4cx/tests/G4CXSimtraceTest.py:        plt.simtrace_selection = t.simtrace_selection
    ./g4cx/tests/G4CXSimtraceTest.py:    elif len(simtrace_spurious) > 0:
    ./g4cx/tests/G4CXSimtraceTest.py:        plt.simtrace_selection = simtrace[simtrace_spurious]
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 




What does ct.sh do ? 
-----------------------

ct.sh uses the CSGSimtraceTest binary which is using the Opticks CSG intersection code on the CPU.
It is not using U4Solid.



Comparing simtrace geometry
----------------------------


Trying to understand these python scripts will 
be much easier after you have written some of your own scripts. 

One random example is g4cx/tests/cf_G4CXSimtraceTest.py  
which can compare four folders s,t,u,v
with offsetting via envvar : as they will often be on top of each other.

Finding which bash function uses that::

    epsilon:opticks blyth$ opticks-sh cf_G4CXSimtraceTest
    ./g4cx/cf_gxt.sh:    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/cf_G4CXSimtraceTest.py 


The g4cx/cf_gxt.sh could be a starting point to comparing simtrace intersects. 
You will need to change it to match your FOLD directories.




