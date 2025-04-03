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
 


Look at Simtrace tests like U4SimtraceTest.sh



Selected Scripts with "Simtrace" code : likely easier to revive
--------------------------------------------------------------------

::

    ./CSGOptiX/cxt_min.sh
    ./cxt_min.sh      ## symbolic link to the above




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

    ./GeoChain/grab.sh

    ./bin/geomlist.sh
    ./bin/geomlist_test.sh
    ./bin/log.sh

    ./cxt_min.sh

    ./extg4/cf_x4t.sh
    ./extg4/ct_vs_x4t.sh
    ./extg4/mx4t.sh
    ./extg4/x4t.sh
    ./extg4/xxs.sh

    ./g4cx/cf_gxt.sh
    ./g4cx/gxt.sh
          ## using G4CXSimtraceTest.cc G4CXOpticks::simtrace QSim::simtrace  CSGOptiX::simtrace_launch
          ## LOTS OF WORK NEEDED TO REVIVE

    ./g4cx/tests/G4CXTest.sh

    ./sysrap/tests/SSimtrace_check.sh
    ./u4/tests/FewPMT.sh
    ./u4/tests/U4SimtracePlot.sh
    ./u4/tests/U4SimtraceTest.sh
    ./u4/tests/U4SimtraceTest_one_pmt.sh
    ./u4/tests/U4SimtraceTest_two_pmt.sh
    ./u4/tests/U4SimtraceTest_two_pmt_cf.sh
    ./u4/tests/U4SimulateTest.sh
    ./u4/tests/viz.sh
    P[blyth@localhost opticks]$ 


