G4CXOpticks/index
==================

render
--------

gxr.sh
    G4CXRenderTest : G4CXOpticks::SetGeometry then render


simulate
---------

gxs.sh
    G4CXSimulateTest : G4CXOpticks::SetGeometry then simulate

gxs_ab.sh
    maybe replaced by ab.sh 

ab.sh
    python comparison of gxs fold


simtrace
---------

gxt.sh
    G4CXSimtraceTest : G4CXOpticks::SetGeometry then simtrace

mgxt.sh
    loop over geomlist setting GEOM and invoking gxt.sh 

cf_gxt.sh
    python comparison of simtrace from three geometries 

hama.sh nnvt.sh
    gxt.sh ana with view settings


collective
------------

gx.sh
    invokes gxs.sh gxt.sh gxr.sh one after the other 
 

