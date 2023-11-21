CSGOptiX/tests
==================


Overview
---------

The minimal mains aim to rationalize API and make 
render/simtrace/simulate APIs more consistent. 
Long term aiming to replace the old tests. 


Minimal Main tests using CSGOptiX::RenderMain SimtraceMain SimulateMain
--------------------------------------------------------------------------

CSGOptiXSMTest.cc
   CSGOptiX::SimulateMain used by cxs_min.sh
   reviving simulation, quick cycle preparation for opticksMode 1,3 in junosw

CSGOptiXTMTest.cc
   CSGOptiX::SimtraceMain
   WIP: reviving simtrace

CSGOptiXRMTest.cc
   CSGOptiX::RenderMain used by cxr_min.sh 


Old Rendering 
---------------

CSGOptiXRenderTest.cc

Old Simtrace 2D
-----------------

CSGOptiXSimtraceTest.cc

Old Simulation
---------------

CSGOptiXSimTest.cc
   CarrierGenstep QSim::simulate call 

CSGOptiXSimulateTest.cc
   CarrierGenstep, local geom

   * TODO: generalize to input photons and any geom 
   * TODO: very simular to the above, consolidate after generalization
 

CXRaindropTest.cc
   Specific geometry


Utility 
--------------

CSGOptiXVersion.cc
CSGOptiXVersionTest.cc

