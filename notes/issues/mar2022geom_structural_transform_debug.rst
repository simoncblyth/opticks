mar2022geom_structural_transform_debug 
==========================================

JUNO Renders cxr OptiX 7 renders now looking to use wrong global transforms : chimney in middle of LS
--------------------------------------------------------------------------------------------------------

I have seen something similar before which was due to the --gparts_transform_offset being 
omitted. But that seems not to be the case this time. 


Test with simple geometry : a grid of PMTs
-----------------------------------------------------

To setup a simpler environment to debug structural transforms and their conversion, 
get PMTSim::GetPV operational and use it from a new test 

* g4ok/tests/G4OKPMTSimTest.cc
* g4ok/G4OKPMTSimTest.sh

::


     01 #!/bin/bash -l 
      2 
      3 export X4PhysicalVolume=INFO
      4 export GInstancer=INFO
      5 
      6 # comment the below to create an all global geometry, uncomment to instance the PMT volume 
      7 export GInstancer_instance_repeat_min=25
      8 
      9 G4OKPMTSimTest
     10 
     11 
     12 


* used this to create a geocache
* grabbed the OPTICKS_KEY and then created CSG_GGeo CSGFoundry geometry
* rendered it with CSGOptiX cxr_overview.sh : no surprises get a grid of PMTs within the world box

::

    PUB=simple_transform_check EYE=1,0,0 ZOOM=1 ./cxr_overview.sh 



With both an instanced and an all global geometry get the expected render of a grid of PMTs in a box. 





