OPTIX_VERSION_branching
=========================


Review OPTIX_VERSION branching and rationalize 
to make handling new versions easier.

With dead code and examples skipped review use of OPTIX_VERSION macro::

    epsilon:CSGOptiX blyth$ opticks-fl OPTIX_VERSION

    ./CSGOptiX/PIP.cc 
         PIP::OptiXVersionIsSupported

    ./CSGOptiX/CMakeLists.txt


    ./CSGOptiX/Params.h
    ./CSGOptiX/CSGOptiX.h
         handle placeholder plus Six compilation
         

    ./CSGOptiX/BI.h
         API 70000 customPrimitiveArray change

    ./CSGOptiX/tests/CSGOptiXVersion.cc
    ./CSGOptiX/tests/CSGOptiXVersionTest.cc
         

    ./CSGOptiX/CSGOptiX.cc
    ./CSGOptiX/Params.cc
         Six compilation

    ./CSGOptiX/OPT.h
         changed to be more liberal with version 

    ./CSG/csg_intersect_tree.h
          removed

    ./cmake/Modules/FindOptiX.cmake
    ./cmake/Modules/FindOpticksOptiX.cmake
    ./externals/optix.bash
    ./opticks.bash




