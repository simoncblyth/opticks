float-precision-saabb-still-being-used
=========================================


Doing more in double precision WITH_S_BB seems
to make no difference to the small bbox shifts.  

::

    epsilon:sysrap blyth$ opticks-f saabb.h 
    ./CSG/CSGFoundry.h:#include "saabb.h"

         AABB VERY MUCH IN USE

    ./CSG/tests/CSGFoundry_addPrimNodes_Test.cc:#include "saabb.h"
    ./CSG/tests/DemoGrid.cc:#include "saabb.h"
    ./sysrap/CMakeLists.txt:    saabb.h
    ./sysrap/tests/sqat4Test.cc:#include "saabb.h"
    ./sysrap/tests/saabbTest.cc:#include "saabb.h"
    epsilon:opticks blyth$ 


