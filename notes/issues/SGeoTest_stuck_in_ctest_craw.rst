SGeoTest_stuck_in_ctest_craw
===============================


ctest trying to run a removed test::

    epsilon:issues blyth$ opticks-f SGeoTest 
    ./sysrap/SGeo.hh:    ./sysrap/tests/SGeoTest.cc:#include "SGeo.hh"
    epsilon:opticks blyth$ 



::

     27/202 Test  #27: SysRapTest.SDigestNPTest .................................   Passed    0.02 sec
            Start  28: SysRapTest.SCFTest
     28/202 Test  #28: SysRapTest.SCFTest .......................................   Passed    0.02 sec
            Start  29: SysRapTest.SGeoTest
    Could not find executable SGeoTest
    Looked in the following places:
    SGeoTest
    SGeoTest
    Release/SGeoTest
    Release/SGeoTest
    Debug/SGeoTest
    Debug/SGeoTest
    MinSizeRel/SGeoTest
    MinSizeRel/SGeoTest
    RelWithDebInfo/SGeoTest
    RelWithDebInfo/SGeoTest
    Deployment/SGeoTest
    Deployment/SGeoTest
    Development/SGeoTest
    Development/SGeoTest
    Unable to find executable: SGeoTest

