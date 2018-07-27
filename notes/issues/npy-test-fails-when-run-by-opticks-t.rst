FIXED npy-test-fails-when-run-by-opticks-t
============================================

FIXED by moving opticks-t to use om-test 

Getting 26 NPY fails only in opticks-t but not npy-t or om-test from top level. 
Suspect that opticks-t is using the old collective CMake ctest.

::

    npy-t () 
    { 
        opticks-t $(npy-bdir) $*
    }

    opticks-t () 
    { 
        opticks-t- $* --interactive-debug-mode 0
    }


Yes, removing some old CMake detritus from top level bdir, gives "No tests were found!!!"
from opticks-t and om-test works::

    epsilon:opticks blyth$ opticks-t
    Fri Jul 27 11:49:54 CST 2018
    Test project /usr/local/opticks/build
    No tests were found!!!
    Fri Jul 27 11:49:54 CST 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks/build/ctest.log



::

    92% tests passed, 26 tests failed out of 309

    Total Test time (real) = 174.35 sec

    The following tests FAILED:
         59 - NPYTest.NOpenMeshFindTest (Child aborted)
         87 - NPYTest.NSensorListTest (Child aborted)
         91 - NPYTest.NPartTest (Child aborted)
         94 - NPYTest.NHyperboloidTest (Child aborted)
         95 - NPYTest.NCubicTest (Child aborted)
         97 - NPYTest.NZSphereTest (Child aborted)
         98 - NPYTest.NBoxTest (SEGFAULT)
         99 - NPYTest.NBox2Test (Child aborted)
        100 - NPYTest.NNode2Test (SEGFAULT)
        101 - NPYTest.NSlabTest (SEGFAULT)
        104 - NPYTest.NDiscTest (Child aborted)
        105 - NPYTest.NConeTest (Child aborted)
        107 - NPYTest.NNodeTest (SEGFAULT)
        109 - NPYTest.NNodeTreeTest (Child aborted)
        110 - NPYTest.NNodePointsTest (SEGFAULT)
        115 - NPYTest.NTrianglesNPYTest (Child aborted)
        122 - NPYTest.NMarchingCubesNPYTest (Child aborted)
        124 - NPYTest.NCSGLoadTest (Child aborted)
        129 - NPYTest.NScanTest (Child aborted)
        143 - NPYTest.HitsNPYTest (Child aborted)
        146 - NPYTest.NImplicitMesherTest (Child aborted)
        147 - NPYTest.NDualContouringSampleTest (Child aborted)
        149 - NPYTest.NSceneTest (Child aborted)
        151 - NPYTest.NSDFTest (Child aborted)
        152 - NPYTest.NSceneLoadTest (Child aborted)
        153 - NPYTest.NSceneMeshTest (Child aborted)
    Errors while running CTest
    Fri Jul 27 11:27:13 CST 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks/build/ctest.log
    epsilon:opticks blyth$ 

