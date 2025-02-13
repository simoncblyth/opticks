Opticks Testing, Geocache Creation, Python setup
===================================================


.. contents:: Table of Contents
   :depth: 2


Testing Installation with **opticks-t**
-------------------------------------------

The *opticks-t* function runs ctests for all the opticks packages::



    N[blyth@localhost opticks]$ opticks-
    N[blyth@localhost opticks]$ opticks-t
    === qudarap-check-installation : /home/blyth/.opticks/rngcache/RNG rc 0
    === qudarap-check-installation : /home/blyth/.opticks/rngcache/RNG/QCurandState_1000000_0_0.bin rc 0
    === qudarap-check-installation : /home/blyth/.opticks/rngcache/RNG/QCurandState_3000000_0_0.bin rc 0
    === qudarap-check-installation : /home/blyth/.opticks/rngcache/RNG/QCurandState_10000000_0_0.bin rc 0
    === opticks-check-installation : rc 0
    === om-env : normal running
    ...

      21 /30  Test #21 : U4Test.U4Material_MakePropertyFold_MakeTest   Passed                         2.10   
      22 /30  Test #22 : U4Test.U4Material_MakePropertyFold_LoadTest   Passed                         0.16   
      23 /30  Test #23 : U4Test.U4TouchableTest                        Passed                         0.15   
      24 /30  Test #24 : U4Test.U4SurfaceTest                          Passed                         0.17   
      25 /30  Test #25 : U4Test.U4SolidTest                            Passed                         0.16   
      26 /30  Test #26 : U4Test.U4SensitiveDetectorTest                Passed                         0.16   
      27 /30  Test #27 : U4Test.U4Debug_Test                           Passed                         0.15   
      28 /30  Test #28 : U4Test.U4Hit_Debug_Test                       Passed                         0.16   
      29 /30  Test #29 : U4Test.G4ThreeVectorTest                      Passed                         0.17   
      30 /30  Test #30 : U4Test.U4PhysicsTableTest                     Passed                         0.16   

    CTestLog :             CSGOptiX :      0/     4 : 2024-02-01 11:03:06.513634 : /home/blyth/junotop/ExternalLibs/opticks/head/build/CSGOptiX/ctest.log 
      1  /4   Test #1  : CSGOptiXTest.CSGOptiXVersion                  Passed                         0.05   
      2  /4   Test #2  : CSGOptiXTest.CSGOptiXVersionTest              Passed                         0.04   
      3  /4   Test #3  : CSGOptiXTest.CSGOptiXRenderTest               Passed                         1.47   
      4  /4   Test #4  : CSGOptiXTest.ParamsTest                       Passed                         0.04   

    CTestLog :                 g4cx :      0/     2 : 2024-02-01 11:03:10.025646 : /home/blyth/junotop/ExternalLibs/opticks/head/build/g4cx/ctest.log 
      1  /2   Test #1  : G4CXTest.G4CXRenderTest                       Passed                         1.49   
      2  /2   Test #2  : G4CXTest.G4CXOpticks_setGeometry_Test         Passed                         1.86   


    LOGS:
    CTestLog :               okconf :      0/     4 : 2024-02-01 11:02:10.441449 : /home/blyth/junotop/ExternalLibs/opticks/head/build/okconf/ctest.log 
    CTestLog :               sysrap :      0/   107 : 2024-02-01 11:02:18.513475 : /home/blyth/junotop/ExternalLibs/opticks/head/build/sysrap/ctest.log 
    CTestLog :                  ana :      0/     1 : 2024-02-01 11:02:18.682476 : /home/blyth/junotop/ExternalLibs/opticks/head/build/ana/ctest.log 
    CTestLog :             analytic :      0/     1 : 2024-02-01 11:02:18.853477 : /home/blyth/junotop/ExternalLibs/opticks/head/build/analytic/ctest.log 
    CTestLog :                  bin :      0/     1 : 2024-02-01 11:02:19.015477 : /home/blyth/junotop/ExternalLibs/opticks/head/build/bin/ctest.log 
    CTestLog :                  CSG :      0/    42 : 2024-02-01 11:02:41.095550 : /home/blyth/junotop/ExternalLibs/opticks/head/build/CSG/ctest.log 
    CTestLog :              qudarap :      0/    20 : 2024-02-01 11:02:50.496581 : /home/blyth/junotop/ExternalLibs/opticks/head/build/qudarap/ctest.log 
    CTestLog :                gdxml :      0/     1 : 2024-02-01 11:02:50.722582 : /home/blyth/junotop/ExternalLibs/opticks/head/build/gdxml/ctest.log 
    CTestLog :                   u4 :      0/    30 : 2024-02-01 11:03:04.775629 : /home/blyth/junotop/ExternalLibs/opticks/head/build/u4/ctest.log 
    CTestLog :             CSGOptiX :      0/     4 : 2024-02-01 11:03:06.513634 : /home/blyth/junotop/ExternalLibs/opticks/head/build/CSGOptiX/ctest.log 
    CTestLog :                 g4cx :      0/     2 : 2024-02-01 11:03:10.025646 : /home/blyth/junotop/ExternalLibs/opticks/head/build/g4cx/ctest.log 


    SLOW: tests taking longer that 15 seconds
    ...

    FAILS:  0   / 213   :  Thu Feb  1 11:03:10 2024   




Many tests will fail until you convert and configure a geometry
----------------------------------------------------------------

You will get many opticks-t test fails until you convert a geometry
and configure the test executables to use the geometry. 
An example script to try to convert your detector GDML to an Opticks 
CSGFoundry geometry is::

   ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh

There is a long comment in this script explaining it and the standard 
config approach via your ~/.opticks/GEOM/GEOM.sh.
You can also see forum threads like the below which explain doing this::

    https://groups.io/g/opticks/topic/which_tests_should_i_run/103068306

If you succeed then you should get about 218 opticks-t tests 
to pass and no fails. 



Opticks Analysis and Debugging using Python, IPython, NumPy and Matplotlib : manage with **miniconda**
--------------------------------------------------------------------------------------------------------

Opticks uses the NumPy (NPY) buffer serialization format 
for geometry and event data, thus analysis and debugging requires
python and ipython with numpy and matplotib extensions.  
Optionally pyvista/VTK is useful for fast 3D plotting of large datasets. 
For management of these and other python packages it is 
convenient to use miniconda.

* https://docs.conda.io/en/latest/miniconda.html

Opticks is in the process of migrating scripts from python2 to python3, 
currently using Python 3.7.8. Please report problems from unmigrated scripts.

After installing miniconda the additional packages can be installed with
commands such as::

    conda install ipython 
    conda install numpy 
    conda install sympy 
    conda install matplotlib

In addition to your PATH you can also control which python Opticks
uses with the optional OPTICKS_PYTHON envvar, for example::

    export OPTICKS_PYTHON=python3 



