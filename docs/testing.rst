Opticks Testing, Geocache Creation, Python setup
===================================================


.. contents:: Table of Contents
   :depth: 2


Testing Installation with **opticks-t**
-------------------------------------------

The *opticks-t* function runs ctests for all the opticks projects::

    simon:opticks blyth$ opticks-
    simon:opticks blyth$ opticks-t
    Test project /usr/local/opticks/build
          Start  1: SysRapTest.SEnvTest
     1/65 Test  #1: SysRapTest.SEnvTest ........................   Passed    0.00 sec
          Start  2: SysRapTest.SSysTest
     2/65 Test  #2: SysRapTest.SSysTest ........................   Passed    0.00 sec
          Start  3: SysRapTest.SDigestTest
     3/65 Test  #3: SysRapTest.SDigestTest .....................   Passed    0.00 sec
    .....
    ..... 
          Start 59: cfg4Test.CPropLibTest
    59/65 Test #59: cfg4Test.CPropLibTest ......................   Passed    0.05 sec
          Start 60: cfg4Test.CTestDetectorTest
    60/65 Test #60: cfg4Test.CTestDetectorTest .................   Passed    0.04 sec
          Start 61: cfg4Test.CGDMLDetectorTest
    61/65 Test #61: cfg4Test.CGDMLDetectorTest .................   Passed    0.45 sec
          Start 62: cfg4Test.CG4Test
    62/65 Test #62: cfg4Test.CG4Test ...........................   Passed    5.06 sec
          Start 63: cfg4Test.G4MaterialTest
    63/65 Test #63: cfg4Test.G4MaterialTest ....................   Passed    0.02 sec
          Start 64: cfg4Test.G4StringTest
    64/65 Test #64: cfg4Test.G4StringTest ......................   Passed    0.02 sec
          Start 65: cfg4Test.G4BoxTest
    65/65 Test #65: cfg4Test.G4BoxTest .........................   Passed    0.02 sec

    100% tests passed, 0 tests failed out of 65

    Total Test time (real) =  59.89 sec
    opticks-ctest : use -V to show output


Creating a geocache geometry directory with **geocache-create**
-----------------------------------------------------------------

Many of the tests will fail in the absence of a geocache. 
Geometries are created using the *geocache-* bash functions.
To create a geometry use::

    geocache-              # run precursor function which defines the others 
    type geocache-create   # take a look at geocache-create
    geocache-create        # create geocache from GDML geometry file, default geometry is Dayabay near site detector

The bash functions essentially boil down to the below commandline 
with additional command line options for metadata recording:: 

    OKX4Test --deletegeocache --gdmlpath /path/to/your/detector.gdml

The **OKX4Test** executable loads a GDML file and translates it into an Opticks GGeo geometry 
instance and persists that into a geocache. 

Geocache creation is time and memory consuming, taking about 1 minute for the JUNO geometry 
on a workstation with lots of memory. Fortunately this only needs to be done once per geometry, and 
as the geocache is composed of binary .npy files they are fast to load and upload to the GPU.

Near to the end of the logging from geocache creation you should find output 
similar to the below which reports the OPTICKS_KEY value of the geometry::

    2019-07-01 16:14:08.129 INFO  [263983] [Opticks::reportGeoCacheCoordinates@755]  ok.idpath  /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
    2019-07-01 16:14:08.129 INFO  [263983] [Opticks::reportGeoCacheCoordinates@756]  ok.keyspec OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    2019-07-01 16:14:08.129 INFO  [263983] [Opticks::reportGeoCacheCoordinates@757]  To reuse this geometry: 
    2019-07-01 16:14:08.129 INFO  [263983] [Opticks::reportGeoCacheCoordinates@758]    1. set envvar OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    2019-07-01 16:14:08.129 INFO  [263983] [Opticks::reportGeoCacheCoordinates@759]    2. enable envvar sensitivity with --envkey argument to Opticks executables 
    2019-07-01 16:14:08.129 FATAL [263983] [Opticks::reportGeoCacheCoordinates@767] THE LIVE keyspec DOES NOT MATCH THAT OF THE CURRENT ENVVAR 
    2019-07-01 16:14:08.129 INFO  [263983] [Opticks::reportGeoCacheCoordinates@768]  (envvar) OPTICKS_KEY=NONE
    2019-07-01 16:14:08.129 INFO  [263983] [Opticks::reportGeoCacheCoordinates@769]  (live)   OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    2019-07-01 16:14:08.129 INFO  [263983] [Opticks::dumpRC@202]  rc 0 rcmsg : -


Opticks executables and scripts read the **OPTICKS_KEY** envvar to determine the geometry to load.
Add some lines to `~/.opticks_config` exporting the OPTICKS_KEY::

    export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce

The corresponding geocache directory is derived from this OPTICKS_KEY and can be found with the
*geocache-keydir* bash function (python and C++ equivalents are *opticks.ana.key/Key* and *optickscore/OpticksResource.cc*)::

    epsilon:ana blyth$ geocache-;geocache-keydir
    /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1



Inspect the geocache with **opticks-kcd**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bash function **opticks-kcd** changes directory to the geocache directory which is derived from the 
**OPTICKS_KEY** envvar.  Searching for NumPy arrays in the geocache reveals a large number of them.

::

    epsilon:1 blyth$ find . -name '*.npy' | wc -l 
        2065

    epsilon:1 blyth$ find . -name volume_transforms.npy 
    ./GMergedMesh/0/volume_transforms.npy
    ./GMergedMesh/6/volume_transforms.npy
    ./GMergedMesh/1/volume_transforms.npy
    ./GMergedMesh/4/volume_transforms.npy
    ./GMergedMesh/3/volume_transforms.npy
    ./GMergedMesh/2/volume_transforms.npy
    ./GMergedMesh/5/volume_transforms.npy
    ./GNodeLib/volume_transforms.npy


IPython session loadind the placement transform of a particular volume in the geometry::

    epsilon:1 blyth$ ipython

    In [1]: import numpy as np
    In [2]: vt = np.load("GNodeLib/volume_transforms.npy")
    In [3]: vt.shape
    Out[3]: (12230, 4, 4)
    In [3]: vt[3154]
    Out[3]: 
    array([[      0.54317,      -0.83962,       0.     ,       0.     ],
           [      0.83962,       0.54317,       0.     ,       0.     ],
           [      0.     ,       0.     ,       1.     ,       0.     ],
           [ -18079.453  , -799699.44   ,   -7100.     ,       1.     ]],
          dtype=float32)



Opticks Analysis and Debugging using Python, IPython, NumPy and Matplotlib : manage with **miniconda**
--------------------------------------------------------------------------------------------------------

Opticks uses the NumPy (NPY) buffer serialization format 
for geometry and event data, thus analysis and debugging requires
python and ipython with numpy and matplotib extensions.  
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



