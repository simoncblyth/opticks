Opticks Externals
====================


.. contents:: Table of Contents
   :depth: 2


Three type of externals : system, foreign and automated 
----------------------------------------------------------

1. **system** externals: NVIDIA GPU Driver, NVIDIA CUDA, NVIDIA OptiX must be installed 
   following the instructions from NVIDIA and assistance from your system administrator. 
   The envvars OPTICKS_CUDA_PREFIX and OPTICKS_OPTIX_PREFIX exported from `~/.opticks_config` communicate 
   the install locations to Opticks.
     
2. **foreign** externals are no longer automatically installed by `opticks-full` in order to facilitate integration of Opticks with detector 
   simulation frameworks.  

   The location of these foreign externals is communicated to the Opticks CMake machinery (`om-cmake` and `om-cmake-okconf`) 
   via the CMAKE_PREFIX_PATH envvar. This envvar and other PATH envvars such as LD_LIBRARY_PATH are 
   appended and exported by the `~/.opticks_config` lines starting `opticks-prepend-prefix`    

   These foreign externals can optionally be installed using::
 
        opticks-foreign    # list them, currently : boost, clhep, xercesc, g4
        opticks-foreign-install  

3. **automated** externals are automatically installed by `opticks-full`.  To list them::

        opticks-externals   # currently:  bcm, glm, glfw, glew, gleq, imgui, assimp, openmesh, plog, opticksaux, ... 


Brief description of foreign and automated externals
------------------------------------------------------

=================  =====================   ==============================================================================
precursor          pkg name                notes
=================  =====================   ==============================================================================
bcm-               BCM                     My fork of Boost CMake Modules, which eases use of modern CMake target import/export 
glm-               GLM                     OpenGL mathematics, 3D transforms 
glfw-              GLFW                    Interface between system and OpenGL, creating windows and receiving input
glew-              GLEW                    OpenGL extensions loading library, cmake build didnt work, includes vc12 sln for windows
gleq-              GLEQ                    Keyboard event handling header from GLFW author, header only
imgui-             ImGui                   OpenGL immediate mode GUI, depends on glfw and glew
assimp-            Assimp                  Assimp 3D asset importer, my fork that handles G4DAE extras
openmesh-          OpenMesh                basis for mesh navigation and fixing
plog-              PLog                    Header only logging, supporting multi dll logging on windows 
oimplicitmesher-   ImplicitMesher          Polygonization of implicitly defined shapes
odcs-              DualContouringSample    Alternate polygonization using Octree for multi-resolution, however its slow
oyoctogl-          YoctoGL                 Used for glTF geometry file format handling, parsing/serializing    
ocsgbsp-           CSGBSP                  Another BSP approach to polygonization under investigation
xercesc            XercesC                 XML handling dependency of Geant4, required for GDML parsing
g4-                Geant4                  The preeminent simulation toolkit
optickaux-         -                       Some GDML geometry files for testing Opticks      
=================  =====================   ==============================================================================


Former externals that are no longer in use
-------------------------------------------

=================  =====================   ==============================================================================
precursor          pkg name                notes
=================  =====================   ==============================================================================
opticksdata-       -                       Dayabay G4DAE and GDML geometry files for testing Opticks      
=================  =====================   ==============================================================================



Separate installation of externals : useful for debugging the build
----------------------------------------------------------------------

The *opticks-externals* function lists current precursor names, *opticks-externals-install* runs each 
of the precursor functions in turn.  To rerun a single external install, use the below pattern of running 
the precursor function and then the installer function.

::

   oyoctogl-
   oyoctogl--

After installation has been done rerunning *opticks-externals-install* completes quickly,
and does no harm.



Foreign Externals
------------------

Listing the foreign externals with bash function **opticks-foreign**::

    epsilon:opticks blyth$ opticks-foreign
    boost
    clhep
    xercesc
    g4


What these do:

boost
    system, program_options, filesystem, regex
clhep
    optionally needed by geant4 
xercesc
    XML parsing needed by g4 GDML functionality
g4
    geant4 


Automated Externals
--------------------

Listing the automated externals with bash function **opticks-externals**::

    epsilon:opticks blyth$ opticks-externals
    bcm
    glm
    glfw
    glew
    gleq
    imgui
    assimp
    openmesh
    plog
    opticksaux
    oimplicitmesher
    odcs
    oyoctogl
    ocsgbsp


What these do is described in the below sections.


Base externals
----------------

bcm
    boost CMake modules, target export/import for CMake 3.5+ 
    allows config to direct dependencies only, the rest of the tree
    gets configured automatically  
glm
    vector, matrix, 3D projection mathematics
plog
    logging   
oyoctogl
    glTF 2.0 3D file format parsing/construction, json parsing 
opticksaux
    git repository containing example GDML files 


Visualization externals
-------------------------

glfw
    cross platform OpenGL and system events : keyboard, mouse  
gleq
    event queue for glfw  
glew
    OpenGL extension wrangler, providing access to OpenGL symbols 
imgui
    immediate mode OpenGL GUI     


Mesh manipulation and polygonization externals (planned to be eliminated)
-------------------------------------------------------------------------

All these are not needed with direct from G4 workflow, they 
are used for mesh manipulation and polygonization functionality.


assimp
    used for COLLADA DAE file format loading  
    (not needed with direct from G4 workflow) 
openmesh
    provides mesh traversal 
oimplicitmesher
    creating meshes from SDF analytic geometries
odcs
    dual-contouring-sample  
ocsgbsp
    polygonization
     
     
Former externals
--------------------

opticksdata
    common repository for geometry 
    (not needed with direct from G4 workflow) 



Boost C++ Libraries
----------------------

The Boost components listed in the table need to be installed.
These are widely available via package managers. Use the standard one for 
your system. The FindBoost.cmake provided with cmake is used to locate the installation.

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
boost                  boost-           Boost           components: system thread program_options log log_setup filesystem regex 
=====================  ===============  =============   ==============================================================================


The recommended minimum boost version is 1.53 as that is what I am using. 
You might be able to survive with an earlier version, 
but anything before 1.41 is known not to work. 


Updating Boost 
~~~~~~~~~~~~~~~~

If your version of Boost is not recent enough the cmake configuring 
step will yield errors like the below.::

      CMake Error at /home/blyth/local/env/tools/cmake/cmake-3.5.2-Linux-x86_64/share/cmake-3.5/Modules/FindBoost.cmake:1657 (message):
      Unable to find the requested Boost libraries.

      Boost version: 1.41.0

If possible use your system package manager to update Boost. If that is 
not possible then do a local Boost install.  Opticks includes bash functions
starting *boost-* that can get and install Boost locally.




Opticks Pre-requisites : NVIDIA OptiX and NVIDIA CUDA 
-----------------------------------------------------------

OptiX requires your system to have a fairly recent NVIDIA GPU of CUDA compute capability 3.0 at least.

To download OptiX you need to join the NVIDIA Developer Program.  
Use the links in the table to register, it is free but may take a few days to be approved.
Follow the NVIDIA instructions to download and install CUDA and OptiX. 
Thrust is installed together with CUDA. 

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
cuda                   cuda-            CUDA            https://developer.nvidia.com/cuda-downloads (includes Thrust)
optix                  optix-           OptiX           https://developer.nvidia.com/optix
=====================  ===============  =============   ==============================================================================

CUDA installation guides:

* http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
* http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html


Finding CUDA
~~~~~~~~~~~~~

Opticks uses the `FindCUDA.cmake` supplied by CMake to, eg 
on macOS at `/opt/local/share/cmake-3.12/Modules/FindCUDA.cmake`.  
Quoting from that::

   29 # The script will prompt the user to specify ``CUDA_TOOLKIT_ROOT_DIR`` if
   30 # the prefix cannot be determined by the location of nvcc in the system
   31 # path and ``REQUIRED`` is specified to :command:`find_package`. 


Thus check that `nvcc` is in your PATH, and preferably compile some CUDA examples
on your system before installing Opticks.:: 

    epsilon:opticks blyth$ which nvcc    # macOS
    /Developer/NVIDIA/CUDA-9.1/bin/nvcc

    [blyth@localhost ~]$ which nvcc   # Linux
    /usr/local/cuda-9.2/bin/nvcc


Versions of CUDA and OptiX 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I recommend you start your installation attempt with OptiX 6.5
together with the version of CUDA that it was built against, as stated in 
the OptiX release notes. 
This version pinning between CUDA and OptiX is because Opticks links against 
both the OptiX library and the CUDA runtime.

If you cannot use the latest CUDA (because of kernel incompatibility) you will need to
use an older OptiX version contemporary with the CUDA version that your kernel supports.

Version combinations that have been used:

*current*
   CUDA 10.1, OptiX 6.5.0

*previous*
   CUDA 9.1, OptiX 5.1.1
   CUDA 9.1, OptiX 5.1.0

*earlier* 
   CUDA 7.0, OptiX 3.80


The *current* version combination is regularly tested, the *previous* one
relies on your bug reports https://groups.io/g/opticks/topics to keep it working. 
Any issues with *earlier* version combinations will not be addressed.  


The reason for the extremes of caution regarding version combinations of drivers 
is that the interface to the GPU is via kernel extensions where if anything goes 
wrong there is no safety net. A bad kernel extension will cause kernel panics, 
your machine crashes and continues to crash until the bad driver is removed 
(on macOS the removal can be done by resetting NVRAM).


NVIDIA Driver Versions
~~~~~~~~~~~~~~~~~~~~~~~~


   ========  ===============  ===================  ============================================  =================================== 
    OptiX     Date              Driver (Linux)       Working                                       Problems Reported
   ========  ===============  ===================  ============================================  ===================================
     6.5.0     Aug 26, 2019       435.17             435.21 CUDA 10.1  TITAN RTX, TITAN V          Sajan: 440.33.01 and CUDA 10.2
     7.0.0     July 29, 2019      435.12
     6.0.0     Feb 2018           418.30 
   ========  ===============  ===================  ============================================  ===================================


The release notes from every version of OptiX states the 
required minimum version of the NVIDIA Driver that must be used
for that version of OptiX. In recent releases that driver version 
has been from the so called short-lived series. 

From https://www.nvidia.com/en-gb/drivers/unix/ on April 29, 2020::

   Latest Long Lived Branch version: 440.82
   Latest Short Lived Branch version: 435.21

Note that the long-lived series may have version numbers that exceed those 
of the short-lived series but the features needed for OptiX take much longer
to appear in that series. 

The releases from the longer-lived driver branches are intended for 
users who do not need the latest and greatest features. 

If you cannot change your driver version this sometimes means that 
an older version of OptiX must be used to work with your driver.
 

OptiX 6.5.0 (August 26, 2019)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quote from release notes::

   OptiX 6.5.0 requires that you install the 436.02 driver on Windows or the 435.17 Driver for linux. Operating System:

   * Windows 7/8.1/10 64-bit
   * Linux RHEL 4.8+ or Ubuntu 10.10+ 64-bit


Problems with OptiX 6.5.0, Driver 440.33.01, CUDA 10.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.nvidia.com/en-gb/drivers/unix/

Search the OptiX forum for "driver" and looking for Linux reports:

* https://forums.developer.nvidia.com/search?q=driver%20%20category%3A167
* https://forums.developer.nvidia.com/t/optix7-and-game-ready-driver-440-97/83907

From droettger of NVIDIA on Nov 20, 2019 regarding a problem with optix7::

    Yes, there was a bug in R440 drivers and a serious test escape.

    This has been fixed in the meantime. The just released Windows driver 441.28 has picked up the fix already.
    Unfortunately Linux 440.31 drivers have been cut before the fix. The next 441 Linux drivers should have it.

    If you already hacked that integer field to 0 when preTransform is null, the expected value when preTransform is actually containing 3x4 matrices is 0x21E1.



OptiX 7.0.0 (July 29, 2019)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**An entirely new API : not yet supported by Opticks.**

Quote from release notes::

   OptiX 7.0.0 requires that you install the 435.80 driver on Windows or the 435.12 Driver for linux. 
   Note OptiX dll from the SDK are no longer needed since the symbols are loaded from the driver.

   * Windows 7/8.1/10 64-bit; 
   * Linux RHEL 4.8+ or Ubuntu 10.10+ 64-bit


OptiX 6.0.0 (February 2018)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**first version of OptiX with support for Turing GPUs and RT Cores**

Quote from release notes::

   Graphics Driver:

   * Windows: driver version 418.81 or later is required.
   * Linux: driver version 418.30 or later is required.

   OS:

   * Windows 7/8.1/10 64-bit
   * Linux RHEL 4.8+ or Ubuntu 10.10+ 64-bit


Using a non-standard OptiX version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Opticks tends to adopt new OptiX versions very soon after they become
available approximately twice per year. This is because OptiX continues 
to improve rapidly. Updating OptiX typically also requires an update of 
the CUDA version and the NVIDIA driver.  

Sometimes due to the suitable NVIDIA driver not yet being installed by 
your system admin it is necessary to use older OptiX+CUDA versions.  
Opticks aims to allow this for recent version combinations only.


Building Opticks against "foreign" externals such as Geant4, Boost
-------------------------------------------------------------------

When integrating Opticks with a detector simulation framework 
it is important that externals that are in common between Opticks and the framework
are one and the same to avoid symbol inconsistency between different versions of libraries. 
The most likely packages to be in common are::

     Boost 
     Geant4 
     XercesC
     GLEW  

The Opticks build is sensitive to the CMAKE_PREFIX_PATH envvar 
allowing Opticks to build against "foreign" externals. To check which external that
the CMake based build will pick use the find_package.py script::  

    epsilon:opticks blyth$ find_package.py Geant4
    Geant4                         : /usr/local/foreign/lib/Geant4-10.5.1/Geant4Config.cmake 
    Geant4                         : /usr/local/opticks/externals/lib/Geant4-10.4.2/Geant4Config.cmake 

If you can integrate Opticks with your framework using CMake then the non-CMake 
opticks-config system which is based on pkg-config pc files is not relevant to you. 
Conversely if you need to integrate with legacy build systems such as CMT 
then it is necessary to arrange consistency between the two config systems.
To check which external that non-CMake pkg-config based builds will pick use the 
pkg_config.py script::

    epsilon:~ blyth$ pkg_config.py Geant4
    geant4                         : /usr/local/foreign/lib/pkgconfig/geant4.pc 
    geant4                         : /usr/local/opticks/externals/lib/pkgconfig/geant4.pc  

And also directly with opticks-config (or shorthand oc)::

     opticks-config --cflags Geant4
     opticks-config --help

To keep consistency between the CMake and pkg-config configuration systems it is 
necessary for do several things:

1. ensure that the original CMAKE_PREFIX_PATH and PKG_CONFIG_PATH are consistent, for example::

    export CMAKE_PREFIX_PATH=/usr/local/foreign
    export PKG_CONFIG_PATH=/usr/local/foreign/lib/pkgconfig
    
2. ensure that pc files are present for relevant packages in the lib/pkgconfig 
   or lib64/pkgconfig directories beneath all relevant prefix dirs, for example::

    /usr/local/foreign/lib/pkgconfig/geant4.pc  
    /usr/local/foreign/lib/pkgconfig/boost.pc  

3. generate any missing pc files with::

      g4-pcc-all
      boost-pcc-all

   These use find_package.py which iterates over prefixes in CMAKE_PREFIX_PATH
   writing .pc files.

4. check consistency with::

    find_package.py Boost 
    pkg_config.py Boost 

5. do cleaninstalls following changes to CMAKE_PREFIX_PATH and PKG_CONFIG_PATH with::

    cd ~/opticks
    om-
    om-cleaninstall



 
Testing CUDA and OptiX Installs and nvcc toolchain
-------------------------------------------------------

Before trying to install Opticks check your CUDA and OptiX installs:

1. run the precompiled CUDA and OptiX sample binaries
2. compile the CUDA and OptiX samples
3. run your compiled samples

Testing Thrust
----------------

Thrust provides a higher level C++ template approach to using CUDA that is used extensively 
by Opticks. The Thrust headers are installed by the CUDA toolkit installater, eg at `/usr/local/cuda/include/thrust`.
You are recommended to try some of the Thrust examples to check your nvcc toolchain.

* http://docs.nvidia.com/cuda/thrust/index.html
* https://github.com/thrust/thrust/tree/master/examples


Geant4
---------

Geant4 is no longer automatically installed by *opticks-full* it can be installed with *g4--*.
The *g4-* precursor selects a version of Geant4.  Currently a bit dated, this is intended to be brought uptodate sometime.
The coupling between Opticks and Geant4 is intended to be weak : so a range of 
recent versions of Geant4 are intended to be supported.
 
