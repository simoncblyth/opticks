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
 
        opticks-foreign    # list them, currently : clhep, xercesc, g4
        opticks-foreign-install  

3. **automated** externals are automatically installed by `opticks-full`.  To list them::

        opticks-externals   # currently:  bcm, glm, glfw, glew, gleq, imgui, plog, nljson


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
plog-              PLog                    Header only logging, supporting multi dll logging on windows 
xercesc            XercesC                 XML handling dependency of Geant4, required for GDML parsing
g4-                Geant4                  The preeminent simulation toolkit
=================  =====================   ==============================================================================



Separate installation of externals : useful for debugging the build
----------------------------------------------------------------------

The *opticks-externals* function lists current precursor names, *opticks-externals-install* runs each 
of the precursor functions in turn.  To rerun a single external install, use the below pattern of running 
the precursor function and then the installer function.

::

   glew-
   glew--

After installation has been done rerunning *opticks-externals-install* completes quickly,
and does no harm.



Foreign Externals
------------------

Listing the foreign externals with bash function **opticks-foreign**::

    epsilon:opticks blyth$ opticks-foreign
    clhep
    xercesc
    g4


What these do:


clhep
    optionally needed by geant4 
xercesc
    XML parsing needed by g4 GDML functionality
g4
    geant4 


Former Foreign Externals
-------------------------

Dependency on boost has been removed. 

boost
    system, program_options, filesystem, regex


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
    plog


What these do is described in the below sections.


Base externals
----------------

bcm
    boost CMake modules (mis-named: not really boost related), 
    target export/import for CMake 3.5+ allows config to direct 
    dependencies only, the rest of the tree gets configured automatically  
glm
    vector, matrix, 3D projection mathematics
plog
    logging   


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

I recommend you start your installation attempt with OptiX 7.0 or 7.5 or 8.0
(depending on your NVIDIA Driver version) 
together with the version of CUDA that it was built against, as stated in 
the OptiX release notes. 
This version pinning between CUDA and OptiX is because Opticks links against 
both OptiX and the CUDA runtime.

If you cannot use the latest CUDA (because of kernel incompatibility) you will need to
use an older OptiX version contemporary with the CUDA version that your kernel supports.


The reason for the extremes of caution regarding version combinations of drivers 
is that the interface to the GPU is via kernel extensions where if anything goes 
wrong there is no safety net. A bad kernel extension will cause kernel panics, 
your machine crashes and continues to crash until the bad driver is removed 


NVIDIA Driver Versions
~~~~~~~~~~~~~~~~~~~~~~~~



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
 


Building Opticks against "foreign" externals such as Geant4
-------------------------------------------------------------------

When integrating Opticks with a detector simulation framework 
it is important that externals that are in common between Opticks and the framework
are one and the same to avoid symbol inconsistency between different versions of libraries. 
The most likely packages to be in common are::

     Geant4 
     XercesC
     GLEW  

The Opticks build is sensitive to the CMAKE_PREFIX_PATH envvar 
allowing Opticks to build against "foreign" externals. To check which external that
the CMake based build will pick use the find_package.py script::  

    epsilon:opticks blyth$ find_package.py Geant4
    Geant4                         : /usr/local/foreign/lib/Geant4-10.5.1/Geant4Config.cmake 
    Geant4                         : /usr/local/opticks/externals/lib/Geant4-10.4.2/Geant4Config.cmake 




Former pkg-config based opticks-config builds (eg for CMT) are no longer maintained and the infrastructure will be removed 
----------------------------------------------------------------------------------------------------------------------------

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
 




