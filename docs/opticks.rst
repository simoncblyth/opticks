Opticks Install Instructions
==================================

Get Opticks 
------------

Clone the repository from bitbucket::

   cd 
   hg clone http://bitbucket.org/simoncblyth/opticks 
   hg clone ssh://hg@bitbucket.org/simoncblyth/opticks   # via SSH for developers 



Bash Shell Setup with .opticks_setup
---------------------------------------

Example `~/.opticks_setup`:

.. code-block:: sh

    # .opticks_setup

    export LOCAL_BASE=$HOME/local       
    export OPTICKS_HOME=$HOME/opticks
    export PYTHONPATH=$HOME

    opticks-(){  [ -r $OPTICKS_HOME/opticks.bash ] && . $OPTICKS_HOME/opticks.bash && opticks-env $* ; }
    opticks-     ##  

    o(){ cd $(opticks-home) ; hg st ; }
    op(){ op.sh $* ; }

    PATH=$OPTICKS_HOME/bin:$OPTICKS_HOME/ana:$LOCAL_BASE/opticks/lib:$PATH  ## easy access to scripts and executables
    export PATH

    export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    ## picking a geometry


    export TMP=/home/blyth/local/opticks/tmp
    tmp(){ cd $TMP ; pwd ;  }
    export OPTICKS_EVENT_BASE=$TMP
    ## define a directory in which to store test events, used by integration tests 



Envvars:

LOCAL_BASE
    configures the root of the install, canonically "/usr/local" but can be $HOME/local of you dont 
    have permissions for "/usr/local" 

OPTICKS_HOME
    path to the "opticks" source directory  

PYTHONPATH
    directory containing the "opticks" source directory, allowing "import opticks" 
    to work from python scripts 

OPTICKS_KEY
    picks a geometry geocache, created with the geocache-create function   

PATH
    Opticks executables including more than 400 tests are installed into $LOCAL_BASE/opticks/lib, 
    setting the PATH as indicated gives easy access to these as well as many scripts

TMP
    when undefined a default directory of /tmp/username/opticks is used as scratch space by unit tests 

OPTICKS_EVENT_BASE
    The OPTICKS_EVENT_BASE envvar is used by both C++ boostrap/BOpticksEvent.cc and 
    python ana/nload.py to define a directory beneath with events are stored.
    When undefined this defaults to the TMP envvar value or its default.

    Some integration tests write photon propagation OpticksEvents which are loaded into
    python for analysis by scripts directly invoked from the C++ executable.  
    OpticksEvents comprise several .npy arrays  necessitating the NumPy python module for loading. 





    

The most important lines of the setup are::

   opticks-(){ . $HOME/opticks/opticks.bash && opticks-env $* ; }
   opticks-


The first line defines the bash function *opticks-* that is termed a precursor function 
as running it will define other functions all starting with *opticks-* such as *opticks-vi*
and *opticks-usage*.  The second line runs the function which defines further functions.


Recommended bash setup arrangement
------------------------------------

The recommended arrangment of bash setup scripts:

* `~/.bash_profile` should source `~/.bashrc`
* `~/.bashrc` should source `~/.opticks_setup` (PRIOR to any early exits)

Using this approach succeeds to setup the opticks bash functions
and exports with either "bash -l" or "bash -i" or from within
scripts using shebang line "#!/bin/bash -l". 

This makes the setup immune to differing treatments of when 
`~/.bash_profile` and `~/.bashrc` are to invoked by various Linux 
distros and macOS. 



Example `~/.bash_profile`:

.. code-block:: sh

    # .bash_profile

    if [ -f ~/.bashrc ]; then                 ## typical setup 
            . ~/.bashrc
    fi



Some Linux distros (Ubuntu) have a default `.bashrc` which early exits. 
It is necessary to *source ~/.opticks_setup* prior to the early exit.  
Example `~/.bashrc`:

.. code-block:: sh

    # .bashrc

    vip(){ vim ~/.bash_profile ~/.bashrc ~/.opticks_setup ; } 
    ini(){ source ~/.bashrc ; } 

    source ~/.opticks_setup

    ##### below from default Ubuntu .bashrc early exits if bash is not invoked with -i option 

    # If not running interactively, don't do anything
    case $- in
        *i*) ;;
          *) return;;
    esac



For notes about this see `notes/issues/ubuntu-bash-login-shell-differences.rst`



Check your bash environment setup
-------------------------------------

If the below commandline gives errors, compare your *.bash_profile*  *.bashrc* and *.opticks_setup* with 
the above examples. 

::

    [blyth@localhost ~]$ bash -lc "opticks- ; opticks-info "    ## RHEL, Centos (and Ubuntu too)

    opticks-locations
    ==================

          opticks-source   :   /home/blyth/opticks/opticks.bash
          opticks-home     :   /home/blyth/opticks
          opticks-name     :   opticks

          opticks-fold     :   /home/blyth/local/opticks
 
    ...


.bash_profile OR .bashrc, macOS and Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With most Linux distributions and terminal managers the *.bash_profile* is run
only on login and *.bashrc* is run for every new terminal window, BUT with macOS Terminal.app
the *.bash_profile* is run for every new terminal window.  Thus for compatibility 
the best approach to put setup into *.bashrc* and source it from *.bash_profile* : giving 
the same behaviour on both Linux and macOS.

For background on dotfiles http://mywiki.wooledge.org/DotFiles



Location Overrides by envvar
-------------------------------

===========================  ========================================
envvar                        precursor-;funcname 
===========================  ========================================
OPTICKS_GEANT4_HOME           g4-;g4-prefix
OPTICKS_OPTIX_HOME            optix-;optix-fold
OPTICKS_COMPUTE_CAPABILITY    opticks-;opticks-compute-capability
OPTICKS_OPTIX_INSTALL_DIR     opticks-;opticks-optix-install-dir
===========================  ========================================

Use the ``1_Utilities/deviceQuery`` CUDA sample to show your compute capability.
A list is provided at https://developer.nvidia.com/cuda-gpus

Opticks Installation Overview
--------------------------------

Opticks installation requires:

* bash shell and build tools such as mercurial, git, curl, etc.. 
* recent cmake 3.8+
* Boost C++ libraries 1.59+ 

* installations of pre-requisites packages, see below for notes on versions

  * NVIDIA OptiX 
  * NVIDIA CUDA 

After meeting these requirements you can install Opticks and its
external packages using a single command: *opticks-full* 


Build Tools
------------

Getting, configuring, unpacking, building and installing Opticks and
its externals requires unix tools including:

* bash shell
* mercurial hg 
* git 
* curl
* tar
* zip
* cmake 3.12+

CMake Version 3.12+
----------------------

* **I recommend use of at least 3.12 for building Opticks**.
* **The most common Opticks build issues arise from using older CMake versions.** 

Check your version with::

    simon:~ blyth$ cmake --version
    cmake version 3.12.0

Although usually preferable to get build tools using your system 
package manager, the system cmake version will almost certainly 
not be recent enough. 

Opticks CMake infrastructure makes heavy use of recent CMake target 
import/export features used by BCM (Boost CMake Modules).
The current Opticks CMake infrastructure was developed in May 2018 
using CMake 3.11 and 3.12 (I am currently using 3.14.1)
The Opticks repository includes bash functions for local installs of 
cmake with precursor function *ocmake-* which will install 3.14.1

For what goes wrong if you use an older CMake version see:

* ``notes/issues/cmake_target_link_libraries_for_imported_target.rst``


To install CMake 3.14.1::

    [blyth@localhost opticks]$ ocmake-     ## run precursor function that defines the others
    [blyth@localhost opticks]$ ocmake-vi   ## take a look at the bash functions 
    [blyth@localhost opticks]$ ocmake-info  
    ocmake-info
    ============

    ocmake-vers : 3.14.1
    ocmake-nam  : cmake-3.14.1
    ocmake-url  : https://github.com/Kitware/CMake/releases/download/v3.14.1/cmake-3.14.1.tar.gz
    ocmake-dir  : /home/blyth/local/opticks/externals/cmake/cmake-3.14.1

    [blyth@localhost opticks]$ ocmake--    ## downloads, configures, builds, installs

After installation you will need to adjust you PATH to 
use the newer *cmake* binary. Check with::

    which cmake
    cmake --version 



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


Platform Support
--------------------

A recent Scientific Linux is the target platform for production running of Opticks, 
but I am happy to try to help with installations on any Linux supported by CUDA.

Initial development was done on macOS (late 2013 MacBook pro : the last Mac laptop with an NVIDIA GPU) 
with occasional ports to keep thinks working on Scientific Linux. But now due to the lack of Macs 
with NVIDIA GPUs development has moved to Linux CentOS 7 and Scientific Linux.



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



Opticks without an CUDA capable GPU ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the past an experimental port of Opticks onto a Windows machine without a CUDA capable GPU 
was made. Using saved propagations it was possible to visualize optical photon propagations through a
detector geometry using OpenGL.  

Although this mode of operation is a low priority, it might be revived in future, for example
allowing outreach demonstrations in schools without CUDA capable GPUs.


Versions of CUDA and OptiX 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I recommend you start your installation attempt with the lastest versions of OptiX
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


Using a non-standard OptiX version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Opticks tends to adopt new OptiX versions very soon after they become
available approximately twice per year. This is because OptiX continues 
to improve rapidly. Updating OptiX typically also requires an update of 
the CUDA version and the NVIDIA driver.  

Sometimes due to the suitable NVIDIA driver not yet being installed by 
your system admin it is necessary to use older OptiX+CUDA versions.  
Opticks aims to allow this for recent version combinations only.

Extract from .opticks_setup::

    # The location to look for OptiX libs defaults to $(opticks-prefix)/externals/OptiX
    # to override that while testing a non-standard OptiX version set the OPTICKS_OPTIX_INSTALL_DIR envvar 
    # which overrides the default in bash function opticks-optix-install-dir
    #
    unset OPTICKS_OPTIX_INSTALL_DIR
    export OPTICKS_OPTIX_INSTALL_DIR=/usr/local/OptiX_511  

    export CUDA_VERSION=10.1
    unset LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64

    # Normally Opticks executables find the OptiX libs via the RPATH 
    #  "$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64"
    #
    # when using a non-standard OptiX version (older or newer) 
    # it is necessary to append to LD_LIBRARY_PATH as penance for being non-standard
    #
    [ -n "$OPTICKS_OPTIX_INSTALL_DIR" ] && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OPTICKS_OPTIX_INSTALL_DIR}/lib64

     



Building Opticks against "foreign" externals such as Geant4, Boost
-------------------------------------------------------------------

When integrating Opticks with a detector simulation framework 
it is important that externals that are incommon between Opticks and the framework
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



 
Testin CUDA and OptiX Installs and nvcc toolchain
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

Geant4 installed as the last external in the *opticks-externals* list.
The *g4-* precursor selects a version of Geant4.  Currently a bit dated, this is intended to be brought uptodate sometime.
The coupling between Opticks and Geant4 is intended to be weak : so a range of 
recent versions of Geant4 are intended to be supported.
 

Building Opticks 
---------------------

Once you have the necessary build tools and the pre-requisites you 
can download and install the externals and build Opticks itself with::

   opticks-
   opticks-full   

Note that repeating *opticks-full* will wipe the Opticks build directory 
and run again from scratch. 

After the first full build, much faster update builds can be done with::

   opticks--


Externals 
~~~~~~~~~~~~

The *opticks-full* command automatically downloads and installs the below external packages
into the places required by Opticks.

To list the externals installed by *opticks-full* use the *opticks-externals* function::

    [blyth@localhost opticks]$ opticks-externals
    bcm
    glm
    glfw
    glew
    gleq
    imgui
    assimp
    openmesh
    plog
    opticksdata
    oimplicitmesher
    odcs
    oyoctogl
    ocsgbsp
    xercesc
    g4


=================  =====================   ==============================================================================
precursor          pkg name                notes
=================  =====================   ==============================================================================
bcm                BCM                     My fork of Boost CMake Modules, which eases use of modern CMake target import/export 
glm-               GLM                     OpenGL mathematics, 3D transforms 
glfw-              GLFW                    Interface between system and OpenGL, creating windows and receiving input
glew-              GLEW                    OpenGL extensions loading library, cmake build didnt work, includes vc12 sln for windows
gleq-              GLEQ                    Keyboard event handling header from GLFW author, header only
imgui-             ImGui                   OpenGL immediate mode GUI, depends on glfw and glew
assimp-            Assimp                  Assimp 3D asset importer, my fork that handles G4DAE extras
openmesh-          OpenMesh                basis for mesh navigation and fixing
plog-              PLog                    Header only logging, supporting multi dll logging on windows 
opticksdata-       -                       Dayabay G4DAE and GDML geometry files for testing Opticks      
oimplicitmesher-   ImplicitMesher          Polygonization of implicitly defined shapes
odcs-              DualContouringSample    Alternate polygonization using Octree for multi-resolution, however its slow
oyoctogl-          YoctoGL                 Used for glTF geometry file format handling, parsing/serializing    
ocsgbsp-           CSGBSP                  Another BSP approach to polygonization under investigation
xercesc            XercesC                 XML handling dependency of Geant4, required for GDML parsing
g4                 Geant4                  The preeminent simulation toolkit
=================  =====================   ==============================================================================


Separate installation of externals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *opticks-externals* function lists current precursor names, *opticks-externals-install* runs each 
of the precursor functions in turn.  To rerun a single external install, use the below pattern of running 
the precursor function and then the installer function.

::

   oyoctogl-
   oyoctogl--

After installation has been done rerunning *opticks-externals-install* completes quickly,
and does no harm.


Manually Configuring Opticks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the automated configuring done by *opticks-full* failed to find the
pre-requisites you may need to specify some options to *opticks-configure* 
to help the build scripts.

CMake is used to configure Opticks and generate Makefiles or Visual Studio solution files on Windows.
For a visualization only build with system Boost 
the defaults should work OK and there is no need to explicitly configure. 
If a local Boost was required then::

    opticks-configure -DBOOST_ROOT=$(boost-prefix) 
    
For a full build with CUDA and OptiX configure with::

    opticks-configure -DCUDA_TOOLKIT_ROOT_DIR=/Developer/NVIDIA/CUDA-7.0 \
                      -DOptiX_INSTALL_DIR=/Developer/OptiX \
                      -DCOMPUTE_CAPABILITY=52 \
                      -DBOOST_ROOT=$(boost-prefix) 

Another configure example::

    opticks-configure -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-7.0 \ 
                      -DOptiX_INSTALL_DIR=/home/gpu/NVIDIA-OptiX-SDK-3.8.0-linux64/ \ 
                      -DCOMPUTE_CAPABILITY=52 \
                      -DBOOST_ROOT=/usr/local/lib



The argument `-DCOMPUTE_CAPABILITY=52` specifies to compile for compute capability 5.2 architectures 
corresponding to Maxwell 2nd generation GPUs. 
Lookup the appropriate capability for your GPU in the below short table.

====================  =========================  =================== 
Compute Capability    Architecture               GPU Examples
====================  =========================  ===================
2.1                   Fermi                      **NOT SUPPORTED BY OPTICKS**
3.0                   Kepler                     GeForce GT 750M
5.0                   Maxwell 1st generation     Quadro M2000M
5.2                   Maxwell 2nd generation     Quadro M5000
6.1                   Pascal                     GeForce GTX 1080
====================  =========================  ===================

For more complete tables see

* https://en.wikipedia.org/wiki/CUDA
* https://developer.nvidia.com/cuda-gpus.

Opticks requires a compute capability of at least 3.0, if you have no suitable GPU 
or would like to test without GPU acceleration use `-DCOMPUTE_CAPABILITY=0`.


These configuration values are cached in the CMakeCache.txt file
in the build directory. These values are not overridden by rebuilding 
with the *opticks--* bash function. 
A subsequent *opticks-configure* however will wipe the build directory 
allowing new values to be set.


To build::

    opticks--



Opticks Without NVIDIA OptiX and CUDA ?
------------------------------------------

High performance optical photon simulation requires an NVIDIA GPU 
with compute capability of 3.0 or better (Kepler, Maxwell or Pascal architectures).
However if your GPU is not able to run OptiX/CUDA but is able to run OpenGL 4.0
(eg if you have an AMD GPU or an integrated Intel GPU) 
it is still possible to make a partial build of Opticks 
using cmake switch WITH_OPTIX=OFF. 

The partial mode provides OpenGL visualizations of geometry and  
photon propagations loaded from file.  
This mode is not tested often, so provide copy/paste errors if it fails for you.


Geant4 Dependency
-------------------

Opticks is structured as a collection of packages 
organized by their local and external dependencies, see :doc:`overview` for a table
or run the bash function *opticks-deps*.
Only a few of the very highest level packages depend on Geant4. 

cfg4
     validation comparisons
okg4
     integrated Opticks+G4 for “gun running"
g4ok
     minimal interface for embedding Opticks inside Geant4 applications

Opticks dependency on Geant4 is intended to be loose 
in order to allow working with multiple G4 versions (within a certain version range), 
using version preprocessor macros to accommodate differences.  
So please send copy/paste reports of incompatibilities together with G4 versions.

The weak G4 dependency allows you to test most of Opticks even 
without G4 installed.  


Embedded Opticks using G4OK package 
-------------------------------------

In production, Opticks is intended to be run in an embedded mode 
where, Geant4 and Opticks communicate via “gensteps” and “hits” 
without using any Geant4 headers. This works via some 
Geant4 dependant glue code within each detectors simulation framework 
that does the below:

* inhibits CPU generation of optical photons from G4Scintillation and G4Cerenkov processes, 
  instead "gensteps" are collected

* invokes embedded Opticks (typically at the end of each event) 
  passing the collected "gensteps" across to Opticks which performs the 
  propagation 

* pulls back the PMT hits and populates standard Geant4 hit collections with these

Once the details of the above integration have been revisted for JUNO example 
integration code will be provided within the Opticks repository. 



Testing Installation
----------------------

The *opticks-t* functions runs ctests for all the opticks projects::

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



Creating Legacy Workflow Geocache
-----------------------------------

Some tests depend on the geometry cache being present. To create the legacy geometry cache::

   op.sh -G 


Creating Direct Workflow Geocache
-----------------------------------

The integration tests require a direct workflow geocache to exist as they 
need it as their base geometry. The direct geocache is created by the OKX4Test executable
which loads a GDML file and directly translates it into an Opticks GGeo geometry 
instance and persists that into a direct geocache. 
The bash function to do this is *geocache-create*, use that with::

    geocache-              # run precursor function which defines the others 
    type geocache-create   # take a look at what its doing 
    geocache-create       

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


To reuse the geometry the OPTICKS_KEY envvar needs to be exported from 
`.opticks_setup`, see the next section for an example.




Opticks NumPy based Analysis
--------------------------------

Opticks uses the NumPy (NPY) buffer serialization format 
for geometry and event data, thus analysis and debugging requires
python and the ipython and numpy extensions.



Systems where Opticks has been Installed
------------------------------------------

macOS 10.13.4 (17E199) High Sierra, Xcode 9.2  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* macOS 10.13.4 (17E199) High Sierra 
* Xcode 9.2 (actually on 9.3 but xcode-select back to 9.2) as required by nvcc (the CUDA compiler)
* NVIDIA GPU Driver Version: 387.10.10.10.30.103  (aka Web Driver)
* NVIDIA CUDA Driver : 387.178
* NVIDIA CUDA 9.1
* NVIDUA OptiX 5.0.1


macOS 10.9.4 Mavericks : Xcode/clang toolchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Primary development platfom : Mavericks 10.9.4 
* NVIDIA Geforce GT 750M (mobile GPU) 

Linux : GCC toolchain
~~~~~~~~~~~~~~~~~~~~~~~~~

* DELL Precision Workstation, running Ubuntu 
* DELL Precision Workstation, running CentOS 7
* NVIDIA Quadro M5000 

Windows : Microsoft Visual Studio 2015, Community edition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Ported to Windows 7 SP1 machine 
* non-CUDA capable GPU

Opticks installation uses the bash shell. 
The Windows bash shell that comes with 
the git-for-windows project was used for this purpose

* https://github.com/git-for-windows
 
Despite lack of an CUDA capable GPU, the OpenGL Opticks
visualization was found to operate successfully.

OpenGL Version Requirements
------------------------------

Opticks uses GLSL shaders with version 400, 
corresponding to at least OpenGL 4.0

OpenGL versions supported by various systems are listed at the below links.

* macOS : https://support.apple.com/en-us/HT202823  (approx all macOS systems from 2010 onwards)




Using a Shared Opticks Installation
-------------------------------------

If someone has installed Opticks for you already 
you just need to set the PATH variable in your .bash_profile 
to easily find the Opticks executables and scripts. 

.. code-block:: sh

    # .bash_profile

    # Get the aliases and functions
    if [ -f ~/.bashrc ]; then
        . ~/.bashrc
    fi

    # User specific environment and startup programs

    PATH=$PATH:$HOME/.local/bin:$HOME/bin
    ini(){ . ~/.bash_profile ; }

    ok-local(){    echo /home/simonblyth/local ; }
    ok-opticks(){  echo /home/simonblyth/opticks ; }
    ok-ctest(){    ( cd $(ok-local)/opticks/build ; ctest3 $* ; ) }

    export PATH=$(ok-opticks)/ana:$(ok-opticks)/bin:$(ok-local)/opticks/lib:$PATH


You can test the installation using the `ok-ctest` function defined in 
the .bash_profile. The output shoule look like the below. 
The permission denied error is not a problem.

.. code-block:: sh

    [blyth@optix ~]$ ok-ctest
    Test project /home/simonblyth/local/opticks/build
    CMake Error: Cannot open file for write: /home/simonblyth/local/opticks/build/Testing/Temporary/LastTest.log.tmp
    CMake Error: : System Error: Permission denied
    Problem opening file: /home/simonblyth/local/opticks/build/Testing/Temporary/LastTest.log
    Cannot create log file: LastTest.log
            Start   1: SysRapTest.SEnvTest
      1/155 Test   #1: SysRapTest.SEnvTest ........................   Passed    0.00 sec
            Start   2: SysRapTest.SSysTest
    ...
    ...
    154/155 Test #154: cfg4Test.G4StringTest ......................   Passed    0.06 sec
            Start 155: cfg4Test.G4BoxTest
    155/155 Test #155: cfg4Test.G4BoxTest .........................   Passed    0.05 sec

    100% tests passed, 0 tests failed out of 155

    Total Test time (real) =  48.30 sec




