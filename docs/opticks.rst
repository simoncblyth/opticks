Opticks Install Instructions
==================================

Get Opticks 
------------

Clone the repository from bitbucket::

   cd 
   hg clone http://bitbucket.org/simoncblyth/opticks 
   hg clone ssh://hg@bitbucket.org/simoncblyth/opticks   # via SSH for developers 

Bash setup, envvars
---------------------

Connect the opticks bash functions to your shell by adding a line to your *.bashrc* (Linux)
OR *.bash_profile* (macOS).  Also configure the location of the install with the LOCAL_BASE environment variable 
and the location of the source with the OPTICKS_HOME envvar::

   opticks-(){ . $HOME/opticks/opticks.bash && opticks-env $* ; }
   export LOCAL_BASE=/usr/local   
   export OPTICKS_HOME=$HOME/opticks

The first line defines the bash function *opticks-* that is termed a precursor function 
as running it will define other functions all starting with *opticks-* such as *opticks-vi*
and *opticks-usage*.

Some further .bash_profile setup simplifies use of Opticks binaries and analysis scripts::

    op(){ op.sh $* ; } 

    export PYTHONPATH=$HOME
    export PATH=$LOCAL_BASE/opticks/lib:$OPTICKS_HOME/bin:$OPTICKS_HOME/ana:$PATH


Check your bash environment setup
-------------------------------------

If the below commandline gives errors, check your *.bash_profile* OR *.bashrc*  

::

    [blyth@localhost ~]$ bash -lc "opticks- ; opticks-info "    ## RHEL, Centos
    [blyth@localhost ~]$ bash -ic "opticks- ; opticks-info "    ## Ubuntu

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

::

    opticks-
    boost-
    opticks-configure -DBOOST_ROOT=$(boost-prefix)


Platform Support
--------------------

A recent Scientific Linux is the target platform for production running of Opticks, 
but I am happy to try to help with installations on any Linux supported by CUDA.

Most development has been done on macOS (late 2013 MacBook pro : the last Mac laptop with an NVIDIA GPU) 
with occasional ports to keep thinks working on Scientific Linux.



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
the OptiX release notes. For example I am currently testing and seeing some success 
with the latest OptiX 5.0.1, CUDA 9.1 on the almost latest build of macOS 10.13.4.
This version pinning between CUDA and OptiX is because Opticks links against 
both the OptiX library and the CUDA runtime.

If you cannot use the latest CUDA (because of kernel incompatibility) you will need to
use an older OptiX version contemporary with the CUDA version that your kernel supports.

Version combinations that have been used:

current
   CUDA 9.1, OptiX 5.0.1

earlier
   CUDA 7.0, OptiX 3.80


The reason for the extremes of caution regarding version combinations of drivers 
is that the interface to the GPU is via kernel extensions where if anything goes 
wrong there is no safety net. A bad kernel extension will cause kernel panics, 
your machine crashes and continue to crash until the bad driver is removed 
(on macOS the removal can be done by resetting NVRAM).
 
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

As installing Geant4 takes a long time and considerable storage space it is not installed by *opticks-full*. 
You can however intall Geant4 and XercesC with::

   opticks-optionals-install    # which uses the xercesc- and g4- precursors 


Geant4 Version
~~~~~~~~~~~~~~~~~

The *g4-* precursor selects a version of Geant4.  Currently a bit dated, this is intended to be brought uptodate soon.
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


=================  =====================   ==============================================================================
precursor          pkg name                notes
=================  =====================   ==============================================================================
glm-               GLM                     OpenGL mathematics, 3D transforms 
assimp-            Assimp                  Assimp 3D asset importer, my fork that handles G4DAE extras
openmesh-          OpenMesh                basis for mesh navigation and fixing
glew-              GLEW                    OpenGL extensions loading library, cmake build didnt work, includes vc12 sln for windows
glfw-              GLFW                    Interface between system and OpenGL, creating windows and receiving input
gleq-              GLEQ                    Keyboard event handling header from GLFW author, header only
imgui-             ImGui                   OpenGL immediate mode GUI, depends on glfw and glew
plog-              PLog                    Header only logging, supporting multi dll logging on windows 
opticksdata-       -                       Dayabay G4DAE and GDML geometry files for testing Opticks      
oimplicitmesher-   ImplicitMesher          Polygonization of implicitly defined shapes
odcs-              DualContouringSample    Alternate polygonization using Octree for multi-resolution, however its slow
ocsgbsp-           CSGBSP                  Another BSP approach to polygonization under investigation
oyoctogl-          YoctoGL                 Used for glTF geometry file format handling, parsing/serializing    
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


Configuration Machinery
------------------------

If the above configuration suceeded for you then 
you do not need to understand this machinery.

The below commands from the *opticks-cmake* bash function 
change directory to the build folder and invokes cmake 
to generate a configuration cache file and multiple Makefiles.::

   opticks-bcd
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       -DOptiX_INSTALL_DIR=$(optix-prefix) \
       $* \
       $(opticks-sdir)

CMake is controlled via CMakeLists.txt files. 
The top level one includes the below lines that 
locate the CUDA and OptiX:: 

    set(OPTICKS_CUDA_VERSION 7.0)
    set(OPTICKS_OPTIX_VERSION 3.8)
    ...
    find_package(CUDA ${OPTICKS_CUDA_VERSION})
    find_package(OptiX ${OPTICKS_OPTIX_VERSION})


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
organized by their local and external dependencies, see :doc:`overview` for a table.
Only the two very highest level packages depend on Geant4. 

cfg4
     validation comparisons
okg4
     integrated Opticks+G4 for “gun running"


Opticks dependency on Geant4 is intended to be loose 
in order to allow working with multiple G4 versions (within a certain version range), 
using version preprocessor macros to accommodate differences.  
So please send copy/paste reports of incompatibilities together with G4 versions.

The weak G4 dependency allows you to test most of Opticks even 
without G4 installed.  


Embedded Opticks 
--------------------

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


Issues With Tests
-------------------

Some tests depend on the geometry cache being present. To create the geometry cache::

   op.sh -G 



Running Opticks Scripts and Executables
----------------------------------------

All Opticks executables including the tests are installed 
into $LOCAL_BASE/opticks/lib/ an example `.bash_profile` 
to is provided below:

.. code-block:: sh

    # .bash_profile

    if [ -f ~/.bashrc ]; then                 ## typical setup 
            . ~/.bashrc
    fi

    export LOCAL_BASE=$HOME/local             ## opticks hookup is needed by all Opticks users 
    export OPTICKS_HOME=$HOME/opticks

    opticks-(){  [ -r $HOME/opticks/opticks.bash ] && . $HOME/opticks/opticks.bash && opticks-env $* ; }
    opticks-                                  ## defines several bash functions beginning opticks- eg opticks-info

    o(){ cd $(opticks-home) ; hg st ; }
    op(){ op.sh $* ; }

    PATH=$OPTICKS_HOME/bin:$LOCAL_BASE/opticks/lib:$PATH  ## easy access to scripts and executables
    export PATH


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

* Opticks has been ported to a DELL Precision Workstation, running Ubuntu 
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




