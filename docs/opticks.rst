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

Connect the opticks bash functions to your shell by adding a line to your .bash_profile
and configure the location of the install with the LOCAL_BASE environment variable 
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


Opticks Installation Overview
--------------------------------

Opticks installation requires:

* bash shell and build tools such as mercurial, git, curl, etc.. 
* recent cmake 2.8.9+
* Boost C++ libraries 1.59+ 

* installations of pre-requisites packages

  * NVIDIA OptiX 3.80
  * NVIDIA CUDA 7.0 

I advise those precise versions of OptiX and CUDA, if you use other versions 
you become a developer rather than a user, please report your findings.
The pre-requisite packages from NVIDIA need to be installed 
following the instructions from NVIDIA. You will also need to 
register as an NVIDIA developer.

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
* cmake 2.8.9+

CMake
-------

A rather recent cmake version is required. Check your version with::

    simon:~ blyth$ cmake --version
    cmake version 3.4.1

Updating build tools is best done via your system package manager.  
For example on OSX with macports update cmake with::

   port info cmake           # check the version the package manager proposes
   sudo port install cmake   # do the install

If you or your system administrator are unable to update a tool via the system
package manager then a local install of the tool must be done and your 
login shell PATH modified to use the updated tool. The Opticks repository 
includes bash functions for local installs of cmake with 
precursor function *cmake-*.


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



Opticks Pre-requisites : NVIDIA OptiX and NVIDIA CUDA 
-----------------------------------------------------------


OptiX requires your system to have a fairly recent NVIDIA GPU of CUDA compute capability 3.0 at least.
However without such a GPU the OpenGL visualization should still work, using saved propagations. 

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





Systems where Opticks has been Installed
------------------------------------------

macOS : Xcode/clang toolchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



