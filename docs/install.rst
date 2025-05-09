Opticks Install Instructions
==================================


.. contents:: Table of Contents
   :depth: 2


Overview of Opticks installation steps
----------------------------------------------

A high level overview of the sequence of steps to install Opticks are listed below.


0. install "system" externals : NVIDIA GPU Driver, CUDA, OptiX 7.0 or later (see below for how to pick the version)  
   following instructions from NVIDIA. Check they are working.
1. use git to clone opticks **bitbucket** (not github, that is often behind) repository to your home directory, creating ~/opticks
2. hookup the opticks bash functions to your bash shell 
  
   * cp ~/opticks/example.opticks_config ~/.opticks_config
   * ensure that your .bash_profile sources .bashrc
   * add line to .bashrc "source ~/.opticks_config"

3. start a new session and check the bash functions are hooked up correctly with:

   * opticks-info
   * bash -lc "opticks-info"

4. install the foreign externals OR use preexisting installs of clhep,xercesc,g4

   * opticks-foreign     # lists them 
   * opticks-foreign-install    # installs them 
   * see :doc:`externals` for the difference between **system**, **foreign** and **automated** externals 

5. edit ~/.opticks_config setting the paths appropriately for the 
   prefixes of the "system" and "foreign" externals and setting 
   the prefix for the opticks install (eg /usr/local/opticks)

6. install the "automated" externals and opticks itself with **opticks-full**

7. test the opticks build with **opticks-t**, see :doc:`testing`


Platform Support
--------------------

A recent Scientific Linux or CentOS Linux or Alma Linux are the target platforms for production running of Opticks, 
but I am happy to try to help with installations on any Linux supported by CUDA.


Get Opticks 
------------

Clone the repository from bitbucket::

   cd 
   git clone https://bitbucket.org/simoncblyth/opticks
   git clone git@bitbucket.org:simoncblyth/opticks.git   # via git url for developers, uses ssh keys for passwordless pushes


Update an existing clone
---------------------------

::

    cd ~/opticks
    git remote -v   # should list bitbucket.org urls 
    git status
    git pull 



Bash Shell Setup with .opticks_config
---------------------------------------

Copy the `example.opticks_config` to your home directory and hook it up to your
bash environment::

   cp ~/opticks/example.opticks_config ~/.opticks_config
   echo "source ~/.opticks_config" >> ~/.bashrc 

Ensure that `.bashrc` is sourced from your `~/.bash_profile`.  The `~/.opticks_config` 
must be customized for your system, changing the PREFIX envvar paths and directory arguments to `opticks-prepend-prefix`.
The lines of `~/.opticks_config` that will typically need to be customized for your system are highlighted 
in the below literal include of `~/opticks/example.opticks_config`.


.. literalinclude:: /example.opticks_config
   :language: sh
   :emphasize-lines: 22-24,26,29-32
   :linenos:



What the `.opticks_config` script does:

1. defines the `opticks-` bash function, this is termed a precursor function as running it will define other functions all starting with *opticks-* such as *opticks-vi*
2. invokes the `opticks-` bash function which defines `opticks-*` functions as well as other precursor functions such as `g4-`
3. exports three mandatory PREFIX envvars identifying where to install Opticks and where CUDA and OptiX are installed.
4. exports mandatory envvar OPTICKS_COMPUTE_CAPABILITY identifying the capability of the GPU

5. invokes opticks-prepend-prefix for "foreign" externals, this
   prepends to path envvars including PATH, CMAKE_PREFIX_PATH, PKG_CONFIG_PATH and LD_LIBRARY_PATH

6. invokes opticks-setup of vital PATH envvars



Note that the directory paths in the above config are examples that you may need to change as appropriate for you.
To determine the appropriate value for OPTICKS_COMPUTE_CAPABILITY, use the ``1_Utilities/deviceQuery`` CUDA sample.


Bash environment setup, checking bash environment
--------------------------------------------------

Opticks installation and usage is based on bash functions, thus it 
is essential that the bash functions are connected to your
bash environment correctly.

The recommended arrangment of bash setup scripts:

* `~/.bash_profile` should source `~/.bashrc`
* `~/.bashrc` should source `~/.opticks_config` (PRIOR to any early exits)

Using this approach succeeds to setup the opticks bash functions
and exports with either "bash -l" or "bash -i" or from within
scripts using shebang line "#!/bin/bash -l". 

This makes the setup immune to differing treatments of when 
`~/.bash_profile` and `~/.bashrc` are to invoked by various Linux 
distros and macOS. 


Check your bash environment setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@localhost ~]$ bash -lc "opticks- ; opticks-info "    ## RHEL, Centos (and Ubuntu too)

    opticks-locations
    ==================

          opticks-source   :   /home/blyth/opticks/opticks.bash
          opticks-home     :   /home/blyth/opticks
          opticks-name     :   opticks

          opticks-fold     :   /home/blyth/local/opticks
 
    ...

If the above commandline gives errors, compare your *.opticks_config* with 
the above example and consult :doc:`misc/bash_setup`. 


Opticks Installation Build Tools
--------------------------------

Getting, configuring, unpacking, building and installing Opticks and
its externals requires unix tools including:

* bash shell
* git 
* curl
* tar
* zip
* recent cmake 3.12+
* python


CMake Version 3.12+
~~~~~~~~~~~~~~~~~~~~~~

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



Version Requirements
------------------------

For details, see :doc:`externals`

OptiX 7.0 or later 
~~~~~~~~~~~~~~~~~~~~~

OptiX 6.5 and earlier is no longer supported.  

OpenGL 4.1
~~~~~~~~~~~~

Opticks uses GLSL shaders with version 410, corresponding to at least OpenGL 4.1


Building Opticks with **opticks-full**
-----------------------------------------

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

The *opticks-full* command automatically downloads and installs the **automated** 
external packages into the places required by Opticks.  See :doc:`externals`.



opticks-setup.sh : Environment setup for building and using Opticks
------------------------------------------------------------------------

During installation an opticks-setup.sh script is generated 
at path $OPTICKS_PREFIX/bin/opticks-setup.sh.
Sourcing this script sets up the paths to allow building and usage of Opticks executables.::

    source $OPTICKS_PREFIX/bin/opticks-setup.sh

Once the setup is working for you, avoid the output on starting each 
session by redirecting the stdout::

    source $OPTICKS_PREFIX/bin/opticks-setup.sh 1> /dev/null 

The *example.opticks_config* includes these lines already. 

For further details on *opticks-setup.sh* see :doc:`misc/opticks_setup_script`.


Moving Externals
------------------

The opticks-setup.sh script complains regarding BUILD_CMAKE_PREFIX_PATH / BUILD_PKG_CONFIG_PATH 
captured at generation not matching the current envvars then the script 
can be regenerated with "opticks-setup-generate".

When moving around externals, it is necessary to change the build environment
using opticks-prepend-prefix eg:: 

    ## hookup paths to access "foreign" externals 
    opticks-prepend-prefix /usr/local/opticks_externals/clhep
    opticks-prepend-prefix /usr/local/opticks_externals/xercesc
    opticks-prepend-prefix /usr/local/opticks_externals/g4 

After that it is necessary to cleaninstall Opticks with::

    o   
    om- 
    om-cleaninstall



