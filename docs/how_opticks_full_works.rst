How opticks-full works
========================

*opticks-full* is a bash function from the opticks.bash file that does the below: 

1. installs externals *opticks-externals-install*
2. configures, builds and installs controlled by *om-install*
4. prepares installcache *opticks-prepare-installcache*

The *opticks-vi* bash function allows you to examine/edit the functions.



Structure of Opticks Build
-----------------------------

Opticks is structured as a "bunch of separate CMake projects" which are linked together using 
the CMake "find_package" mechanism and CMake target export/import using my fork of BCM (Boost CMake Modules, which is
downloaded and installed as the first external *bcm*). BCM avoids all the boilerplate that is otherwise needed to do CMake 
target export/import.   

The "bunch of separate CMake" projects are tied together using the *om-* (which stands for "Opticks Minimal" ) 
bash functions.


opticks-full bash function
---------------------------

::

    opticks-full()
    {
        local msg="=== $FUNCNAME :"
        echo $msg START $(date)
        opticks-info

        if [ ! -d "$(opticks-prefix)/externals" ]; then
            echo $msg installing the below externals into $(opticks-prefix)/externals
            opticks-externals
            opticks-externals-install
        else
            echo $msg using preexisting externals from $(opticks-prefix)/externals
        fi

        om-
        cd $(om-home)
        om-install

        opticks-prepare-installcache

        echo $msg DONE $(date)
    }



Installation Basics, bash reminder
---------------------------------------

Opticks builds and installs are based on CMake but are controlled from
a layer of bash functions. These functions follow consistent naming conventions, 
functions such as *opticks-* or *om-* ending in *-* are termed *precursor* functions.
Running these functions define other functions all with the corresponding 
prefix, to see the functions use eg *opticks-vi* or *om-vi*.
To introspect the definition of a function use *type*::

    epsilon:opticks blyth$ type opticks-full
    opticks-full is a function
    opticks-full () 
    { 
        local msg="=== $FUNCNAME :";
        echo $msg START $(date);
    ...
    
It is convenient to alias type for simple introspection::

    alias t="type" 

Allowing::

    epsilon:opticks blyth$  t t
    t is aliased to `type'
    epsilon:opticks blyth$ t opticks-
    opticks- is a function
    opticks- () 
    { 
        source $(opticks-source) && opticks-env $*
    }

    epsilon:opticks blyth$ opticks-source
    /Users/blyth/opticks/opticks.bash

    epsilon:opticks blyth$ t om-
    om- is a function
    om- () 
    { 
        . $(opticks-home)/om.bash && om-env $*
    }


Troubleshooting installation of externals
-------------------------------------------

It is necessary for the externals to install without serious error for the 
rest of the build to succeed.
First use introspection to see how the functions work::

    opticks-   ## run the precursor 

    epsilon:opticks blyth$ t opticks-externals    ## introspect using "t" which is an alias for "type"
    opticks-externals is a function
    opticks-externals () 
    { 
        : emits to stdout the names of the bash precursors that download and install the externals;
        cat  <<EOL
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
    EOL

    }


These names get piped to *-opticks-installer* by *opticks-externals-install*::

    epsilon:opticks blyth$ t opticks-externals-install
    opticks-externals-install is a function
    opticks-externals-install () 
    { 
        echo $FUNCNAME;
        opticks-externals | -opticks-installer
    }


The installer runs the precursor *name-* and then the installer *name--* for each in turn::

    epsilon:opticks blyth$ t -- -opticks-installer
    -opticks-installer is a function
    -opticks-installer () 
    { 
        local msg="=== $FUNCNAME :";
        echo $msg START $(date);
        local ext;
        while read ext; do
            echo $msg $ext;
            $ext-;
            $ext--;
        done;
        echo $msg DONE $(date)
    }


To install any of the externals singly do this manually, eg::

    glm-     ## defines the below functions, this precursor function is defined by opticks-
    glm--    ## downloads, builds and installs
    glm-vi   ## to see what its doing 

If there are errors with the externals debug them individually as shown above.


checking externals or Opticks subprojects
--------------------------------------------

The Opticks *examples* directory has many *Use* examples which test single externals 
or subprojects.   

::

    epsilon:opticks blyth$ ls -1d examples/Use*
    examples/UseAssimpRap
    examples/UseBCM
    examples/UseBoost
    examples/UseBoostOld
    examples/UseBoostRap
    examples/UseCFG4
    examples/UseCSGBSP
    examples/UseCUDA
    examples/UseCUDARap
    examples/UseCUDARapThrust
    examples/UseDualContouringSample
    examples/UseG4
    ...


Try to build the examples corresponding to the externals or subprojects 
that you have installation problems with, using the standalone **go.sh** 
script that is in each directory. 


One example : examples/UseOpticksGLFW
------------------------------------------------

Running the executable should open a window containing 
a muticolored rotating triangle.

Complete output from configuring, building and installing::

    epsilon:opticks blyth$ cd examples/UseOpticksGLFW
    epsilon:UseOpticksGLFW blyth$ l
    total 24
    -rwxr-xr-x  1 blyth  staff   383 Jun 25 14:05 go.sh
    -rw-r--r--  1 blyth  staff   577 Jun 25 14:05 CMakeLists.txt
    -rw-r--r--  1 blyth  staff  1790 Jun 25 14:05 UseOpticksGLFW.cc
    epsilon:UseOpticksGLFW blyth$ ./go.sh
    /tmp/blyth/opticks/UseOpticksGLFW/build
    -- The C compiler identification is AppleClang 9.0.0.9000039
    -- The CXX compiler identification is AppleClang 9.0.0.9000039
    -- Check for working C compiler: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Configuring UseOpticksGLFW
    -- FindOpticksGLFW.cmake : OpticksGLFW_MODULE      : /Users/blyth/opticks/cmake/Modules/FindOpticksGLFW.cmake 
    -- FindOpticksGLFW.cmake : OpticksGLFW_LIBRARY     : /usr/local/opticks/externals/lib/libglfw.dylib 
    -- FindOpticksGLFW.cmake : OpticksGLFW_INCLUDE_DIR : /usr/local/opticks/externals/include 
    -- OpticksGLFW_MODULE  : /Users/blyth/opticks/cmake/Modules/FindOpticksGLFW.cmake 
    ====== tgt:Opticks::OpticksGLFW tgt_DIR: ================
    tgt='Opticks::OpticksGLFW' prop='INTERFACE_INCLUDE_DIRECTORIES' defined='0' set='1' value='/usr/local/opticks/externals/include' 

    tgt='Opticks::OpticksGLFW' prop='INTERFACE_LINK_LIBRARIES' defined='0' set='1' value='/System/Library/Frameworks/Cocoa.framework;/System/Library/Frameworks/OpenGL.framework;/System/Library/Frameworks/IOKit.framework;/System/Library/Frameworks/CoreFoundation.framework;/System/Library/Frameworks/CoreVideo.framework' 

    tgt='Opticks::OpticksGLFW' prop='INTERFACE_FIND_PACKAGE_NAME' defined='1' set='1' value='OpticksGLFW MODULE REQUIRED' 

    tgt='Opticks::OpticksGLFW' prop='IMPORTED_LOCATION' defined='0' set='1' value='/usr/local/opticks/externals/lib/libglfw.dylib' 


    -- Configuring done
    -- Generating done
    -- Build files have been written to: /tmp/blyth/opticks/UseOpticksGLFW/build
    Scanning dependencies of target UseOpticksGLFW
    [ 50%] Building CXX object CMakeFiles/UseOpticksGLFW.dir/UseOpticksGLFW.cc.o
    [100%] Linking CXX executable UseOpticksGLFW
    [100%] Built target UseOpticksGLFW
    [100%] Built target UseOpticksGLFW
    Install the project...
    -- Install configuration: "Debug"
    -- Installing: /usr/local/opticks/lib/UseOpticksGLFW
    epsilon:UseOpticksGLFW blyth$ UseOpticksGLFW
    epsilon:UseOpticksGLFW blyth$ 


The building of these small examples is typically::

    epsilon:okconf blyth$ cd ~/opticks/examples/UseBoost
    epsilon:UseBoost blyth$ 
    epsilon:UseBoost blyth$ 
    epsilon:UseBoost blyth$ l
    total 48
    -rw-r--r--  1 blyth  staff   107 Jul  6 17:02 PTreeMinimal.cc
    -rw-r--r--  1 blyth  staff  1469 Jul  6 17:02 CMakeLists.txt
    -rwxr-xr-x  1 blyth  staff   730 Jul  6 11:04 go.sh
    -rw-r--r--  1 blyth  staff   408 Jun 25 14:05 TestUseBoost.cc
    -rw-r--r--  1 blyth  staff   618 Jun 25 14:05 UseBoost.cc
    -rw-r--r--  1 blyth  staff   209 Jun 25 14:05 UseBoost.hh
    epsilon:UseBoost blyth$ ./go.sh 
    ...




om-install : configures, builds and installs 
---------------------------------------------------

For details *om-;om-vi* or:

* https://bitbucket.org/simoncblyth/opticks/src/default/om.bash

Note that the invoking directory is taken as an "argument" to the `om-` functions 
such as *om-install*, *om-conf*, *om-make* controlling whether to operate on all subprojects 
or only one.   

If there is a problem with build first try to debug the initial subproject "OKConf".


OKConf subproject 
----------------------

::

    cd ~/opticks/okconf
    om-install



This subproject is the first to be built and installed. 
It introspects the versions of the externals and creates 
the OKConfTest executable which dumps this information::

    epsilon:okconf blyth$ OKConfTest 
    OKConf::Dump
                         OKConf::CUDAVersionInteger() 9010
                        OKConf::OptiXVersionInteger() 50001
                   OKConf::ComputeCapabilityInteger() 30
                            OKConf::CUDA_NVCC_FLAGS() MISSING
                            OKConf::CMAKE_CXX_FLAGS()  -fvisibility=hidden -fvisibility-inlines-hidden -Wall -Wno-unused-function -Wno-unused-private-field -Wno-shadow
                            OKConf::OptiXInstallDir() /Developer/OptiX_501
                       OKConf::Geant4VersionInteger() 1042
                       OKConf::OpticksInstallPrefix() /usr/local/opticks

    OKConf::Check() 0


If that fails to build, then investigate how OKConf is configured::

    epsilon:okconf blyth$ t om-cmake-okconf
    om-cmake-okconf is a function
    om-cmake-okconf () 
    { 
        local sdir=$1;
        local bdir=$PWD;
        [ "$sdir" == "$bdir" ] && echo ERROR sdir and bdir are the same $sdir && return 1000;
        local rc;
        cmake $sdir -G "$(om-cmake-generator)" \
                       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
                       -DCMAKE_PREFIX_PATH=$(om-prefix)/externals \
                       -DCMAKE_INSTALL_PREFIX=$(om-prefix) \
                       -DCMAKE_MODULE_PATH=$(om-home)/cmake/Modules \
                       -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
                       -DCOMPUTE_CAPABILITY=$(opticks-compute-capability);
        rc=$?;
        return $rc
    }


Check that the location of OptiX install *OptiX_INSTALL_DIR* and *COMPUTE_CAPABILITY* of your GPU are correct::

    epsilon:okconf blyth$ om-cmake-info

    om-cmake-info
    ===============

       om-cmake-generator         : Unix Makefiles
       opticks-buildtype          : Debug
       om-prefix                  : /usr/local/opticks

       opticks-optix-install-dir  : /Developer/OptiX_501
       OPTICKS_OPTIX_INSTALL_DIR  : 
     
       opticks-compute-capability : 30
       OPTICKS_COMPUTE_CAPABILITY :  

       NODE_TAG                   : E

    epsilon:okconf blyth$ 



The input variables come from bash functions such as, 

1. *opticks-optix-install-dir*
2. *opticks-compute-capability*


The outputs have defaults, but you can override them using envvars::

    epsilon:okconf blyth$ t opticks-optix-install-dir
    opticks-optix-install-dir is a function
    opticks-optix-install-dir () 
    { 
        echo ${OPTICKS_OPTIX_INSTALL_DIR:-$($FUNCNAME-)}
    }
    epsilon:okconf blyth$ t opticks-compute-capability
    opticks-compute-capability is a function
    opticks-compute-capability () 
    { 
        echo ${OPTICKS_COMPUTE_CAPABILITY:-$($FUNCNAME-)}
    }
    epsilon:okconf blyth$ 

    epsilon:okconf blyth$ t opticks-compute-capability-
    opticks-compute-capability- is a function
    opticks-compute-capability- () 
    { 
        local t=$NODE_TAG;
        case $t in 
            E)
                echo 30
            ;;
            D)
                echo 30
            ;;
    ...



