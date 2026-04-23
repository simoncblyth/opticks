generalizing-build-install-dirs-with-OPTICKS_CONFIG
=====================================================


Motivation
-----------

From ~/opticks/notes/sop/client_server_opticks_testing.rst

The client build needs different config flags::

     -DWITH_CUDA=OFF -DWITH_CURL=ON

Do not want to om-cleaninstall everytime switch between configs.

So that means the build and install directories need to
be separated by a config string beyond the current approach 
of just dividing by CMAKE_BUILD_TYPE Debug Release

Do I need 2-dimensions ? Debug/Release and CONFIG "Standard/Client/...".
Probably not, can use just the one config dimension where "Debug"
build is implied for non-standard config. Or could use ClientDebug/ClientRelease
when need to.


+----------------------+------------------------+------------------------------------------+
|     OPTICKS_CONFIG   |    CMAKE_BUILD_TYPE    |   Notes                                  |
+======================+========================+==========================================+
|     Release          |    Release             |  Standard flags with CUDA, OptiX...      |
+----------------------+------------------------+------------------------------------------+
|     Debug            |    Debug               |  Standard flags with CUDA, OptiX...      |
+----------------------+------------------------+------------------------------------------+
|     Client           |    Debug               |  -DWITH_CUDA=OFF -DWITH_CURL=ON          |
+----------------------+------------------------+------------------------------------------+
     

* Q: Expand existing OPTICKS_BUILDTYPE OR rename to OPTICKS_CONFIG ?
* A: Need OPTICKS_CONFIG to be a higher level control as opticks-buildtype directly feeds CMAKE_BUILD_TYPE


Current Approach on Workstation
-----------------------------------


::

    (ok) A[blyth@localhost opticks]$ t om-prefix
    om-prefix () 
    { 
        echo ${OPTICKS_PREFIX:-/usr/local/opticks}
    }
    (ok) A[blyth@localhost opticks]$ echo $OPTICKS_PREFIX
    /data1/blyth/local/opticks_Debug


Q: Where is the "_Debug" being slid in there ?
A: Done at input level in ~/.opticks_config

OPTICKS_PREFIX is an input, 2nd setting from ~/.opticks_config that is sourced by local_ok_build (and hence "lo")::

     17 export OPTICKS_HOME=$HOME/opticks
     18 
     19 #buildtype=Release
     20 buildtype=Debug
     21 export OPTICKS_BUILDTYPE=$buildtype
     22 export OPTICKS_PREFIX=/data1/blyth/local/opticks_${OPTICKS_BUILDTYPE}
     23     


Generalize putting OPTICKS_CONFIG at highest level::

     15 ### TOP CONFIG RELEVANT TO BOTH BUILD AND USAGE ###
     16 
     17 export OPTICKS_HOME=$HOME/opticks
     18 
     19 config=Debug
     20 export OPTICKS_CONFIG=${OPTICKS_CONFIG:-$config}
     21 
     22 buildtype=Debug
     23 case $OPTICKS_CONFIG in
     24    Debug)    buildtype=Debug ;;
     25    Release)  buildtype=Release ;;
     26    Client)   buildtype=Debug  ;;
     27 esac
     28 
     29 export OPTICKS_BUILDTYPE=$buildtype
     30 #export OPTICKS_PREFIX=/data1/blyth/local/opticks_${OPTICKS_BUILDTYPE}
     31 export OPTICKS_PREFIX=/data1/blyth/local/opticks_${OPTICKS_CONFIG}    # Apr23 2026 : Generalize to OPTICKS_CONFIG
     32 



om-bdir needs no change as it is within the prefix::

    (ok) A[blyth@localhost opticks]$ om-bdir sysrap
    /data1/blyth/local/opticks_Debug/build/sysrap
    (ok) A[blyth@localhost opticks]$ t om-bdir
    om-bdir () 
    { 
        : TODO separate bdir depending on Release/Debug so its faster to switch;
        local gen=$(om-cmake-generator);
        case $gen in 
            "Unix Makefiles")
                echo $(om-prefix)/build/$1
            ;;
            "Xcode")
                echo $(om-prefix)/build_xcode/$1
            ;;
        esac
    }


replace low level opticks-build-with-cuda/curl with high level opticks-config
-------------------------------------------------------------------------------

::


    (ok) A[blyth@localhost opticks]$ opticks-f opticks-build-with
    ./CSG/CMakeLists.txt:See opticks-build-with-cuda-notes re:rebuilding after flipping CUDA ON->OFF/OFF->ON
    ./g4cx/CMakeLists.txt:See opticks-build-with-cuda-notes re:rebuilding after flipping CUDA ON->OFF/OFF->ON
    ./okconf/CMakeLists.txt:See opticks-build-with-cuda-notes re:rebuilding after flipping CUDA ON->OFF/OFF->ON
    ./sysrap/CMakeLists.txt:See opticks-build-with-cuda-curl-notes re:rebuilding after flipping CUDA ON->OFF/OFF->ON
    ./u4/CMakeLists.txt:See opticks-build-with-cuda-notes re:rebuilding after flipping CUDA ON->OFF/OFF->ON
    ./om.bash:       if [ "$(opticks-build-with-cuda)" == "ON" ]; then
    ./om.bash:       elif [ "$(opticks-build-with-cuda)" == "OFF" ]; then
    ./om.bash:    local pkgopt="-DBUILD_WITH_CUDA=$(opticks-build-with-cuda) -DBUILD_WITH_CURL=$(opticks-build-with-curl)"
    ./om.bash:       -DBUILD_WITH_CUDA=$(opticks-build-with-cuda)
    ./om.bash:   opticks-build-with-cuda    : $(opticks-build-with-cuda)
    ./opticks.bash:#opticks-build-with-cuda(){ echo ${OPTICKS_BUILD_WITH_CUDA:-ON}  ; }  # ON or OFF
    ./opticks.bash:#opticks-build-with-curl(){ echo ${OPTICKS_BUILD_WITH_CURL:-OFF}  ; } # ON or OFF
    (ok) A[blyth@localhost opticks]$ 

    vi CSG/CMakeLists.txt g4cx/CMakeLists.txt okconf/CMakeLists.txt sysrap/CMakeLists.txt u4/CMakeLists.txt om.bash opticks.bash



::

     528 om-subs--()
     529 {
     530    if [ -n "$OM_SUBS" ]; then
     531        case ${OM_SUBS} in
     532          all)     om-subs--all ;;
     533          nocuda)  om-subs--nocuda ;;
     534          minimal) om-subs--minimal ;;
     535          alt)     om-subs--alt ;;
     536        esac
     537    else
     538        if [[ "$(opticks-config)" =~ Debug|Release ]]; then
     539             om-subs--all
     540        elif [[ "$(opticks-config)" =~ Client ]]; then
     541             om-subs--nocuda
     542        fi
     543    fi
     544 }




     910 om-pkg-opt()
     911 {
     912     : om.bash
     913     : packages with optional building with or without CUDA need the -DBUILD_WITH_CUDA=ON/OFF
     914     : eg locate them with:
     915     :
     916     :    find . -name CMakeLists.txt -exec grep -H option\(BUILD_WITH_CUDA {} \;
     917     :
     918     : Other packages like QUDARap and CSGOptiX can only be built with CUDA so they have no such option
     919     :
     920 
     921     local name=$1
     922     local opt=""
     923     local pkgopt=""
     924 
     925     case $(opticks-config) in
     926         Debug|Release) pkgopt="-DBUILD_WITH_CUDA=ON -DBUILD_WITH_CURL=OFF" ;;
     927                Client) pkgopt="-DBUILD_WITH_CUDA=OFF -DBUILD_WITH_CURL=ON" ;;
     928     esac
     929     
     930     case $name in
     931        okconf) opt="$pkgopt" ;;
     932        sysrap) opt="$pkgopt" ;;
     933           CSG) opt="$pkgopt" ;;
     934            u4) opt="$pkgopt" ;;
     935          g4cx) opt="$pkgopt" ;;
     936     esac 
     937     echo $opt
     938 }   







Too much  use of OPTICKS_BUILDTYPE to change its meaning
---------------------------------------------------------

::

    (ok) A[blyth@localhost opticks]$ opticks-f OPTICKS_BUILDTYPE
    ./opticks.bash:opticks-buildtype(){       echo ${OPTICKS_BUILDTYPE:-Debug}  ; }

::

    ok) A[blyth@localhost opticks]$ opticks-f opticks-buildtype
    ./bin/opticks-setup-minimal.sh:opticks-buildtype(){ echo Debug ; }
    ./examples/UseG4OK/go.sh:#     -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./examples/UseInstance/go.sh:            -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./examples/UseOKConf/go.sh:     -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./externals/g4.bash:       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \\
    ./externals/g4.bash:   opticks-buildtype       : $(opticks-buildtype)
    ./externals/g4.bash:       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./externals/imgui.bash:      -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./externals/ocsgbsp.bash:       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./externals/ocsgbsp.bash:    cmake --build . --config $(opticks-buildtype) --target ${1:-install}
    ./externals/odcs.bash:       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./externals/odcs.bash:    cmake --build . --config $(opticks-buildtype) --target ${1:-install}
    ./externals/oimplicitmesher.bash:       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./externals/oimplicitmesher.bash:    #cmake --build . --config $(opticks-buildtype) --target ${1:-install}
    ./okconf/go.sh:    -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./om.bash:       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./om.bash:       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    ./om.bash:       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \\
    ./om.bash:    opticks-buildtype  : $(opticks-buildtype)
    ./om.bash:   opticks-buildtype          : $(opticks-buildtype)
    ./om.bash:   #local libprefix=$LOCAL_BASE/opticks_$(opticks-buildtype)
    ./opticks.bash:opticks-buildtype(){       echo ${OPTICKS_BUILDTYPE:-Debug}  ; }
    (ok) A[blyth@localhost opticks]$ 



lo_client shakedown
---------------------


Missing "internally" managed externals directory : so fails to find BCM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This means need to "opticks-full" for the Client build.


lo_client::

    (ok) A[blyth@localhost opticks]$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/python-numpy/1.26.4
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Python/3.11.10
    /data1/blyth/local/custom4_Debug/0.1.9
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/CLHEP/2.4.7.1
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Xercesc/3.2.4

lo::

    ok) A[blyth@localhost opticks]$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/python-numpy/1.26.4
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Python/3.11.10
    /data1/blyth/local/custom4_Debug/0.1.9
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/CLHEP/2.4.7.1
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Xercesc/3.2.4

    /data1/blyth/local/opticks_Debug
    /data1/blyth/local/opticks_Debug/externals
    /cvmfs/opticks.ihep.ac.cn/external/OptiX_800




::

    (ok) A[blyth@localhost opticks]$ opticks-externals
    bcm
    glm
    glfw
    glew
    gleq
    imgui
    plog
    nljson
    (ok) A[blyth@localhost opticks]$ t opticks-externals-install
    opticks-externals-install () 
    { 
        opticks-installer- $(opticks-externals)
    }




How to client build
---------------------

::

    lo_client
    opticks-
    opticks-full





