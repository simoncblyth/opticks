building-against-foreign-geant4-and-boost
===================================================

Objective
-----------

Build Opticks using the Geant4/Xercesc/Boost that was installed by junoenv ?

* aiming to avoid problems of having multiple versions of dependencies 
* G4 multithreading may prove problematic

Progess
--------

::

    find examples -name 'jgo*.sh'


Status
-------

* direct usage working, UseUseBoost still not




How to organize juno + Opticks ?
-----------------------------------

::

   bash junoenv libs all opticks


Making the above install Opticks and its tree CUDA, OptiX, NVIDIA driver 
seems totally unrealistic.
Opticks should be regarded more of a "peer" that needs to 
be installed separately than an external or preq that gets installed 
by junoenv. 
So instead can instruct that opticks-config must be in the PATH and that the 
junoenv install scripts just check that the Opticks installation
is where opticks-config points to and that the necessary glue interfaces
are generated.

* need to understand "bash junoenv libs all" for this


Opticks build workhorse om- 
-------------------------------------------------------------------------------------------------------

* needs to be sensitive to foreign CMAKE_PREFIX_PATH using foreign Boost, Geant4, XercesC


Switching between 3 boost versions across dependent libs via CMAKE_PREFIX_PATH working
-------------------------------------------------------------------------------------------

In examples/UseBoost and examples/UseUseBoost succeeded to pick between boosts 1.70 1.71 and 1.72 
simply by setting CMAKE_PREFIX_PATH in om-export.

::

    # find_package.py Boost 

    Boost                          : /opt/local/lib/cmake/Boost-1.71.0/BoostConfig.cmake 
    Boost                          : /usr/local/foreign/lib/cmake/Boost-1.72.0/BoostConfig.cmake 
    Boost                          : /usr/local/opticks/externals/lib/cmake/Boost-1.70.0/BoostConfig.cmake 





Locations
-----------

junoenv/junoenv-external-libs.sh::

    296 function juno-ext-libs-install-root {
    297     echo "$JUNOTOP/ExternalLibs"
    298 }

junoenv/packages/geant4.sh::

    026 function juno-ext-libs-geant4-install-dir {
     27     local version=${1:-$(juno-ext-libs-geant4-version)}
     28     echo $(juno-ext-libs-install-root)/$(juno-ext-libs-geant4-name)/$version
     29 }


    128 function juno-ext-libs-geant4-conf-10 {
    129     local msg="===== $FUNCNAME: "
    130     cmake .. -DCMAKE_INSTALL_PREFIX=$(juno-ext-libs-geant4-install-dir) \
    131         -DGEANT4_USE_GDML=ON \
    132         -DGEANT4_INSTALL_DATA=ON \
    133         -DGEANT4_USE_OPENGL_X11=ON \
    134         -DGEANT4_USE_RAYTRACER_X11=ON \
    135         -DGEANT4_BUILD_MULTITHREADED=ON \
    136         -DGEANT4_BUILD_TLS_MODEL=global-dynamic \
    137         -DXERCESC_ROOT_DIR=$(juno-ext-libs-xercesc-install-dir)
    138 
    139         # $(juno-ext-libs-geant4-conf-use-qt) \
    140 
    141     local st=$?
    142     echo $msg $st 1>&2
    143     if [ "$st" != "0" ]; then
    144         exit 1
    145     fi
    146 }


::

    529 g4-cmake(){
    530    local iwd=$PWD
    531 
    532    local bdir=$(g4-bdir)
    533    mkdir -p $bdir
    534 
    535    local idir=$(g4-prefix)
    536    mkdir -p $idir
    537 
    538    g4-cmake-info
    539 
    540    g4-bcd
    541 
    542    cmake \
    543        -G "$(opticks-cmake-generator)" \
    544        -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    545        -DGEANT4_INSTALL_DATA=ON \
    546        -DGEANT4_USE_GDML=ON \
    547        -DXERCESC_LIBRARY=$(xercesc-library) \
    548        -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
    549        -DCMAKE_INSTALL_PREFIX=$idir \
    550        $(g4-dir)
    551 
    552    cd $iwd


    epsilon:issues blyth$ g4-prefix
    /usr/local/opticks/externals


/home/blyth/junotop/ExternalInterface/Externals/Geant4/cmt/requirements::

    package Geant4

    macro Geant4_home "${JUNO_EXTLIB_Geant4_HOME}"

    macro Geant4_cppflags " `geant4-config --cflags` "
    macro Geant4_linkopts " `geant4-config --libs` "

    include_dirs "${G4INCLUDE}"



::

    [blyth@localhost ExternalLibs]$ l Geant4/10.05.p01/bin/
    total 32
    -rwxr-xr-x. 1 blyth blyth 18023 Mar 24 18:35 geant4-config
    -rwxr-xr-x. 1 blyth blyth  4510 Mar 24 18:35 geant4.csh
    -rwxr-xr-x. 1 blyth blyth  3432 Mar 24 18:35 geant4.sh




::

    [blyth@localhost junotop]$ cat /home/blyth/junotop/ExternalLibs/Geant4/10.05.p01/bashrc
    if [ -z "${JUNOTOP}" ]; then
    export JUNO_EXTLIB_Geant4_HOME=/home/blyth/junotop/ExternalLibs/Geant4/10.05.p01
    else
    export JUNO_EXTLIB_Geant4_HOME=${JUNOTOP}/ExternalLibs/Geant4/10.05.p01
    fi

    export PATH=${JUNO_EXTLIB_Geant4_HOME}/bin:${PATH}
    if [ -d ${JUNO_EXTLIB_Geant4_HOME}/lib ];
    then
    export LD_LIBRARY_PATH=${JUNO_EXTLIB_Geant4_HOME}/lib:${LD_LIBRARY_PATH}
    fi
    if [ -d ${JUNO_EXTLIB_Geant4_HOME}/lib/pkgconfig ];
    then
    export PKG_CONFIG_PATH=${JUNO_EXTLIB_Geant4_HOME}/lib/pkgconfig:${PKG_CONFIG_PATH}
    fi
    if [ -d ${JUNO_EXTLIB_Geant4_HOME}/lib/python2.7/site-packages ];
    then
    export LD_LIBRARY_PATH=${JUNO_EXTLIB_Geant4_HOME}/lib/python2.7/site-packages:${LD_LIBRARY_PATH}
    export PYTHONPATH=${JUNO_EXTLIB_Geant4_HOME}/lib/python2.7/site-packages:${PYTHONPATH}
    fi
    if [ -d ${JUNO_EXTLIB_Geant4_HOME}/lib64 ];
    then
    export LD_LIBRARY_PATH=${JUNO_EXTLIB_Geant4_HOME}/lib64:${LD_LIBRARY_PATH}
    fi
    if [ -d ${JUNO_EXTLIB_Geant4_HOME}/lib64/pkgconfig ];
    then
    export PKG_CONFIG_PATH=${JUNO_EXTLIB_Geant4_HOME}/lib64/pkgconfig:${PKG_CONFIG_PATH}
    fi
    if [ -d ${JUNO_EXTLIB_Geant4_HOME}/lib64/python2.7/site-packages ];
    then
    export LD_LIBRARY_PATH=${JUNO_EXTLIB_Geant4_HOME}/lib64/python2.7/site-packages:${LD_LIBRARY_PATH}
    export PYTHONPATH=${JUNO_EXTLIB_Geant4_HOME}/lib64/python2.7/site-packages:${PYTHONPATH}
    fi
    export CPATH=${JUNO_EXTLIB_Geant4_HOME}/include:${CPATH}
    export MANPATH=${JUNO_EXTLIB_Geant4_HOME}/share/man:${MANPATH}

    # For CMake search path
    export CMAKE_PREFIX_PATH=${JUNO_EXTLIB_Geant4_HOME}:${CMAKE_PREFIX_PATH}
    source ${JUNO_EXTLIB_Geant4_HOME}/bin/geant4.sh

::

    $JUNO_EXTLIB_Geant4_HOME/include/Geant4 
    $JUNO_EXTLIB_Geant4_HOME/lib64/

::

    [blyth@localhost junotop]$ cat bashrc.sh
    export JUNOTOP=/home/blyth/junotop
    export CMTPROJECTPATH=/home/blyth/junotop:${CMTPROJECTPATH}
    source /home/blyth/junotop/ExternalLibs/Python/2.7.15/bashrc
    source /home/blyth/junotop/ExternalLibs/Boost/1.70.0/bashrc
    source /home/blyth/junotop/ExternalLibs/Cmake/3.15.2/bashrc
    source /home/blyth/junotop/ExternalLibs/Git/1.8.4.3/bashrc
    source /home/blyth/junotop/ExternalLibs/Xercesc/3.2.2/bashrc
    source /home/blyth/junotop/ExternalLibs/gsl/2.5/bashrc
    source /home/blyth/junotop/ExternalLibs/fftw3/3.3.8/bashrc
    source /home/blyth/junotop/ExternalLibs/sqlite3/3.29.0/bashrc
    source /home/blyth/junotop/ExternalLibs/tbb/2019_U8/bashrc
    source /home/blyth/junotop/ExternalLibs/CMT/v1r26/bashrc
    source /home/blyth/junotop/ExternalLibs/CLHEP/2.4.1.0/bashrc
    source /home/blyth/junotop/ExternalLibs/xrootd/4.10.0/bashrc
    source /home/blyth/junotop/ExternalLibs/ROOT/6.18.00/bashrc
    source /home/blyth/junotop/ExternalLibs/HepMC/2.06.09/bashrc
    source /home/blyth/junotop/ExternalLibs/Geant4/10.05.p01/bashrc
    source /home/blyth/junotop/ExternalLibs/libmore/0.8.3/bashrc
    source /home/blyth/junotop/ExternalLibs/mysql-connector-c/6.1.9/bashrc
    source /home/blyth/junotop/ExternalLibs/mysql-connector-cpp/1.1.8/bashrc
    source /home/blyth/junotop/ExternalLibs/libyaml/0.2.2/bashrc
    source /home/blyth/junotop/ExternalLibs/python-yaml/5.1.2/bashrc
    source /home/blyth/junotop/ExternalLibs/podio/master/bashrc
    [blyth@localhost junotop]$ 



