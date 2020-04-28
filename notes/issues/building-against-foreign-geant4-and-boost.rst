building-against-foreign-geant4-and-boost
===================================================

Objective 1
--------------

Build Opticks using the Geant4/Xercesc/Boost that was installed by junoenv ?

* aiming to avoid problems of having multiple versions of dependencies 
* G4 multithreading may prove problematic

Objective 2 
--------------

Create the script glue that allows opticks to be an external 
to junoenv 

* entails simplifications of opticks environment setup



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
* also non CMake workflow with oc.bash must follow PKG_CONFIG_PATH


Switching between 3 boost versions across dependent libs via CMAKE_PREFIX_PATH working
-------------------------------------------------------------------------------------------

In examples/UseBoost and examples/UseUseBoost succeeded to pick between boosts 1.70 1.71 and 1.72 
simply by setting CMAKE_PREFIX_PATH in om-export.

::

    # find_package.py Boost 

    Boost                          : /opt/local/lib/cmake/Boost-1.71.0/BoostConfig.cmake 
    Boost                          : /usr/local/foreign/lib/cmake/Boost-1.72.0/BoostConfig.cmake 
    Boost                          : /usr/local/opticks/externals/lib/cmake/Boost-1.70.0/BoostConfig.cmake 




Issue : g4-export-ini not run when using foreign Geant4
----------------------------------------------------------

* instead need to use standard geant4 env setup

  * done in oe-export-geant4




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



JUNO : PATH setup for ExternalLibs
-----------------------------------

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



The below define JUNO_EXTLIB_Name_HOME envvars and setup the runtime PATH envvars::

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


* Q: what generates this ? and what are the inputs to this generation ?



These look to all follow the same pattern, setting a HOME like JUNO_EXTLIB_sqlite3_HOME
and using it to prepend to the various PATH envvars.

Contrast with om-export could use two packages Opticks and OpticksExternals 
to setup the $(om-prefix) and $(om-prefix)/externals 

As need the JUNO externals to take precedence, these two need go first.



Does "junoenv libs" simply append to the $JUNOTOP/bashrc.sh ?

::

   bash junoenv libs all python






junoenv
------------


::

    102 function setup-juno-external-libs {
    103     echo == $FUNCNAME
    104     source junoenv-external-libs.sh
    105     junoenv-external-libs $@
    106 }
    107 
    108 function setup-juno-external-interface {
    109     echo == $FUNCNAME
    110     source junoenv-env.sh
    111     source junoenv-external-interface.sh
    112     junoenv-external-interface $@
    113 }
    114 
    115 function setup-juno-env {
    116     echo == $FUNCNAME
    117     source junoenv-env.sh
    118     junoenv-env $@
    119 }

    191 function main {
    192     echo = $FUNCNAME
    193     echo = THE JUNOTOP is $JUNOTOP
    194     echo = THE JUNOENVDIR is $JUNOENVDIR
    195     pushd $JUNOENVDIR >& /dev/null
    196     setup-juno-basic-preq
    197     cmd=$1
    198     shift
    199     case $cmd in
    200         all)
    201             setup-juno-all $@
    202             ;;
    203         preq)
    204             setup-juno-preq $@
    205             ;;
    206         libs)
    207             setup-juno-external-libs $@
    208             ;;
    209         cmtlibs)
    210             setup-juno-external-interface $@
    211             ;;
    212         sniper)
    213             setup-juno-sniper $@
    214             ;;
    215         offline)
    216             setup-juno-offline $@
    217             ;;
    218         offline-data)
    219             setup-juno-offline-data $@
    220             ;;
    221         env)
    222             setup-juno-env $@
    223             ;;
    224         fixed)
    225             setup-juno-fixed $@
    226             ;;
    227         archive)
    228             setup-juno-archive $@
    229             ;;
    230         deploy)
    231             setup-juno-deploy $@
    232             ;;
    233         *)
    234             echo Unknown Sub Command $cmd
    235             setup-juno-help
    236             ;;
    237     esac
    238     popd >& /dev/null
    239 }
    240 
    241 main $@




bash junoenv libs all opticks
-------------------------------

* all here means : get, conf, make, install, setup ? 

::

   rm /home/blyth/junotop/ExternalLibs/Build/opticks-download-filename-0.1.0
   bash junoenv libs get opticks


::

    236 function juno-ext-libs-check-is-reused {
    237     local msg="==== $FUNCNAME: "
    238     # just check the install prefix is a soft link or not
    239     local pkg=$1
    240     local newpath=$(juno-ext-libs-${pkg}-install-dir)
    241     if [[ -L "$newpath" && -d "$newpath" ]];
    242     then
    243         echo $msg The installation prefix for $pkg: \"$newpath\" is a soft link. 1>&2
    244         echo $msg It can be a reused library. 1>&2
    245         return 1
    246     else
    247         return 0
    248     fi
    249 }



Planting a link can get it to be regarded as reused::

    mkdir -p $JUNOTOP/ExternalLibs/opticks && cd $JUNOTOP/ExternalLibs/opticks && ln -s $(opticks-prefix) 0.1.0

But it seems the other functions are then not run, when do all or reuse. The setup is needed for path envvar appending::

    [blyth@localhost junoenv]$ bash junoenv libs reuse opticks
    = The junoenv is in /home/blyth/junotop/junoenv
    = main
    = THE JUNOTOP is /home/blyth/junotop
    = THE JUNOENVDIR is /home/blyth/junotop/junoenv
    == setup-juno-basic-preq: ================================================================
    == setup-juno-basic-preq: GLOBAL Environment Variables:
    == setup-juno-basic-preq: $JUNOTOP is "/home/blyth/junotop"
    == setup-juno-basic-preq: $JUNO_EXTLIB_OLDTOP: ""
    == setup-juno-basic-preq: $JUNOARCHIVEGET: ""
    == setup-juno-basic-preq: $JUNOARCHIVEURL: ""
    == setup-juno-basic-preq: ================================================================
    == setup-juno-external-libs
    === junoenv-external-libs: command: reuse
    === junoenv-external-libs: packages: opticks
    === junoenv-external-libs: create function juno-ext-libs-opticks-version- to override default
    === junoenv-external-libs: juno-ext-libs-check-init opticks
    ==== juno-ext-libs-check-init: setup dependencies for opticks
    ==== juno-ext-libs-dependencies-setup-rec-impl: # setup opticks: create function juno-ext-libs-opticks-version- to override default
    ==== juno-ext-libs-dependencies-setup-rec-impl: # setup opticks: source /home/blyth/junotop/junoenv/packages/opticks.sh
    ==== juno-ext-libs-dependencies-setup-rec-impl: # setup opticks: After source: opticks
    === junoenv-external-libs: juno-ext-libs-check-is-reused opticks
    ==== juno-ext-libs-check-is-reused: /home/blyth/junotop/ExternalLibs/opticks/0.1.0
    ==== juno-ext-libs-check-is-reused: The installation prefix for opticks: "/home/blyth/junotop/ExternalLibs/opticks/0.1.0" is a soft link.
    ==== juno-ext-libs-check-is-reused: It can be a reused library.


::

    Available sub commands:
    * all
    * get
    * conf
    * make
    * install
    * setup
    * reuse
    * list



Planting dummies gets all the steps to run::

    touch $JUNOTOP/ExternalLibs/Build/opticks-download-filename-0.1.0
    mkdir -p $JUNOTOP/ExternalLibs/Build/opticks-tardst-0.1.0


ExternalLibs bashrc for setup of paths
----------------------------------------

Generated by::

    652 # helper for setup
    653 function juno-ext-libs-PKG-setup {
    654     local curpkg=$1 # this is the pkg to be intalled.
    655     shift
    656     local msg="===== $FUNCNAME: "
    657     juno-ext-libs-install-root-check || exit $?
    658     pushd $(juno-ext-libs-install-root) >& /dev/null
    659 
    660     if [ ! -d "$(juno-ext-libs-${curpkg}-install-dir)" ]; then
    661         echo $msg Please install the Package first
    662         exit 1
    663     fi
    664     local install=$(juno-ext-libs-${curpkg}-install-dir)
    665     pushd $install
    666     juno-ext-libs-generate-sh $(juno-ext-libs-${curpkg}-name) ${install}
    667     juno-ext-libs-generate-csh $(juno-ext-libs-${curpkg}-name) ${install}
    668     popd
    669 
    670     popd >& /dev/null
    671 }


::

    [blyth@localhost junotop]$ cat ~/local/opticks/bashrc
    if [ -z "${JUNOTOP}" ]; then
    export JUNO_EXTLIB_opticks_HOME=/home/blyth/local/opticks
    else
    export JUNO_EXTLIB_opticks_HOME=${JUNOTOP}/../local/opticks
    fi

    export PATH=${JUNO_EXTLIB_opticks_HOME}/bin:${PATH}
    if [ -d ${JUNO_EXTLIB_opticks_HOME}/lib ];
    then
    export LD_LIBRARY_PATH=${JUNO_EXTLIB_opticks_HOME}/lib:${LD_LIBRARY_PATH}
    fi

"${JUNOTOP}/../local/opticks" looks funny but it is correct::

    076 function juno-ext-libs-generate-sh {
     77 local pkg=$1
     78 local install=$2
     79 local lib=${3:-lib}
     80 local install_wo_top=$(perl -e 'use File::Spec; print File::Spec->abs2rel(@ARGV) . "\n"' $install $JUNOTOP)
     81 
     82 # avoid '-' in $pkg
     83 pkg=${pkg//-/_}
     84 
     85 cat << EOF > bashrc
     86 if [ -z "\${JUNOTOP}" ]; then
     87 export JUNO_EXTLIB_${pkg}_HOME=${install}
     88 else
     89 export JUNO_EXTLIB_${pkg}_HOME=\${JUNOTOP}/${install_wo_top}
     90 fi
     91 
     92 export PATH=\${JUNO_EXTLIB_${pkg}_HOME}/bin:\${PATH}
     93 EOF
    ...
    127     # user defined generate
    128     type -t juno-ext-libs-${curpkg}-generate-sh >& /dev/null
    129     if [ "$?" = 0 ]; then
    130         echo $msg call juno-ext-libs-${curpkg}-generate-sh to generate user defined 
    131         juno-ext-libs-${curpkg}-generate-sh $@
    132     fi
    133 }


* pkg specific env setup that gets appended to the bashrc is done if the below functions are defined::

    juno-ext-libs-opticks-generate-sh
    juno-ext-libs-opticks-generate-csh




junoenv env : just the umbrella script 
---------------------------------------

::

    [blyth@localhost junoenv]$ . junoenv-env.sh
    [blyth@localhost junoenv]$ JUNOENVDIR=$JUNOTOP/junoenv
    [blyth@localhost junoenv]$ junoenv-env-setup-external-libraries-list 
    /home/blyth/junotop/ExternalLibs/Python/2.7.15
    /home/blyth/junotop/ExternalLibs/Boost/1.70.0
    /home/blyth/junotop/ExternalLibs/Cmake/3.15.2
    /home/blyth/junotop/ExternalLibs/Git/1.8.4.3
    /home/blyth/junotop/ExternalLibs/Xercesc/3.2.2
    /home/blyth/junotop/ExternalLibs/gsl/2.5
    /home/blyth/junotop/ExternalLibs/fftw3/3.3.8
    /home/blyth/junotop/ExternalLibs/sqlite3/3.29.0
    /home/blyth/junotop/ExternalLibs/tbb/2019_U8
    /home/blyth/junotop/ExternalLibs/CMT/v1r26
    /home/blyth/junotop/ExternalLibs/CLHEP/2.4.1.0
    /home/blyth/junotop/ExternalLibs/xrootd/4.10.0
    /home/blyth/junotop/ExternalLibs/ROOT/6.18.00
    /home/blyth/junotop/ExternalLibs/HepMC/2.06.09
    /home/blyth/junotop/ExternalLibs/Geant4/10.05.p01
    /home/blyth/junotop/ExternalLibs/libmore/0.8.3
    /home/blyth/junotop/ExternalLibs/mysql-connector-c/6.1.9
    /home/blyth/junotop/ExternalLibs/mysql-connector-cpp/1.1.8
    /home/blyth/junotop/ExternalLibs/libyaml/0.2.2
    /home/blyth/junotop/ExternalLibs/python-yaml/5.1.2
    /home/blyth/junotop/ExternalLibs/podio/master
    [blyth@localhost junoenv]$ 

::

    089 function junoenv-env-setup-external-libraries-list {
    ...
    117     # python boost cmake git xercesc qt4 gsl fftw3 tbb cmt clhep xrootd ROOT hepmc geant4 libmore mysql-connector-c mysql-connector-cpp
    118     for guesspkg in $(junoenv-external-libs-list)
    119     do
    120         guesspkg=$env_scripts_dir/${guesspkg}.sh
    121         source $guesspkg
    122         local pkg_short_name=$(basename $guesspkg)
    123         pkg_short_name="${pkg_short_name%.*}"
    124 
    125         # check the bashrc and tcshrc in the External Libraries
    126         local installdir=$(juno-ext-libs-${pkg_short_name}-install-dir)
    127         if [ -f "$installdir/bashrc" -a -f "$installdir/tcshrc" ]; then
    128             echo $installdir
    129         fi
    130     done
    131 }


::

    [blyth@localhost junoenv]$ touch $(opticks-prefix)/bashrc
    [blyth@localhost junoenv]$ touch $(opticks-prefix)/tcshrc


Ordering comes from this hardcoded list::

    [blyth@localhost junoenv]$ junoenv-external-libs-list
    python boost cmake
    git
    xercesc
    gsl fftw3
    sqlite3
    tbb cmt clhep xrootd ROOT hepmc geant4
    libmore
    libmore-data
    mysql-connector-c mysql-connector-cpp
    libyaml python-yaml
    podio
    [blyth@localhost junoenv]$ 





