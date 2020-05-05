##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

g4-source(){   echo $BASH_SOURCE ; }
g4-vi(){       vi $(g4-source) ; }
g4-usage(){ cat << \EOU

Geant4
========



Breakpoints 
-------------

::

    (lldb) b G4Exception(char const*, char const*, G4ExceptionSeverity, char const*)
      ## break on exceptions


Introductions
--------------

* http://www.niser.ac.in/sercehep2017/notes/serc17_geant4.pdf

  111 slides of Geant4 intro 


Install non-default Geant4 version
------------------------------------

::


   OPTICKS_GEANT4_PREFIX=/usr/local/foreign OPTICKS_GEANT4_NOM=geant4.10.05.p01  g4-info
   OPTICKS_GEANT4_PREFIX=/usr/local/foreign OPTICKS_GEANT4_NOM=geant4.10.05.p01  g4--





Migration to 10 (Multithreaded)
--------------------------------

Quick migration guide for Geant4 version 10.x series

* https://twiki.cern.ch/twiki/bin/view/Geant4/QuickMigrationGuideForGeant4V10

data : G4NDL failed repeatedly, so copy from epsilon
-----------------------------------------------------

::

    epsilon
    /usr/local/opticks/externals/g4/geant4_10_02_p01.Debug.build/Externals
    precise
    /home/blyth/local/opticks/externals/g4/geant4_10_02_p01.Debug.build/Externals

    epsilon:Externals blyth$ scp -r G4NDL-4.5 J:local/opticks/externals/g4/geant4_10_02_p01.Debug.build/Externals/

::

    epsilon:Externals blyth$ du -hs *
     80K	G4ABLA-3.0
     25M	G4EMLOW-6.48
    2.1M	G4ENSDFSTATE-1.2.1
    416M	G4NDL-4.5
    3.1M	G4NEUTRONXS-1.4
    5.1M	G4PII-1.3
     52K	G4SAIDDATA-1.1
    9.1M	PhotonEvaporation-3.2
    8.1M	RadioactiveDecay-4.3.1
    2.1M	RealSurface-1.0
    epsilon:Externals blyth$ 


Not finding xercesc
--------------------

::

    In file included from /usr/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/gdml/include/G4GDMLReadDefine.hh:45:
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/gdml/include/G4GDMLRead.hh:42:10: fatal error: 'xercesc/parsers/XercesDOMParser.hpp' file not found
    #include <xercesc/parsers/XercesDOMParser.hpp>
             ^
    1 error generated.

::

	/opticks/externals/g4/geant4_10_02_p01/source/persistency/gdml/include -I/home/blyth/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/mctruth/include  -W -Wall -pedantic -Wno-non-virtual-dtor -Wno-long-long -Wwrite-strings -Wpointer-arith -Woverloaded-virtual -Wno-variadic-macros -Wshadow -pipe -DG4USE_STD11 -O2 -g -fPIC   -std=c++11 -o CMakeFiles/G4persistency.dir/mctruth/src/G4VPHitsCollectionIO.cc.o -c /home/blyth/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/mctruth/src/G4VPHitsCollectionIO.cc
	gmake[2]: *** No rule to make target `/home/blyth/local/opticks/externals/lib/libxerces-c-3-1.so', needed by `BuildProducts/lib64/libG4persistency.so'.  Stop.
	gmake[2]: Leaving directory `/home/blyth/local/opticks/externals/g4/geant4_10_02_p01.Debug.build'
	gmake[1]: *** [source/persistency/CMakeFiles/G4persistency.dir/all] Error 2
	gmake[1]: Leaving directory `/home/blyth/local/opticks/externals/g4/geant4_10_02_p01.Debug.build'
	gmake: *** [all] Error 2
	-bash: /home/blyth/local/opticks/externals/bin/geant4.sh: No such file or directory
	=== g4-export-ini : writing G4 environment to /home/blyth/local/opticks/externals/config/geant4.ini
	[blyth@localhost geant4_10_02_p01.Debug.build]$ 




Expat
-------

::

    yum install expat-devel



G4 Version Macro
-----------------

::

   find . -name '*.*' -exec grep -H G4VERSION_NUMBER {} \;


Geant4 10.2, December 4th, 2015
----------------------------------

* https://geant4.web.cern.ch/geant4/support/ReleaseNotes4.10.2.html
* https://geant4.web.cern.ch/geant4/support/download.shtml
* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/

G4 Windows dllexport/dllimport ?
-----------------------------------

::

    delta:geant4.10.02 blyth$ find source -name '*.hh' -exec grep -H dll {} \;
    source/g3tog4/include/G3toG4Defs.hh:      #define G3G4DLL_API __declspec( dllexport )
    source/g3tog4/include/G3toG4Defs.hh:      #define G3G4DLL_API __declspec( dllimport )
    source/global/management/include/G4Types.hh:    #define G4DLLEXPORT __declspec( dllexport )
    source/global/management/include/G4Types.hh:    #define G4DLLIMPORT __declspec( dllimport )
    delta:geant4.10.02 blyth$ 


Huh not many of them::

    delta:geant4.10.02 blyth$ find source -name '*.hh' -exec grep -H DLL {} \; | wc -l
         125

See g4win-


Prerequisites
---------------

CMake 3.3 or higher
~~~~~~~~~~~~~~~~~~~~

Minimum required version of CMake to build Geant4 is 3.3. User applications can
still use CMake 2.8.X or above for configuration and compilation. It is however
recommended to migrate to CMake 3.3 or above for its improved support of C++
compile and target features.

Installed via macports, annoyingly required to install java too and entailed
rebuilding several other packages including python, mysql, llvm see ~/macports/cmake33.log

Xcode 6 or higher
~~~~~~~~~~~~~~~~~~

See xcode-vi for install notes


storePhysicsTable
-------------------

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/TrackingAndPhysics/physicsTable.html
* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/Control/UIcommands/_run_particle_.html

...Note that physics tables are calculated before the event loop, not in the initialization phase. 
So, at least one event must be executed before using the "storePhysicsTable" command...

* /run/particle/storePhysicsTable [dirName]
* /run/particle/retrievePhysicsTable [dirName]


* how use this cache ? 


Compiling GDML module
-----------------------

examples/extended/persistency/gdml/G01/README::

  You need to have built the persistency/gdml module by having
  set the -DGEANT4_USE_GDML=ON flag during the CMAKE configuration step, 
  as well as the -DXERCESC_ROOT_DIR=<path_to_xercesc> flag pointing to 
  the path where the XercesC XML parser package is installed in your system.

After adding the above::

    simon:G01 blyth$ g4-cmake
    -- Found XercesC: /opt/local/lib/libxerces-c.dylib  
    -- Reusing dataset G4NDL (4.5)
    -- Reusing dataset G4EMLOW (6.48)
    -- Reusing dataset PhotonEvaporation (3.2)
    -- Reusing dataset RadioactiveDecay (4.3)
    -- Reusing dataset G4NEUTRONXS (1.4)
    -- Reusing dataset G4PII (1.3)
    -- Reusing dataset RealSurface (1.0)
    -- Reusing dataset G4SAIDDATA (1.1)
    -- Reusing dataset G4ABLA (3.0)
    -- Reusing dataset G4ENSDFSTATE (1.2)
    -- The following Geant4 features are enabled:
    GEANT4_BUILD_CXXSTD: Compiling against C++ Standard '11'
    GEANT4_USE_SYSTEM_EXPAT: Using system EXPAT library
    GEANT4_USE_GDML: Building Geant4 with GDML support

    -- Configuring done
    -- Generating done
    -- Build files have been written to: /usr/local/env/g4/geant4.10.02.build
    simon:G01 blyth$ 



Example B1
------------

::

    simon:B1.build blyth$ ./exampleB1 
    Available UI session types: [ GAG, tcsh, csh ]

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : PART70000
          issued by : G4NuclideTable
    G4ENSDFSTATEDATA environment variable must be set
    *** Fatal Exception *** core dump ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------


    *** G4Exception: Aborting execution ***
    Abort trap: 6




EOU
}
g4-env(){      
   olocal-  
   xercesc-  
   opticks-
}

g4-prefix(){ 
   if [ -n "$OPTICKS_GEANT4_HOME" ]; then
       echo $OPTICKS_GEANT4_HOME        # backward compat for a poor name 
   else
       echo ${OPTICKS_GEANT4_PREFIX:-$(opticks-prefix)/externals} 
   fi
}

g4-libsuffix(){ 
    case $NODE_TAG in 
         X) echo 64  ;;
         *) echo -n ;;
    esac
}

g4-idir(){ echo $(g4-prefix) ; }
g4-dir(){   echo $(g4-prefix)/$(g4-tag)/$(g4-name) ; } 

# follow env/psm1/dist/dist.psm1 approach : everythinh based off the url



g4-nom(){ echo ${OPTICKS_GEANT4_NOM:-geant4_10_04_p02} ; }

g4-title()
{
   case $(g4-nom) in 
      Geant4-10.2.1)    echo Geant4 10.2 first released 4 December 2015 \(patch-03, released 27 January 2017\) ;;
      geant4_10_04_p01) echo Geant4 10.4 patch-01, released 28 February 2018 ;; 
      geant4_10_04_p02) echo Geant4 10.4 patch-02, released 25 May 2018 ;; 
   esac
}

g4-version-hh() { echo $(g4-dir)/source/global/management/include/G4Version.hh ; }
g4-version-number() { perl -n -e 'm,#define G4VERSION_NUMBER\s*(\d*), && print $1' $(g4-version-hh) ; } 





g4-nom-notes(){ cat << EON

::

  geant4-9.5.0      # ancient version
  Geant4-10.2.1     # long time default
  Geant4-10.2.2     # used on SDU X node 
  geant4_10_04_p01  # never proceeded with this one
  geant4_10_04_p02  # attempt on E  
  geant4.10.05.b01  # beta, not yet tried

The nom identifier needs to match the name of the folder created by exploding the zip or tarball, 
unfortunately this is not simply connected with the basename of the url and also Geant4 continues to 
reposition URLs and names so these are liable to going stale.

EON
}

g4-url(){   
   case $(g4-nom) in
        Geant4-10.2.1)    echo http://geant4.cern.ch/support/source/geant4_10_02_p01.zip ;;
        Geant4-10.2.2)    echo http://geant4.cern.ch/support/source/geant4_10_02_p02.zip ;;
        geant4-9.5.0)     echo https://github.com/Geant4/geant4/archive/v9.5.0.zip ;;
        geant4_10_04_p01) echo http://geant4-data.web.cern.ch/geant4-data/releases/geant4_10_04_p01.zip  ;; 
        geant4_10_04_p02) echo http://geant4-data.web.cern.ch/geant4-data/releases/geant4.10.04.p02.tar.gz ;; 
        geant4.10.05.b01) echo http://geant4-data.web.cern.ch/geant4-data/releases/geant4.10.05.b01.tar.gz ;; 
        geant4.10.05.p01) echo http://cern.ch/geant4-data/releases/geant4.10.05.p01.tar.gz ;; 
   esac
}

g4-tag(){   echo g4 ; }
g4-dist(){ echo $(dirname $(g4-dir))/$(basename $(g4-url)) ; }

g4-filename(){  echo $(basename $(g4-url)) ; }
g4-name(){  local name=$(g4-filename) ; name=${name/.tar.gz} ; name=${name/.zip} ; echo ${name} ; }  

g4-txt(){ vi $(g4-dir)/CMakeLists.txt ; }

g4-config(){ echo Debug ; }
#g4-config(){ echo RelWithDebInfo ; }

g4-bdir(){ echo $(g4-dir).$(g4-config).build ; }

g4-cmake-dir(){     echo $(g4-prefix)/lib$(g4-libsuffix)/$(g4-nom) ; }
g4-examples-dir(){  echo $(g4-prefix)/share/$(g4-nom)/examples ; }
g4-gdml-dir(){      echo $(g4-dir)/source/persistency/gdml ; }
g4-optical-dir(){   echo $(g4-dir)/source/processes/optical/src ; }

g4-c(){    cd $(g4-dir); }
g4-cd(){   cd $(g4-dir); }
g4-icd(){  cd $(g4-prefix); }
g4-bcd(){  cd $(g4-bdir); }
g4-ccd(){  cd $(g4-cmake-dir); }
g4-xcd(){  cd $(g4-examples-dir); }

g4-gcd(){  cd $(g4-gdml-dir); }
g4-ocd(){  cd $(g4-optical-dir); }

g4-get-info(){ cat << EOI

   g4-dir : $(g4-dir)
   g4-nom : $(g4-nom)
   g4-url : $(g4-url)

EOI
}


g4-get(){
   local dir=$(dirname $(g4-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4-url)
   local dst=$(basename $url)
   local nam=$dst
   nam=${nam/.zip}
   nam=${nam/.tar.gz}

   g4-get-info

   [ ! -f "$dst" ] && echo getting $url && curl -L -O $url

   if [ "${dst/.zip}" != "${dst}" ]; then
        [ ! -d "$nam" ] && unzip $dst
   fi
   if [ "${dst/.tar.gz}" != "${dst}" ]; then
        [ ! -d "$nam" ] && tar zxvf $dst
   fi

   [ -d "$nam" ]
}

g4--()
{
    local msg="=== $FUNCNAME :"
    g4-get
    [ $? -ne 0 ] && echo $msg get FAIL && return 1
    g4-configure 
    [ $? -ne 0 ] && echo $msg configure FAIL && return 2
    g4-build
    [ $? -ne 0 ] && echo $msg build FAIL && return 3
    g4-export-ini
    [ $? -ne 0 ] && echo $msg export-ini FAIL && return 4
    g4-pc
    [ $? -ne 0 ] && echo $msg pc FAIL && return 5

    return 0 
}


g4-wipe(){
   local bdir=$(g4-bdir)
   rm -rf $bdir
}

g4-configure()
{
   local 
   local bdir=$(g4-bdir)
   [ -f "$bdir/CMakeCache.txt" ] && g4-configure-msg  && return

   g4-cmake $*
}

g4-configure-msg(){ cat << EOM
g4 has been configured already, to change configuration use eg: g4-cmake-modify-xercesc 
or to start over use : g4-wipe then g4-configure 
EOM
}



g4-info(){ cat << EOI

   g4-tag          : $(g4-tag)
   g4-url          : $(g4-url)
   g4-dist         : $(g4-dist)
   g4-filename     : $(g4-filename)
   g4-name         : $(g4-name)
   g4-nom          : $(g4-nom)
   g4-title        : $(g4-title)
   g4-version-number : $(g4-version-number)

   g4-prefix       : $(g4-prefix) 
   g4-cmake-dir    : $(g4-cmake-dir)
   g4-examples-dir : $(g4-examples-dir)
   g4-bdir         : $(g4-bdir)
   g4-dir          : $(g4-dir)

EOI
}



g4-cmake-info(){ cat << EOI
$FUNCNAME
===============

   opticks-cmake-generator : $(opticks-cmake-generator)

Expected locations:

   xercesc-library-        : $(xercesc-library-)
   xercesc-include-dir-    : $(xercesc-include-dir-)

Possibly overridden results using the below envvars:

   xercesc-library         : $(xercesc-library)
   xercesc-include-dir     : $(xercesc-include-dir)
 
   OPTICKS_XERCESC_LIBRARY     : $OPTICKS_XERCESC_LIBRARY  
   OPTICKS_XERCESC_INCLUDE_DIR : $OPTICKS_XERCESC_INCLUDE_DIR


   g4-prefix               : $(g4-prefix) 
   g4-bdir                 : $(g4-bdir) 
   g4-dir                  : $(g4-dir) 



EOI
}

g4-whats-installed-where(){ cat << EON

Intallation paths with g4-prefix of /usr/local/opticks/externals 

/usr/local/opticks/externals/lib/
    libG4* get mingled with other externals lib.
    Convenient, but a bit messy when changing Geant4 versions.

/usr/local/opticks/externals/lib/Geant4-10.4.2
    cmake machinery, including crucial Geant4Config.cmake 
    that is found by CMake via CMAKE_PREFIX_PATH::

       -DCMAKE_PREFIX_PATH=$(om-prefix)/externals

/usr/local/opticks/externals/include/Geant4/
    Found mixed timestamp versions of installed headers
    so deleted the include/Geant4 and reinstalled with g4-build


/usr/local/opticks/externals/share/Geant4-10.4.2/data/
/usr/local/opticks/externals/share/Geant4-10.4.2/examples
    data and examples are placed under a version dir 

/usr/local/opticks/externals/bin/
    some unversioned scripts:
    geant4.sh 
    geant4.csh
    geant4-config






EON
}



g4-cmake(){
   local iwd=$PWD
   local rc=0

   local bdir=$(g4-bdir)
   mkdir -p $bdir

   local idir=$(g4-prefix)
   mkdir -p $idir

   g4-cmake-info

   g4-bcd

   cmake \
       -G "$(opticks-cmake-generator)" \
       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
       -DGEANT4_INSTALL_DATA=ON \
       -DGEANT4_USE_GDML=ON \
       -DXERCESC_LIBRARY=$(xercesc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
       -DCMAKE_INSTALL_PREFIX=$idir \
       $(g4-dir)

   rc=$?

   cd $iwd
   return $rc
}

g4-cmake-modify(){
   local iwd=$PWD
   local msg="=== $FUNCNAME : "
   local bdir=$(g4-bdir)
   local bcache=$bdir/CMakeCache.txt
   [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior g4-cmake run && return
   g4-bcd

   cmake $* . 

   cd $iwd
}

g4-cmake-modify-xercesc()
{
   xercesc-
   g4-cmake-modify \
      -DXERCESC_LIBRARY=$(xercesc-library) \
      -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) 
}

g4-cmake-modify-xercesc-system()
{
   g4-cmake-modify \
      -DXERCESC_LIBRARY=$(xercesc-library) \
      -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) 
}

g4-build(){
   local rc=0
   g4-bcd
   date
   cmake --build . --config $(g4-config) --target ${1:-install}
   rc=$?
   date
   return $rc
}

g4-sh(){   echo $(g4-idir)/bin/geant4.sh ; }

g4-configdir(){ echo ${G4_CONFIGDIR:-$(opticks-prefix)/externals/config} ; }

g4-export(){ source $(g4-sh) ; }


g4-export-ini-notes(){ cat << EON

geant4.ini
    has tokenized paths for the G4..DATA envvars allowing relocation of geant4 data 


EON
}


g4-export-ini()
{
    local msg="=== $FUNCNAME :"
    local dir=$(g4-configdir)
    mkdir -p $dir 
    echo $msg writing G4 environment into dir $dir

    g4-export

    $(opticks-home)/bin/envg4.py  > $dir/geant4.ini
    cat $dir/geant4.ini

    g4-envg4
}



g4-envg4-notes(){ cat << EON

g4-envg4
    generates opticks-envg4.bash that exports the G4..DATA envvars and appends to LD_LIBRARY_PATH
    This is done using BASH_SOURCE which makes the Geant4 installation relocatable, 
    assuming the relative paths between which script, data and libs are kept the same.

    Then can relocatably setup library and data access with 
    a single absolute path upfront::

        source /path/to/opticks-envg4.bash

EON
}

g4-envg4()
{
    local msg="=== $FUNCNAME :"
    local externals=$OPTICKS_PREFIX/externals
    local script="opticks-envg4.bash"
    echo $msg writing script $script to externals $externals/$script
    $(opticks-home)/bin/envg4.py --token here  --prefix $externals --bash  > $externals/$script
    cat $externals/$script
}



################# below funcions for studying G4 source ##################################

g4-ifind(){ find $(g4-idir) -name ${1:-G4VUserActionInitialization.hh} ; }
g4-sfind(){ find $(g4-dir)/source -name ${1:-G4VUserActionInitialization.hh} ; }

g4-hh(){ find $(g4-dir)/source -name '*.hh' -exec grep -H "${1:-G4GammaConversion}" {} \; ; }
g4-icc(){ find $(g4-dir)/source -name '*.icc' -exec grep -H "${1:-G4GammaConversion}" {} \; ; }
g4-cc(){ find $(g4-dir)/source -name '*.cc' -exec grep -H "${1:-G4GammaConversion}" {} \; ; }

g4-cls-copy(){
   local iwd=$PWD
   local name=${1:-G4Scintillation}
   local lname=${name/G4}

   local sauce=$(g4-dir)/source
   local hh=$(find $sauce -name "$name.hh")
   local cc=$(find $sauce -name "$name.cc")
   local icc=$(find $sauce -name "$name.icc")

   [ "$hh" != "" ]  && echo cp $hh $iwd/$lname.hh
   [ "$cc" != "" ] && echo cp $cc $iwd/$lname.cc
   [ "$icc" != "" ] && echo cp $icc $iwd/$lname.icc
}

g4-cls-copyv-notes(){ cat << EON

Making a versioned copy of a G4 class::

    epsilon:cfg4 blyth$ g4-cls-copyv G4Cerenkov 
    cp /usr/local/opticks/externals/g4/geant4.10.04.p02/source/processes/electromagnetic/xrays/include/G4Cerenkov.hh /Users/blyth/opticks/cfg4/G4Cerenkov1042.hh
    cp /usr/local/opticks/externals/g4/geant4.10.04.p02/source/processes/electromagnetic/xrays/src/G4Cerenkov.cc /Users/blyth/opticks/cfg4/G4Cerenkov1042.cc

    epsilon:cfg4 blyth$ g4-cls-copyv G4Cerenkov | sh 
    epsilon:cfg4 blyth$ perl -pi -e 's,G4Cerenkov,G4Cerenkov1042,g' G4Cerenkov1042.*

    ## then make another to have local mods

    perl -pi -e 's,G4Cerenkov1042,C4Cerenkov1042,g' C4Cerenkov1042.*




EON
}


g4-cls-copyv(){
   local iwd=$PWD
   local name=${1:-G4Scintillation}
   local number=$(g4-version-number)
   local lname=${name}${number}

   local sauce=$(g4-dir)/source
   local hh=$(find $sauce -name "$name.hh")
   local cc=$(find $sauce -name "$name.cc")
   local icc=$(find $sauce -name "$name.icc")

   [ "$hh" != "" ]  && echo cp $hh $iwd/$lname.hh
   [ "$cc" != "" ] && echo cp $cc $iwd/$lname.cc
   [ "$icc" != "" ] && echo cp $icc $iwd/$lname.icc
}





g4-cls(){  
   local iwd=$PWD
   g4-cd 
   g4-cls- source    $* ; 
   cd $iwd
}


g4-cls-(){
   local base=${1:-source}
   local name=${2:-G4Scintillation}

   local h=$(find $base -name "$name.h")
   local hh=$(find $base -name "$name.hh")
   local cc=$(find $base -name "$name.cc")
   local icc=$(find $base -name "$name.icc")
   local src=$(find $base -name "$name.src")

   local cc2=""
   if [ "$name" == "G4SteppingManager" ] ; then
       cc2=$(find $base -name "${name}2.cc")
   fi  

   local vcmd="vi -R $h $hh $icc $cc $src $cc2"
   echo $vcmd
   eval $vcmd
}




g4-look-info(){ cat << EOI
g4-look : vim -R at line of G4 source file
==================================================

Examples::

   g4-look G4RunManagerKernel.cc:707  

EOI
}

g4-look(){ 
   local iwd=$PWD
   g4-cd
   local spec=${1:-G4RunManagerKernel.cc:707}

   local name=${spec%:*}
   local line=${spec##*:}
   [ "$line" == "$spec" ] && line=1

   local fcmd="find source -name $name"
   local path=$($fcmd)

   echo $spec $name $line $path 

   if [ "$path" == "" ]; then 
      echo $msg FAILED to find $name with : $fcmd
      return 
   fi 
   local vcmd="vi -R $path +$line"
   echo $vcmd
   eval $vcmd
     
   cd $iwd
}


g4-find-(){ find $(g4-dir) $* ; }
g4-find-gdml(){ g4-find- -name '*.gdml' ; } 


g4-pc-path(){ echo $(opticks-prefix)/externals/lib/pkgconfig/G4.pc ; }

g4-libs--(){ cat << EOL
G4Tree;G4FR;G4GMocren;G4visHepRep;G4RayTracer;G4VRML;G4vis_management;G4modeling;G4interfaces;G4persistency;G4analysis;G4error_propagation;G4readout;G4physicslists;G4run;G4event;G4tracking;G4parmodels;G4processes;G4digits_hits;G4track;G4particles;G4geometry;G4materials;G4graphics_reps;G4intercoms;G4global;G4clhep;G4zlib
EOL
}


g4-libs-(){ g4-libs-- | tr ";" "\n" ; }
g4-libs(){
 g4-libs- | while read lib ; do
    printf "%s " "-l$lib" 
 done   
}


g4-libdir-(){ cat << EOD
externals/lib64
externals/lib
EOD
}

g4-libdir(){
   local rdir
   local dir
   $FUNCNAME- | while read rdir ; do 
      dir=$(opticks-prefix)/$rdir
      [ -f "$dir/libG4Tree.dylib" -o -f "$dir/libG4Tree.so" ] && echo $rdir
   done
}

g4-pc-(){ cat << EOP

# $FUNCNAME $(date)

prefix=$(opticks-prefix)
includedir=\${prefix}/externals/include/Geant4
libdir=\${prefix}/$(g4-libdir)

Name: Geant4
Description: 
Version: 
Libs: -L\${libdir} $(g4-libs) -lstdc++
Cflags: -I\${includedir}


EOP
}
g4-pc(){
    local msg="=== $FUNCNAME :";
    local path=$(g4-pc-path);
    local dir=$(dirname $path);
    [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p $dir;
    echo $msg $path 
    g4-pc- > $path
}






g4-pcc-(){ 
   local config=$1
   cat << EOP

prefix=$(cd $($config --prefix) ; pwd)

Name: Geant4
Description:
Version: $($config --version)
Libs: $($config --libs) -lstdc++
Cflags: $($config --cflags)

# on Darwin with clang should be -lc++ but I think there is some compat to make -lstdc++ work 

EOP
}

g4-pcc-path(){
   local prefix=$1
   if [ -d "$prefix/lib64" ]; then  
      echo $prefix/lib64/pkgconfig/geant4.pc 
   elif [ -d "$prefix/lib" ]; then  
      echo $prefix/lib/pkgconfig/geant4.pc 
   fi   
}

g4-pcc()
{
    local index=${1:-0}
    local msg="=== $FUNCNAME :"
    local prefix=$(find_package.py Geant4 --prefix --index $index)

    local config=$prefix/bin/geant4-config
    local prefix2
    if [ -x "$config" ]; then 
        prefix2=$(cd $($config --prefix) ; pwd )
        if [ "$prefix" != "$prefix2" ]; then 
           echo $msg ERROR prefix inconsitent $prefix $prefix2 
        fi  
    fi 

    local path=$(g4-pcc-path $prefix)

    local dir=$(dirname $path)
    [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p $dir;

    echo $msg writing to path $path 
    g4-pcc- $config  > $path
}


g4-pcc-all () 
{ 
    : NOT A GOOD APPROACH : AS THE PC WILL GO STALE WHEN PACKAGES ARE REINSTALLED;
    : BETTER TO PLANT THE PC AS PART OF THE INSTALLATION;
    local pkg=Geant4
    local num=$(find_package.py $pkg --count);
    if [ "$num" == "0" ]; then
        echo $msg FAILED to find_package.py $pkg;
        return;
    fi;
    local idxs=$(seq 0 $(( $num - 1 )) | tr "\n" " ");
    for idx in $idxs;
    do
        echo g4-pcc $idx;
        g4-pcc $idx;
    done
}



g4-setup(){ cat << EOS
# $FUNCNAME
EOS
}

