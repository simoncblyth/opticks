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




GDML auxiliary
---------------

* https://github.com/hanswenzel/G4OpticksTest/blob/master/gdml/G4Opticks.gdml

::


    166         <volume name="Obj">
    167             <materialref ref="LS0x4b61c70"/>
    168             <solidref ref="Obj"/>
    169             <colorref ref="blue"/>
    170             <auxiliary auxtype="StepLimit" auxvalue="0.4" auxunit="mm"/>
    171             <auxiliary auxtype="SensDet" auxvalue="lArTPC"/>
    172             <physvol name="Det">
    173                 <volumeref ref="Det"/>
    174                 <position name="Det" unit="mm" x="0" y="0" z="100"/>
    175             </physvol>
    176         </volume>




::

    epsilon:geant4.10.04.p02 blyth$ g4-hh G4GDMLAuxMapType
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:typedef std::map<G4LogicalVolume*,G4GDMLAuxListType> G4GDMLAuxMapType;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:   const G4GDMLAuxMapType* GetAuxMap() const {return &auxMap;}
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:   G4GDMLAuxMapType auxMap;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLParser.hh:   inline const G4GDMLAuxMapType* GetAuxMap() const;
    epsilon:geant4.10.04.p02 blyth$ 


    epsilon:geant4.10.04.p02 blyth$ g4-hh G4GDMLAuxMapType
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:typedef std::map<G4LogicalVolume*,G4GDMLAuxListType> G4GDMLAuxMapType;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:   const G4GDMLAuxMapType* GetAuxMap() const {return &auxMap;}
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:   G4GDMLAuxMapType auxMap;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLParser.hh:   inline const G4GDMLAuxMapType* GetAuxMap() const;
    epsilon:geant4.10.04.p02 blyth$ g4-hh G4GDMLAuxListType
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLRead.hh:   const G4GDMLAuxListType* GetAuxList() const;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLRead.hh:   G4GDMLAuxListType auxGlobalList;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLAuxStructType.hh:typedef std::vector<G4GDMLAuxStructType> G4GDMLAuxListType;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLWriteStructure.hh:   std::map<const G4LogicalVolume*, G4GDMLAuxListType> auxmap;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:typedef std::map<G4LogicalVolume*,G4GDMLAuxListType> G4GDMLAuxMapType;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:   G4GDMLAuxListType GetVolumeAuxiliaryInformation(G4LogicalVolume*) const;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLParser.hh:   inline G4GDMLAuxListType GetVolumeAuxiliaryInformation(G4LogicalVolume* lvol) const;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLParser.hh:   inline const G4GDMLAuxListType* GetAuxList() const;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLParser.hh:   G4GDMLAuxListType *rlist, *ullist;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLWrite.hh:    void AddAuxInfo(G4GDMLAuxListType* auxInfoList, xercesc::DOMElement* element);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLWrite.hh:    G4GDMLAuxListType auxList;
    epsilon:geant4.10.04.p02 blyth$ 





g4-cls G4GDMLWriteStructure::

    95    std::map<const G4LogicalVolume*, G4GDMLAuxListType> auxmap;

    580 void
    581 G4GDMLWriteStructure::AddVolumeAuxiliary(G4GDMLAuxStructType myaux,
    582                                          const G4LogicalVolume* const lvol)
    583 {
    584   std::map<const G4LogicalVolume*,
    585            G4GDMLAuxListType>::iterator pos = auxmap.find(lvol);
    586 
    587   if (pos == auxmap.end())  { auxmap[lvol] = G4GDMLAuxListType(); }
    588 
    589   auxmap[lvol].push_back(myaux);
    590 }

g4-cls G4GDMLAuxStructType::

    042 struct G4GDMLAuxStructType
     43 {
     44    G4String type;
     45    G4String value;
     46    G4String unit;
     47    std::vector<G4GDMLAuxStructType>* auxList;
     48 };
     49 
     50 typedef std::vector<G4GDMLAuxStructType> G4GDMLAuxListType;

g4-cls G4GDMLParser::

    119    inline G4VPhysicalVolume* GetWorldVolume(const G4String& setupName="Default") const;
    120    inline G4GDMLAuxListType GetVolumeAuxiliaryInformation(G4LogicalVolume* lvol) const;
    121    inline const G4GDMLAuxMapType* GetAuxMap() const;
    122    inline const G4GDMLAuxListType* GetAuxList() const;
    123    inline void AddAuxiliary(G4GDMLAuxStructType myaux);




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


Geant4 1062 : requires minimum of gcc 4.9.3
-----------------------------------------------

Iterator erase compilation failure with gcc 4.8.5::

    [ 12%] Building CXX object source/geometry/CMakeFiles/G4geometry.dir/navigation/src/G4ParameterisedNavigation.cc.o
    /home/blyth/local/opticks_externals/g4_1062.build/geant4.10.06.p02/source/geometry/management/src/G4SolidStore.cc: In static member function ‘static void G4SolidStore::DeRegister(G4VSolid*)’:
    /home/blyth/local/opticks_externals/g4_1062.build/geant4.10.06.p02/source/geometry/management/src/G4SolidStore.cc:141:49: error: no matching function for call to ‘G4SolidStore::erase(std::reverse_iterator<__gnu_cxx::__normal_iterator<G4VSolid* const*, std::vector<G4VSolid*> > >::iterator_type)’
             GetInstance()->erase(std::next(i).base());
                                                     ^
    [blyth@localhost ~]$ l /usr/include/c++/
    total 8
    lrwxrwxrwx.  1 root root    5 Mar 23  2020 4.8.5 -> 4.8.2
    drwxr-xr-x. 12 root root 4096 Mar 23  2020 4.8.2


* https://geant4-forum.web.cern.ch/t/error-when-making-geant4/1774

gcosmo::

    You must use a more recent gcc compiler to build Geant4 10.6.
    The minimum required is gcc-4.9.3 and you’re using gcc-4.8.2…i

See env-;centos- for notes on yum installation of devtoolset-9 which comes with gcc 9.3.1 



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


Compiling GDML module
-----------------------

examples/extended/persistency/gdml/G01/README::

  You need to have built the persistency/gdml module by having
  set the -DGEANT4_USE_GDML=ON flag during the CMAKE configuration step, 
  as well as the -DXERCESC_ROOT_DIR=<path_to_xercesc> flag pointing to 
  the path where the XercesC XML parser package is installed in your system.



EOU
}
g4-env(){      
   olocal-  
   xercesc-  
   opticks-
}



g4-prefix-notes(){ cat << EON

OPTICKS_GEANT4_PREFIX

    envvar set by $OPTICKS_PREFIX/bin/opticks-setup.sh
    based on first Geant4Config.cmake found in the CMAKE_PREFIX_PATH 
    (using opticks-setup-find-geant4-prefix)
    The om-/oe- machinery sources $OPTICKS_PREFIX/bin/opticks-setup.sh

    The opticks-setup.sh script is generated by opticks-full/opticks-setup-generate 
    this setup script encapsulates the externals that an Opticks build used. 
    Because of this it should not be edited, but rather the script needs to 
    be regenerated at Opticks build time.

EON
}


g4-libsuffix(){ 
    case $(uname) in 
         Linux) echo 64  ;;
        Darwin) echo -n ;;
    esac
}


g4-info(){ cat << EOI

  g4-prefix             : $(g4-prefix)
      installation directory 
  OPTICKS_GEANT4_PREFIX : $OPTICKS_GEANT4_PREFIX

  g4-libsuffix          : $(g4-libsuffix)

  g4-dir                : $(g4-dir)
      exploded distribution dir

  g4-bdir               : $(g4-bdir)
  
  g4-nom                : $(g4-nom)
  g4-title              : $(g4-title)

  g4-version-hh         : $(g4-version-hh)
  g4-version-number     : $(g4-version-number)

  g4-url                : $(g4-url) 
      case statement based on g4-nom      

  g4-filename           : $(g4-filename)
      basename of url 

  g4-name               : $(g4-name) 
      from basename of url

  g4-dist               : $(g4-dist)
      distribution tarball 


  g4-pc-path            : $(g4-pc-path) 


EOI
}

g4-ver(){    echo ${OPTICKS_GEANT4_VER:-1042} ; }
g4-prefix(){ echo ${OPTICKS_GEANT4_PREFIX:-$(opticks-prefix)_externals/g4_$(g4-ver)}  ; }

g4-dir(){   echo $(g4-prefix).build/$(g4-name) ; }  # exploded distribution dir

g4-nom(){ 
  case $(g4-ver) in 
     1021) echo Geant4-10.2.1 ;;
     1041) echo geant4_10_04_p01 ;; 
     1042) echo geant4_10_04_p02 ;;
     1062) echo geant4.10.06.p02 ;;
  esac
}

g4-nom-notes(){ cat << EON

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
        geant4.10.06.p02) echo http://cern.ch/geant4-data/releases/geant4.10.06.p02.tar.gz ;;

   esac
}

g4-title()
{
   case $(g4-ver) in 
      1021) echo Geant4 10.2 first released 4 December 2015 \(patch-03, released 27 January 2017\) ;;
      1041) echo Geant4 10.4 patch-01, released 28 February 2018 ;; 
      1042) echo Geant4 10.4 patch-02, released 25 May 2018 ;; 
   esac
}

g4-version-hh() { echo $(g4-dir)/source/global/management/include/G4Version.hh ; }
g4-version-number() { perl -n -e 'm,#define G4VERSION_NUMBER\s*(\d*), && print $1' $(g4-version-hh) ; } 


g4--1062(){ OPTICKS_GEANT4_VER=1062 g4-- ; }
g4--1042(){ OPTICKS_GEANT4_VER=1042 g4-- ; }



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
    # g4-export-ini
    # [ $? -ne 0 ] && echo $msg export-ini FAIL && return 4
    g4-pc
    [ $? -ne 0 ] && echo $msg pc FAIL && return 5

    return 0 
}




g4-wipe(){
   g4-wipe-build
   g4-wipe-install
}

g4-wipe-build(){
   local bdir=$(g4-bdir)
   rm -rf $bdir
}

g4-wipe-install(){
   local iwd=$PWD

   cd $(g4-prefix)
   local suffix=$(g4-libsuffix) 

   rm -rf lib$suffix/cmake/Geant4*    ## ??

   rm -f lib$suffix/pkgconfig/geant4.pc 
   rm -f lib$suffix/libG4*    
   rm -rf lib$suffix/Geant4-*
   rm -rf include/Geant4
   rm -rf share/Geant4-*
   rm -f bin/geant4.*
   rm -f bin/geant4-config


   cd $iwd
}



g4-optional-hookup(){

    if [ -z "$CMAKE_PREFIX_PATH" ]; then 
        export CMAKE_PREFIX_PATH=$(g4-prefix)
    else
        export CMAKE_PREFIX_PATH=$(g4-prefix):$CMAKE_PREFIX_PATH
    fi 

    if [ -z "$PKG_CONFIG_PATH" ]; then 
        export PKG_CONFIG_PATH=$(g4-prefix)
    else
        export PKG_CONFIG_PATH=$(g4-prefix)/lib:$PKG_CONFIG_PATH
    fi 

    case $(uname) in 
       Darwin) libpathvar=DYLD_LIBRARY_PATH ;; 
        Linux) libpathvar=LD_LIBRARY_PATH ;; 
    esac

    if [ -z "${!libpathvar}" ]; then 
        export ${libpathvar}=$(g4-prefix)/lib
    else
        export ${libpathvar}=$(g4-prefix)/lib:${!libpathvar}
    fi 
}



g4-configure()
{
   local bdir=$(g4-bdir)
   [ -f "$bdir/CMakeCache.txt" ] && g4-configure-msg  && return

   g4-cmake $*
}

g4-configure-msg(){ cat << EOM
g4 has been configured already, to change configuration use eg: g4-cmake OR g4-cmake-modify-xercesc 
or to start over use : g4-wipe then g4-configure 
EOM
}



g4-cmake-info(){ cat << EOI
$FUNCNAME
===============

   cmake \\ 
       -G "$(opticks-cmake-generator)" \\
       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \\
       -DGEANT4_INSTALL_DATA=ON \\ 
       -DGEANT4_USE_GDML=ON \\
       -DGEANT4_USE_SYSTEM_CLHEP=ON \\ 
       -DGEANT4_INSTALL_DATA_TIMEOUT=3000  \\
       -DXERCESC_LIBRARY=$(xercesc-pc-library) \\
       -DXERCESC_INCLUDE_DIR=$(xercesc-pc-includedir) \\
       -DCMAKE_INSTALL_PREFIX=$(g4-prefix) \\
       $(g4-dir)                                   


   opticks-cmake-generator : $(opticks-cmake-generator)
   opticks-buildtype       : $(opticks-buildtype)
   xercesc-pc-library      : $(xercesc-pc-library)
   xercesc-pc-includedir   : $(xercesc-pc-includedir)
   g4-prefix               : $(g4-prefix)
   g4-dir                  : $(g4-dir)

EOI
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
       -DGEANT4_USE_SYSTEM_CLHEP=ON \
       -DGEANT4_INSTALL_DATA_TIMEOUT=3000 \
       -DXERCESC_LIBRARY=$(xercesc-pc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-pc-includedir) \
       -DCMAKE_INSTALL_PREFIX=$idir \
       $(g4-dir)

   # NB need a clhep prefix on CMAKE_PREFIX_PATH when using GEANT4_USE_SYSTEM_CLHEP=ON
   # default GEANT4_INSTALL_DATA_TIMEOUT is 1500s (25min), doubled that 

   rc=$?

#       -DXERCESC_ROOT_DIR=$(xercesc-prefix) \
# huh:this fails to find it 

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
      -DXERCESC_LIBRARY=$(xercesc-pc-library) \
      -DXERCESC_INCLUDE_DIR=$(xercesc-pc-includedir) 
}

g4-cmake-modify-xercesc-prefix()
{
   xercesc-
   g4-cmake-modify \
      -DXERCESC_ROOT_DIR=$(xercesc-prefix) 

   : observe that this fails to find the xerces-c under the prefix but the above library and includedir approach does work
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



g4-export(){ source $(g4-prefix)/bin/geant4.sh ; }

g4-export-ini-notes(){ cat << EON

geant4.ini
    has tokenized paths for the G4..DATA envvars allowing relocation of geant4 data 


EON
}


g4-inipath(){ echo $(g4-prefix)/bin/geant4.ini ; }

g4-export-ini()
{
    local msg="=== $FUNCNAME :"

    local inipath=$(g4-inipath)
    local dir=$(dirname $inipath)
    mkdir -p $dir 
    echo $msg writing G4 environment into $inipath 

    g4-export

    $(opticks-home)/bin/envg4.py  > $inipath
    cat $inipath

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


    TODO : eliminate this, just use the standard geant4 env setup, 
           despite clunkiness it integrates easier

EON
}

g4-envg4-path(){ echo $(g4-prefix)/opticks-envg4.bash ; }
g4-envg4()
{
    local msg="=== $FUNCNAME :"
    local prefix=$(g4-prefix)
    local path=$(g4-envg4-path)
    echo $msg writing $path
    $(opticks-home)/bin/envg4.py --token here  --prefix $prefix --bash  > $path
    cat $path
}



################# below funcions for studying G4 source ##################################

g4-ifind(){ find $(g4-prefix) -name ${1:-G4VUserActionInitialization.hh} ; }
g4-sfind(){ find $(g4-dir)/source -name ${1:-G4VUserActionInitialization.hh} ; }

g4-hh(){ find $(g4-dir)/source -name '*.hh' -exec grep -H "${1:-G4GammaConversion}" {} \; ; }
g4-icc(){ find $(g4-dir)/source -name '*.icc' -exec grep -H "${1:-G4GammaConversion}" {} \; ; }
g4-cc(){ find $(g4-dir)/source -name '*.cc' -exec grep -H "${1:-G4GammaConversion}" {} \; ; }

g4-cls-copy(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local name=${1:-G4Cerenkov}
   local number=$(g4-version-number)
   local lname=Local${name}${number}

   local src=$(g4-dir)/source
   local hh=$(find $src -name "$name.hh")
   local cc=$(find $src -name "$name.cc")
   local icc=$(find $src -name "$name.icc")

   local tt="hh cc icc"
   local t
   for t in $tt ; do 
      [ "${!t}" == "" ] && continue
      local p=$lname.$t
      printf "%3s %20s %s\n"  $t ${!t} $p
       
      echo "// $FUNCNAME : ${!t}"        > $p
      perl -pe "s,$name,$lname,g" ${!t} >> $p
      echo "// $FUNCNAME : ${!t}"       >> $p
   done 
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



g4-libsuffix(){
   local suffix
   case $(uname) in 
      Linux) suffix=64 ;; 
     Darwin) suffix=   ;;
   esac
   echo $suffix
}

g4-pc-notes(){ cat << EON

Standard Geant4 does not install a geant4.pc file, 
the g4-pc function attempts to create the missing G4.pc
file that corresponds exactly to what geant4-config gives.

EON
}

g4-pc-path(){ echo $(g4-prefix)/lib$(g4-libsuffix)/pkgconfig/Geant4.pc ; }
g4-pcfiledir(){ pkg-config --variable=pcfiledir Geant4 ; }




g4-generate-pcfile(){

   local prefix=${1:-$(g4-prefix)}

   local lib 
   if [ -d "$prefix/lib64" ]; then
      lib="lib64"
   elif [ -d "$prefix/lib" ]; then
      lib="lib"
   fi  

   if [ ! -d "$prefix/$lib/pkgconfig" ]; then 
       mkdir -p $prefix/$lib/pkgconfig  
   fi  

   local config=$prefix/bin/geant4-config
   local prefix0=$($config --prefix)
   local prefix1=$(cd $prefix0 ; pwd)

   local path=$prefix/$lib/pkgconfig/Geant4.pc
   echo generate $path

   cat << EOF > $path

# $FUNCNAME $(date)
# prefix $prefix 
# prefix0 $prefix0 
# prefix1 $prefix1 

prefix=$prefix
includedir=\${prefix}/include/Geant4
libdir=\${prefix}/$lib

Name: Geant4
Description: PC generated from geant4-config outputs 
Version: $($config --version)
Libs: $($config --libs) 
Cflags: $($config --cflags)

# Requires: clhelp : not needed as CLHEP config included in above

EOF

}

g4-pc(){ g4-generate-pcfile $* ; }


g4-pco-path(){ echo $(opticks-prefix)/externals/lib/pkgconfig/G4.pc ; }
g4-pco(){

   local msg="=== $FUNCNAME :"
   local name=geant4
   local pcfiledir=$(pkg-config --variable=pcfiledir $name)
   local path=$pcfiledir/$name.pc 
   local path2=$(g4-pco-path)
   local dir=$(dirname $path2)
   if [ ! -d "$dir" ]; then
       mkdir -p $dir   
   fi  

   if [ -f "$path" -a ! -f "$path2" ]; then
       echo $msg copy $path to $path2
       cp $path $path2  
   elif [ -f "$path2" ]; then 
       echo $msg path2 $path2 exists already
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

