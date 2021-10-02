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



1100_beta Updates in optical physics, Daren Sawkey, Geant4 Collaboration Meeting Sept 22, 2021
-----------------------------------------------------------------------------------------------

* https://indico.cern.ch/event/1052654/contributions/4528557/attachments/2313908/3939383/sawkey_optical.pdf


Install non-default Geant4 version
------------------------------------

1. With a separate user account (using S "simon" account on Precision for this) as liable to long standing breakage.
   It is convenient to use a symbolically linked "opticks" folder such that the opticks source is the same as with 
   standard versions of externals used from other user accounts but the install directories and external prefix 
   environment are different.

2. In the config for the non-standard user comment prefix setup for G4, usually in ~/.opticks_config
   as well as the opticks setup line. The idea is to return to a pre-opticks and pre-geant4 environment.

   * NB often new versions of Geant4 will require new versions of CLHEP also, the clhep- functions can be used 

3. Open a new session with the modified environment without G4 and Opticks
4. Check that the g4.bash functions have everything needed for the new version, including the new url::

   ##g4-;OPTICKS_GEANT4_PREFIX=/home/simon/local/opticks_externals/g4_1072 OPTICKS_GEANT4_VER=1072  g4-info
   ##  actually its safer not to override like this, better to just change the environment setup and use it:

   g4-;g4-info

* check that g4-info gives no errors, fix problems by g4.bash updates over in O account 
* check values dumped by g4-info are as expected for the Geant4 version
* check install directories are writable from this account 

5. Proceed with the download and build::

   ##g4-;OPTICKS_GEANT4_PREFIX=/home/simon/local/opticks_externals/g4_1072 OPTICKS_GEANT4_VER=1072  g4--
   ## safer not to override 

   g4-;g4--   # this downloads and installs into S owned directories


NB to work in this split source manner need two sessions, as S does not have write permission into Opticks source

* O: to update opticks in the standard account 
* S: to build opticks in the non-standard account with non-standard externals 


Build Opticks against non-default Geant4 version
---------------------------------------------------

Now that a different Geant4 is installed, configure Opticks to use it in ~/.opticks_config::

     opticks-prepend-prefix $ext/g4_1072

::

    o
    om-
    om-clean    ## deletes all the build dirs 
    om-conf     ## visit the standard packages running CMake checking have all needed externals and generating Makefile
    om--        ## visit standard package build dirs, running make 



Darwin
--------

::

    -- Detecting CXX compile features - done
    -- Checking C++ feature CXXSTDLIB_FILESYSTEM_NATIVE - Failed
    -- Checking C++ feature CXXSTDLIB_FILESYSTEM_STDCXXFS - Failed
    -- Checking C++ feature CXXSTDLIB_FILESYSTEM_CXXFS - Failed
    CMake Error at cmake/Modules/G4BuildSettings.cmake:249 (message):
      No support for C++ filesystem found for compiler 'Clang', '10.0.1'
    Call Stack (most recent call first):
      cmake/Modules/G4CMakeMain.cmake:53 (include)
      CMakeLists.txt:48 (include)

Linux
--------

* https://geant4-data.web.cern.ch/ReleaseNotes/Beta4.11.0-1.txt

* Requiring C++17 as minimum standard to compile Geant4.

* Bumped minimum CMake version to 3.12.


N fails as gcc version not supported::

    CMake Error at cmake/Modules/G4BuildSettings.cmake:199 (message):
      Geant4 requested compilation using C++ standard '17' with compiler

      'GNU', version '4.8.5'

      but CMake 3.13.4 is not aware of any support for that standard by this
      compiler.  You may need a newer CMake and/or compiler.


S has devtoolset sourced to use newer gcc and gets further::

    -- Checking C++ feature CXXSTDLIB_FILESYSTEM_CXXFS - Failed
    CMake Error at cmake/Modules/G4OptionalComponents.cmake:64 (find_package):
      Could not find a configuration file for package "CLHEP" that is compatible
      with requested version "2.4.4.0".

      The following configuration files were considered but not accepted:

        /home/simon/local/opticks_externals/clhep/lib/CLHEP-2.4.1.0/CLHEPConfig.cmake, version: 2.4.1.0

    Call Stack (most recent call first):
      cmake/Modules/G4CMakeMain.cmake:59 (include)
      CMakeLists.txt:48 (include)


S, after clhep-- with 2440 proceed to g4-wipe g4-configure g4-build::

    Scanning dependencies of target G4global
    [  3%] Building CXX object source/CMakeFiles/G4global.dir/global/HEPNumerics/src/G4AnalyticalPolSolver.cc.o
    [  3%] Building CXX object source/CMakeFiles/G4global.dir/global/HEPNumerics/src/G4ChebyshevApproximation.cc.o
    In file included from /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/global/HEPNumerics/src/G4ChebyshevApproximation.cc:32:
    /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/global/management/include/G4PhysicalConstants.hh:69:14: error: ‘CLHEP::Bohr_magneton’ has not been declared
     using CLHEP::Bohr_magneton;
                  ^~~~~~~~~~~~~
    /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/global/management/include/G4PhysicalConstants.hh:70:14: error: ‘CLHEP::nuclear_magneton’ has not been declared
     using CLHEP::nuclear_magneton;
                  ^~~~~~~~~~~~~~~~
    gmake[2]: *** [source/CMakeFiles/G4global.dir/global/HEPNumerics/src/G4ChebyshevApproximation.cc.o] Error 1
    gmake[1]: *** [source/CMakeFiles/G4global.dir/all] Error 2
    gmake: *** [all] Error 2
    Wed Sep 22 03:57:37 CST 2021


Bump clhep to 2.4.5.1 and clhep--::

    epsilon:CLHEP blyth$ find . -type f -exec grep -H magneton {} \;
    ./ChangeLog:     Added constants: Bohr_magneton and nuclear_magneton.
    ./Units/ChangeLog:     Added constants: Bohr_magneton and nuclear_magneton.
    ./Units/Units/PhysicalConstants.h:// 06.05.21 Added Bohr_magneton and nuclear_magneton constants
    ./Units/Units/PhysicalConstants.h:static constexpr double Bohr_magneton = (eplus*hbarc*c_light)/(2*electron_mass_c2);
    ./Units/Units/PhysicalConstants.h:static constexpr double nuclear_magneton = (eplus*hbarc*c_light)/(2*proton_mass_c2);
    epsilon:CLHEP blyth$ 
    epsilon:CLHEP blyth$ 
    epsilon:CLHEP blyth$ pwd
    /usr/local/opticks_externals/clhep.build/2.4.5.1/CLHEP
    epsilon:CLHEP blyth$ 


After Geant4 build change the prefix setup in ~/.opticks_config to::

     30 ## hookup paths to access "foreign" externals 
     31 ext=/home/simon/local/opticks_externals
     32 opticks-prepend-prefix $ext/boost
     33 opticks-prepend-prefix $ext/clhep_2451
     34 opticks-prepend-prefix $ext/xercesc
     35 
     36 opticks-prepend-prefix $ext/g4_1100 
     37 
     38 # it is necessary to keep this override envvar set when using non-default version
     39 # if you want to use the g4- functions such as g4-cls 
     40 #export OPTICKS_GEANT4_VER=1062
     41 export OPTICKS_GEANT4_VER=1100
     42 
     43 opticks-setup > /dev/null
     44 [ $? -ne 0 ] && echo ERROR running opticks-setup && sleep 1000000000 

Then::

    o
    om-
    om-clean   # must clean in order to do full CMake reconfigure
    om-conf 


::

     9/31 Test  #9: ExtG4Test.X4MaterialTest ................................Child aborted***Exception:   0.14 sec
    STTF::GetFontPath dpath /home/simon/local/opticks/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf epath  

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat220
          issued by : G4MaterialPropertiesTable::AddProperty()
    Attempting to create a new material property key FASTCOMPONENT without setting
    createNewKey parameter of AddProperty to true.
    *** Fatal Exception ***


From 1100 adds createNewKey bool needed for some key::

     84     G4MaterialPropertyVector* AddProperty(const char     *key,
     85                                           G4double *PhotonEnergies,
     86                                           G4double *PropertyValues,
     87                                           G4int     NumEntries);


Try readonly sharing geocache, so "simon" reads from "blyth", that fixes many geocache version fails from x4 om-test::

    (base) [simon@localhost .opticks]$ mv geocache geocache_simon
    (base) [simon@localhost .opticks]$ ln -s /home/blyth/.opticks/geocache


* hmm suspect that geocache sharing like this is the cause of X4ScintillationTest to fail 
  as it sees a mismatch between interpolation from geocache creation with 1042 and a re-run 
  of interpolation with 1100 

  * TODO: find out how Geant4 ::GetEnergy interpolation has changed 




Breakpoints 
-------------

::

    (lldb) b G4Exception(char const*, char const*, G4ExceptionSeverity, char const*)
      ## break on exceptions


Introductions
--------------

* http://www.niser.ac.in/sercehep2017/notes/serc17_geant4.pdf

  111 slides of Geant4 intro 



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

g4-ver-default(){ echo 1042 ; }
g4-ver(){         echo ${OPTICKS_GEANT4_VER:-$(g4-ver-default)} ; }


g4-prefix-notes(){ cat << EON
g4-prefix-notes
=================

When using the Opticks g4- bash functions to build and install Geant4 it is required that there is no 
prior Geant4 installation in the CMAKE_PREFIX_PATH. If there is g4-- aborts via a non zero rc 
from g4-check-no-prior-prefix. In this building situation the g4-prefix-default is the g4-prefix. 

To build a new version of Geant4 for testing, modify ~/.opticks_config as shown 
below, start a new session, check g4-info and then g4-- to get and build it.
Changing g4 version in ~/.opticks_config::

    export OPTICKS_GEANT4_VER=1070
    #   uncomment above temporary setting whilst building a new geant4 
    #   make sure to comment the below standard g4_1042 prepend whilst building the new geant4 version
    #   for ease of testing multiple versions of geant4 with opticks a convenient approach is to
    #   create separate user accounts for each geant4 version and make them all use the same opticks
    #   code via a symbolic "opticks" link to /Users/blyth/opticks
    #   This is done on Dell Precision Workstation at IHEP using account "simon" for non-standard Geant4.

    ## hookup paths to access "foreign" externals 
    opticks-prepend-prefix /usr/local/opticks_externals/boost

    opticks-prepend-prefix /usr/local/opticks_externals/clhep
    opticks-prepend-prefix /usr/local/opticks_externals/xercesc

    #opticks-prepend-prefix /usr/local/opticks_externals/g4_1042
    # comment this whilst getting a building a new geant4 version


When using a pre-existing Geant4 installation in a build the CMAKE_PREFIX_PATH determines 
which Geant4 is used and g4-prefix-frompath will locate the prefix directory.  

Note that you do not need to use the Opticks g4- bash functions. Opticks can work 
with a Geant4 installed by other means into any location. All that is required is that 
the CMAKE_PREFIX_PATH is setup appropriately to find the install.  This can be 
done using the opticks-prepend-prefix bash function.

 g4-ver-default     : $(g4-ver-default) 
 g4-ver             : $(g4-ver) 
    default version can be overridden using OPTICKS_GEANT4_VER envvar, eg set it to 1062 
    The version is also incorporated into the basename of the g4-prefix

 g4-prefix          : $(g4-prefix)
 g4-prefix-default  : $(g4-prefix-default)
 g4-prefix-frompath : $(g4-prefix-frompath)    
    should be blank when building otherwise g4-- will abort via g4-check-no-prior-prefix 
 g4-prefix-old      : $(g4-prefix-old)         
    was formerly sensitive to OPTICKS_GEANT4_PREFIX envvar, but that was confusing 

OPTICKS_GEANT4_VER    : $OPTICKS_GEANT4_VER  
    this envvar overrides the default version of the Geant4 distribution to be downloaded 
    and installed

OPTICKS_GEANT4_PREFIX : $OPTICKS_GEANT4_PREFIX
    envvar set by $OPTICKS_PREFIX/bin/opticks-setup.sh
    based on first Geant4Config.cmake found in the CMAKE_PREFIX_PATH 
    (using opticks-setup-find-geant4-prefix)

    in order to assist with sourcing of Geant4 environment setup script.  This is used for the 
    runtime use of Geant4 in executables : it is no longer used to control install locations.  

    The om-/oe- machinery sources $OPTICKS_PREFIX/bin/opticks-setup.sh
    The opticks-setup.sh script is generated by opticks-full/opticks-setup-generate 
    this setup script encapsulates the externals that an Opticks build used. 
    Because of this it should not be edited, but rather the script needs to 
    be regenerated at Opticks build time.

EON
}

g4-1017-notes(){ cat << EON


::

    -- Detecting CXX compile features - done
    CMake Error at cmake/Modules/G4OptionalComponents.cmake:64 (find_package):
      Could not find a configuration file for package "CLHEP" that is compatible
      with requested version "2.4.4.0".

      The following configuration files were considered but not accepted:

        /usr/local/opticks_externals/clhep/lib/CLHEP-2.4.1.0/CLHEPConfig.cmake, version: 2.4.1.0




https://proj-clhep.web.cern.ch/proj-clhep/

 The latest releases are:

2.4.4.0, released November 9, 2020.
2.3.4.6, released February 15, 2018.


EON
}



g4-prefix-old(){      echo $(g4-prefix-default) ; }
g4-prefix-frompath(){ echo $(opticks-setup-find-geant4-prefix) ; }
g4-prefix-default(){  echo $(opticks-prefix)_externals/g4_$(g4-ver)  ; }
g4-prefix(){          echo ${OPTICKS_GEANT4_PREFIX:-$(g4-prefix-default)}  ; }

g4-dir(){   echo $(g4-prefix).build/$(g4-name) ; }  # exploded distribution dir

g4-nom(){ 
  case $(g4-ver) in 
     1021) echo Geant4-10.2.1 ;;
     1041) echo geant4_10_04_p01 ;; 
     1042) echo geant4_10_04_p02 ;;
     1062) echo geant4.10.06.p02 ;;
     1070) echo geant4.10.07 ;;
     1072) echo geant4.10.07.p02 ;;
    91072) echo geant4.10.07.r08 ;;    # unreleased dist for testing 
     1100) echo geant4.11.00.b01 ;;
  esac
}

g4-nom-notes(){ cat << EON

The nom identifier needs to match the name of the folder created by exploding the zip or tarball, 
unfortunately this is not simply connected with the basename of the url and also Geant4 continues to 
reposition URLs and names so these are liable to going stale.

EON
}

g4-url-notes(){ cat << EON

To get the url of a download with Safari start the download and then immediately 
pause it. Then use ctrl-click on the download item to "Copy Address" and paste 
the url into the below g4-url function, then delete the download. 

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
        geant4.10.07)     echo http://cern.ch/geant4-data/releases/geant4.10.07.tar.gz ;;
        geant4.10.07.p02) echo https://geant4-data.web.cern.ch/releases/geant4.10.07.p02.tar.gz ;;
        geant4.11.00.b01) echo https://geant4-data.web.cern.ch/releases/geant4.11.00.b01.tar.gz ;;
        geant4.10.07.r08) echo https://dummy/geant4.10.7.r08.tar ;;     # unreleased dummy 
   esac
}


g4-title()
{
   case $(g4-ver) in 
      1021) echo Geant4 10.2 first released 4 December 2015 \(patch-03, released 27 January 2017\) ;;
      1041) echo Geant4 10.4 patch-01, released 28 February 2018 ;; 
      1042) echo Geant4 10.4 patch-02, released 25 May 2018 ;; 
     91072) echo Dummy pre-release for 1100 beta version grabbed from OPTICKS_DOWNLOAD_CACHE ;; 
   esac
}

g4-version-hh() { echo $(g4-dir)/source/global/management/include/G4Version.hh ; }
g4-version-number() { perl -n -e 'm,#\s*define G4VERSION_NUMBER\s*(\d*), && print $1' $(g4-version-hh) ; } 

g4-version-notes(){
   local ver=${1:-$(g4-version-number)}
   echo $FUNCNAME $ver 
   g4-version-notes-$ver 2> /dev/null 
   [ $? -ne 0 ] && g4-version-notes-other $ver 
}

g4-version-notes-other(){  echo $1 : no known issues ; }

g4-version-notes-1060(){ g4-bug-2305 ; }
g4-version-notes-1061(){ g4-bug-2305 ; }
g4-version-notes-1062(){ g4-bug-2305 ; }
g4-version-notes-1063(){ g4-bug-2305 ; }
g4-version-notes-1070(){ g4-bug-2305 ; g4-bug-2311 ; }

g4-bug-2305(){ cat << EON

https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2305

Geant4 versions 1060,1061,1062,1063,1070 are known to have 
a severe GDML optical surface bug 2305 in 
source/persistency/gdml/src/G4GDMLReadSolids.cc

If your geometry has more than one optical surface and you 
use GDML then you are advised not to use these Geant4 versions with Opticks.

EON
}

g4-bug-2305-fix(){
  local msg="=== $FUNCNAME :"

  local cc=$(g4-dir)/source/persistency/gdml/src/G4GDMLReadSolids.cc 

  if [ -f "$cc.orig" ]; then 
     echo $msg it looks like a fix has been applied already : aborting 
     return 0  
  fi 

  local tmp=/tmp/$USER/opticks/$FUNCNAME/$(basename $cc) 
  mkdir -p $(dirname $tmp)

  cp $cc $tmp
  echo cc $cc
  echo tmp $tmp

  perl -pi -e "s,(\s*)(mapOfMatPropVects\[Strip\(name\)\] = propvect;),\$1//\$2 //$FUNCNAME," $tmp

  echo diff $cc $tmp
  diff $cc $tmp

  local ans
  read -p "Enter YES to copy the changed cc file into location $cc "  ans

  if [ "$ans" == "YES" ]; then 
     echo $msg proceeding 
     cp $cc $cc.orig
     cp $tmp $cc   
     echo diff $cc.orig $cc
     diff $cc.orig $cc 

     echo $msg following this change recompile libG4persistency with command : g4-build

  else
     echo $msg skip leaving cc untouched $cc  
  fi 

}


g4-bug-2311(){ cat << EON

https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2311

Geant4 1070 is known to feature an API change that means that 
the ordering of G4LogicalBorderSurface objects may not be reliable.
Although not an major issue in isolation this might cause confusion in 
comparisons between separate invokations or runs on different systems.

EON
}


# the below inline version approach is not recommended, 
# see above g4-prefix-notes for how to build a non-standard geant4 version for testing
#g4--1070(){ OPTICKS_GEANT4_VER=1070 g4-- ; }
#g4--1062(){ OPTICKS_GEANT4_VER=1062 g4-- ; }
#g4--1042(){ OPTICKS_GEANT4_VER=1042 g4-- ; }


g4-tag(){   echo g4 ; }
g4-dist(){ echo $(dirname $(g4-dir))/$(basename $(g4-url)) ; }

g4-filename(){  echo $(basename $(g4-url)) ; }
g4-name(){  local name=$(g4-filename) ; name=${name/.tar.gz} ; name=${name/.tar} ; name=${name/.zip} ; echo ${name} ; }  

g4-txt(){ vi $(g4-dir)/CMakeLists.txt ; }

g4-config(){ echo Debug ; }
#g4-config(){ echo RelWithDebInfo ; }

g4-bdir(){ echo $(g4-dir).$(g4-config).build ; }

g4-cmake-dir(){     echo $(g4-prefix)/lib$(g4-libsuffix)/$(g4-nom) ; }
g4-examples-dir(){  echo $(g4-prefix)/share/$(g4-nom)/examples ; }
g4-gdml-dir(){      echo $(g4-dir)/source/persistency/gdml ; }
g4-optical-dir(){   echo $(g4-dir)/source/processes/optical/src ; }

g4-c(){    cd $(g4-dir)/$1 ; }
g4-cd(){   cd $(g4-dir)/$1 ; }
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

   OPTICKS_DOWNLOAD_CACHE : $OPTICKS_DOWNLOAD_CACHE

   As opticks-curl is used distributions present within the 
   directory pointed to by OPTICKS_DOWNLOAD_CACHE envvar are 
   used ahead of downloads. 

EOI
}


g4-get(){
   local dir=$(dirname $(g4-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4-url)
   local dst=$(basename $url)
   local nam=$dst

   g4-get-info

   [ ! -f "$dst" ] && echo opticks-curl getting $url && opticks-curl $url

   if [ "${dst/.zip}" != "${dst}" ]; then
        nam=${nam/.zip}
        [ ! -d "$nam" ] && unzip $dst
   elif [ "${dst/.tar.gz}" != "${dst}" ]; then
        nam=${nam/.tar.gz}
        [ ! -d "$nam" ] && tar zxvf $dst
   elif [ "${dst/.tar}" != "${dst}" ]; then
        nam=${nam/.tar}
        [ ! -d "$nam" ] && tar xvf $dst
   fi

   [ -d "$nam" ]
}

g4--()
{
    local msg="=== $FUNCNAME :"
    g4-check-no-prior-prefix
    [ $? -ne 0 ] && echo $msg check-prior-prefix FAIL && return 1
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


g4-check-no-prior-prefix()
{
    local msg="=== $FUNCNAME :"
    local prior=$(opticks-setup-find-geant4-prefix)
    local rc 
    if [ "$prior" == "" ]; then 
        rc=0
    else
        echo $msg prior prefix found : $prior : remove geant4 prefix from CMAKE_PREFIX_PATH and or remove the prefix dir and try again 
        rc=1
    fi   
    return $rc
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
       -DGEANT4_INSTALL_DATA_TIMEOUT=100000  \\
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
       -DGEANT4_INSTALL_DATA_TIMEOUT=100000 \
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



g4n-cd(){   OPTICKS_GEANT4_VER=1062 g4-cd $* ; }
g4n-cls(){  OPTICKS_GEANT4_VER=1062 g4-cls $* ; }

g4-cls(){  
   local iwd=$PWD
   g4-cd 
   pwd
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


g4-ccd(){ 
   : cd to the directory containing the .cc of the cls argument 
   local name=${1:-G4GDMLWrite}
   g4-cd 
   pwd
   local cc=$(find source -name "$name.cc")
   local dir=$(dirname $cc)
   echo dir $dir
   cd $dir
   pwd
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

