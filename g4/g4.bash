# === func-gen- : g4/g4 fgp g4/g4.bash fgn g4 fgh g4
g4-src(){      echo g4/g4.bash ; }
g4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4-src)} ; }
g4-vi(){       vi $(g4-source) ; }
g4-env(){      elocal- ; }
g4-usage(){ cat << EOU

Geant4
========

Geant4 10.2, December 4th, 2015
----------------------------------

* https://geant4.web.cern.ch/geant4/support/ReleaseNotes4.10.2.html
* https://geant4.web.cern.ch/geant4/support/download.shtml
* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/

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

g4-name(){ echo geant4.10.02 ; } 
g4-name2(){ echo Geant4-10.2.0 ; }


g4-edir(){ echo $(env-home)/g4 ; }
g4-dir(){  echo $(local-base)/env/g4/$(g4-name) ; }
g4-idir(){ echo $(g4-dir).install ; }
g4-bdir(){ echo $(g4-dir).build ; }

g4-ifind(){ find $(g4-idir) -name ${1:-G4VUserActionInitialization.hh} ; }
g4-sfind(){ find $(g4-dir)/source -name ${1:-G4VUserActionInitialization.hh} ; }


g4-cmake-dir(){ echo $(g4-idir)/lib/$(g4-name2) ; }
g4-examples-dir(){  echo $(g4-idir)/share/$(g4-name2)/examples ; }


g4-ecd(){  cd $(g4-edir); }
g4-cd(){   cd $(g4-dir); }
g4-icd(){  cd $(g4-idir); }
g4-bcd(){  cd $(g4-bdir); }
g4-ccd(){  cd $(g4-cmake-dir); }
g4-xcd(){  cd $(g4-examples-dir); }


g4-url(){ echo http://geant4.cern.ch/support/source/$(g4-name).tar.gz ; }
g4-get(){
   local dir=$(dirname $(g4-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz 
}


g4-wipe(){
   local bdir=$(g4-bdir)
   rm -rf $bdir
}

g4-cmake(){
   local iwd=$PWD

   local bdir=$(g4-bdir)
   mkdir -p $bdir

   local idir=$(g4-idir)
   mkdir -p $idir

   g4-bcd
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DGEANT4_INSTALL_DATA=ON \
       -DCMAKE_INSTALL_PREFIX=$idir \
       $(g4-dir)

   cd $iwd
}

g4-make(){
   g4-bcd
   make -j4
}

g4-install(){
   g4-bcd
   make install
}


g4-export(){
   source $(g4-idir)/bin/geant4.sh
}



