# === func-gen- : g4/g4 fgp g4/g4.bash fgn g4 fgh g4
g4-src(){      echo g4/g4.bash ; }
g4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4-src)} ; }
g4-vi(){       vi $(g4-source) ; }
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
   elocal-  
   xercesc-  
}

g4-name(){ echo geant4.10.02 ; } 
g4-name2(){ echo Geant4-10.2.0 ; }


g4-edir(){ echo $(env-home)/g4 ; }
g4-dir(){  echo $(local-base)/env/g4/$(g4-name) ; }
g4-idir(){ echo $(g4-dir).install ; }
g4-bdir(){ echo $(g4-dir).build ; }

g4-ifind(){ find $(g4-idir) -name ${1:-G4VUserActionInitialization.hh} ; }
g4-sfind(){ find $(g4-dir)/source -name ${1:-G4VUserActionInitialization.hh} ; }

g4-hh(){ find $(g4-dir)/source -name '*.hh' -exec grep -H ${1:-G4GammaConversion} {} \; ; }
g4-icc(){ find $(g4-dir)/source -name '*.icc' -exec grep -H ${1:-G4GammaConversion} {} \; ; }
g4-cc(){ find $(g4-dir)/source -name '*.cc' -exec grep -H ${1:-G4GammaConversion} {} \; ; }

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

g4-cls(){
   local iwd=$PWD
   g4-cd
   local name=${1:-G4Scintillation}

   local hh=$(find source -name "$name.hh")
   local cc=$(find source -name "$name.cc")
   local icc=$(find source -name "$name.icc")

   local vcmd="vi -R $hh $icc $cc"
   echo $vcmd
   eval $vcmd

   cd $iwd
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
       -DGEANT4_USE_GDML=ON \
       -DXERCESC_ROOT_DIR=$(xercesc-prefix) \
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



