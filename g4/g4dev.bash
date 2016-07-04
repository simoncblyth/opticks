g4dev-src(){      echo g4/g4.bash ; }
g4dev-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(g4dev-src)} ; }
g4dev-vi(){       vi $(g4dev-source) ; }
g4dev-usage(){ cat << \EOU

Geant4
========

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

    simon:G01 blyth$ g4dev-cmake
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
g4dev-env(){      
   olocal-  
   xercesc-  
   opticks-
}


g4dev-edir(){ echo $(opticks-home)/g4 ; }

#g4dev-dir(){  echo $(local-base)/env/g4/$(g4dev-name) ; }
#g4dev-dir(){  echo $(opticks-prefix)/externals/g4/$(g4dev-name) ; }

#g4dev-prefix(){  echo $(opticks-prefix)/externals ; }

g4dev-prefix(){ 
    case $NODE_TAG in 
       MGB) echo $HOME/local/opticks/externals ;;
         D) echo /usr/local/opticks/externals ;;
         *) echo ${LOCAL_BASE:-/usr/local}/opticks/externals ;;
    esac
 }

g4dev-idir(){ echo $(g4dev-prefix) ; }
g4dev-dir(){   echo $(g4dev-prefix)/$(g4dev-tag)/$(g4dev-name) ; } 

# follow env/psm1/dist/dist.psm1 approach : everythinh based off the url


g4dev-tag(){   echo g4 ; }
g4dev-url(){   echo http://geant4.cern.ch/support/source/geant4_10_02_p01.zip ; }
g4dev-name2(){  echo Geant4-10.2.1 ; }

g4dev-filename(){  echo $(basename $(g4dev-url)) ; }
g4dev-name(){  local filename=$(g4dev-filename) ; echo ${filename%.*} ; }  
# hmm .tar.gz would still have a .tar on the name


g4dev-txt(){ vi $(g4dev-dir)/CMakeLists.txt ; }


g4dev-bdir(){ echo $(g4dev-dir).build ; }

g4dev-cmake-dir(){     echo $(g4dev-prefix)/lib/$(g4dev-name2) ; }
g4dev-examples-dir(){  echo $(g4dev-prefix)/share/$(g4dev-name2)/examples ; }


g4dev-ecd(){  cd $(g4dev-edir); }
g4dev-cd(){   cd $(g4dev-dir); }
g4dev-icd(){  cd $(g4dev-prefix); }
g4dev-bcd(){  cd $(g4dev-bdir); }
g4dev-ccd(){  cd $(g4dev-cmake-dir); }
g4dev-xcd(){  cd $(g4dev-examples-dir); }


g4dev-get-tgz(){
   local dir=$(dirname $(g4dev-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4dev-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz 
}

g4dev-get(){
   local dir=$(dirname $(g4dev-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4dev-url)
   local dst=$(basename $url)
   local nam=${dst/.zip}

   [ ! -f "$dst" ] && curl -L -O $url 
   [ ! -d "$nam" ] && unzip $dst 
}




g4dev-wipe(){
   local bdir=$(g4dev-bdir)
   rm -rf $bdir
}

g4dev-cmake-old(){
   local iwd=$PWD

   local bdir=$(g4dev-bdir)
   mkdir -p $bdir

   local idir=$(g4dev-prefix)
   mkdir -p $idir

   g4dev-bcd
   cmake \
       -G "$(opticks-cmake-generator)" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DGEANT4_INSTALL_DATA=ON \
       -DGEANT4_USE_GDML=ON \
       -DXERCESC_ROOT_DIR=$(xercesc-prefix) \
       -DCMAKE_INSTALL_PREFIX=$idir \
       $(g4dev-dir)

   cd $iwd
}



g4dev-cmake(){
   local iwd=$PWD

   local bdir=$(g4dev-bdir)
   mkdir -p $bdir

   local idir=$(g4dev-prefix)
   mkdir -p $idir

   g4dev-bcd
   cmake \
       -G "$(opticks-cmake-generator)" \
       -DGEANT4_INSTALL_DATA=ON \
       -DGEANT4_USE_GDML=ON \
       -DXERCESC_LIBRARY=$(xercesc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
       -DCMAKE_INSTALL_PREFIX=$idir \
       $(g4dev-dir)


   ## NB MSVC cmake build was configured using PowerShell invokation 
   #
   # env/psm1/g4/g4.psm1   
   # env/psm1/xercesc/xerces.psm1   
   # 

   cd $iwd
}



g4dev-configure()
{
   g4dev-wipe
   g4dev-cmake $*
}


#g4dev-config(){ echo Debug ; }
g4dev-config(){ echo RelWithDebInfo ; }
g4dev--(){
   g4dev-bcd
   cmake --build . --config $(g4dev-config) --target ${1:-install}
}


g4dev-sh(){  echo $(g4dev-idir)/bin/geant4.sh ; }
g4dev-ini(){ echo $(opticks-prefix)/externals/config/geant4.ini ; }

g4dev-export(){ source $(g4dev-sh) ; }
g4dev-export-ini()
{
    local msg="=== $FUNCNAME :"
    g4dev-export
    local ini=$(g4dev-ini)
    local dir=$(dirname $ini)
    mkdir -p $dir 
    echo $msg writing G4 environment to $ini
    env | grep G4 > $ini

    cat $ini

}




################# below funcions for styduing G4 source ##################################

g4dev-ifind(){ find $(g4dev-idir) -name ${1:-G4VUserActionInitialization.hh} ; }
g4dev-sfind(){ find $(g4dev-dir)/source -name ${1:-G4VUserActionInitialization.hh} ; }

g4dev-hh(){ find $(g4dev-dir)/source -name '*.hh' -exec grep -H ${1:-G4GammaConversion} {} \; ; }
g4dev-icc(){ find $(g4dev-dir)/source -name '*.icc' -exec grep -H ${1:-G4GammaConversion} {} \; ; }
g4dev-cc(){ find $(g4dev-dir)/source -name '*.cc' -exec grep -H ${1:-G4GammaConversion} {} \; ; }

g4dev-cls-copy(){
   local iwd=$PWD
   local name=${1:-G4Scintillation}
   local lname=${name/G4}

   local sauce=$(g4dev-dir)/source
   local hh=$(find $sauce -name "$name.hh")
   local cc=$(find $sauce -name "$name.cc")
   local icc=$(find $sauce -name "$name.icc")

   [ "$hh" != "" ]  && echo cp $hh $iwd/$lname.hh
   [ "$cc" != "" ] && echo cp $cc $iwd/$lname.cc
   [ "$icc" != "" ] && echo cp $icc $iwd/$lname.icc
}

g4dev-cls(){
   local iwd=$PWD
   g4dev-cd
   local name=${1:-G4Scintillation}

   local h=$(find source -name "$name.h")
   local hh=$(find source -name "$name.hh")
   local cc=$(find source -name "$name.cc")
   local icc=$(find source -name "$name.icc")

   local vcmd="vi -R $h $hh $icc $cc"
   echo $vcmd
   eval $vcmd

   cd $iwd
}

g4dev-look(){ 
   local iwd=$PWD
   g4dev-cd
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


