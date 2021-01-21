g4_1070_G4LogicalBorderSurface_vector_to_map_problems
=========================================================

create new user "francis" for testing g4_1070
------------------------------------------------

* System Preferences > Users & Groups
* add new user name and set passwould 
* login using the GUI, click thru the dialogs skipping things like appleid and Siri 

* make basic GUI customizations:

  * Trackpad > Tap to click 
  * Accessibility > Mouse & Trackpad > Trackpad Options : Enable Dragging (without drag lock)
  * Dock > Autohide

* GUI logout 

* back to main blyth account, add username to .ssh/config
* attempt to ssh in fails::

    epsilon:notes blyth$ ssh F
    Password:
    Connection closed by 127.0.0.1 port 22
    epsilon:notes blyth$  

* in Sharing > Remote Login > add the new user to the list of permitted 

* check can ssh in now, and place the ssh key for passwordless ssh from blyth::

  ssh--putkey F

* minimal setup for using opticks::

    epsilon:~ francis$ ln -s /Users/blyth/opticks
    epsilon:~ francis$ cp ~charles/.bash_profile . 
    epsilon:~ francis$ cp ~charles/.bashrc . 
    epsilon:~ francis$ cp ~charles/.opticks_config . 

    epsilon:~ francis$ cp ~blyth/.vimrc .

* try sharing the rngcache::

    epsilon:~ francis$ mkdir .opticks
    epsilon:~ francis$ cd .opticks
    epsilon:.opticks francis$ ln -s /Users/blyth/.opticks/rngcache
    epsilon:.opticks francis$ 


G4LogicalBorderSurfaceTable has changed from std::vector to std::map
-----------------------------------------------------------------------


::

    === om-make-one : extg4           /Users/francis/opticks/extg4                                 /Users/francis/local/opticks/build/extg4                     
    Scanning dependencies of target ExtG4
    [  2%] Building CXX object CMakeFiles/ExtG4.dir/X4Gen.cc.o
    [  2%] Building CXX object CMakeFiles/ExtG4.dir/X4CSG.cc.o
    [  3%] Building CXX object CMakeFiles/ExtG4.dir/X4.cc.o
    /Users/francis/opticks/extg4/X4.cc:323:25: error: no matching function for call to 'GetItemIndex'
        int idx_lbs = lbs ? GetItemIndex<G4LogicalBorderSurface>( lbs_table, lbs ) : -1 ;    
                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /Users/francis/opticks/extg4/X4.cc:111:9: note: candidate function not viable: no known conversion from 'const G4LogicalBorderSurfaceTable *' (aka 'const map<std::pair<const G4VPhysicalVolume *, const
          G4VPhysicalVolume *>, G4LogicalBorderSurface *> *') to 'const std::vector<G4LogicalBorderSurface *> *' for 1st argument
    int X4::GetItemIndex( const std::vector<T*>* vec, const T* const item )
            ^
    1 error generated.
    make[2]: *** [CMakeFiles/ExtG4.dir/X4.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs....


Idea for how to keep hold of creation order
------------------------------------------------

It would be simple to assign a creation order index to the G4LogicalBorderSurface::

     44 G4LogicalBorderSurface::
     45 G4LogicalBorderSurface(const G4String& name,
     46                              G4VPhysicalVolume* vol1,
     47                              G4VPhysicalVolume* vol2,
     48                              G4SurfaceProperty* surfaceProperty)
     49   : G4LogicalSurface(name, surfaceProperty),
     50     Volume1(vol1), Volume2(vol2), 
            Index( theBorderSurfaceTable ? theBorderSurfaceTable->size() : 0 )  // Assign creation order index to the border surface 
     51 {
     52   if (theBorderSurfaceTable == nullptr)
     53   {
     54     theBorderSurfaceTable = new G4LogicalBorderSurfaceTable;
     55   }
     56 
     57   // Store in the table of Surfaces
     58   //
     59   theBorderSurfaceTable->insert(std::make_pair(std::make_pair(vol1,vol2),this));
     60 }
     61 


With the creation order index can explicitly control the ordering despite using a std::map with std::pair of pointers key::

    size_t GetIndex() const ; 


    inline
    size_t G4LogicalBorderSurface::GetIndex() const 
    {
      return Index;
    }
       

1070 has "PTL/Globals.hh" include dir issue 
----------------------------------------------     

::

    === om-make-one : extg4           /Users/francis/opticks/extg4                                 /Users/francis/local/opticks/build/extg4                     
    Scanning dependencies of target ExtG4
    [  2%] Building CXX object CMakeFiles/ExtG4.dir/X4LogicalBorderSurfaceTable.cc.o
    [  2%] Building CXX object CMakeFiles/ExtG4.dir/X4PhysicalVolume.cc.o
    In file included from /Users/francis/opticks/extg4/X4PhysicalVolume.cc:33:
    In file included from /usr/local/opticks_externals/g4_1070/include/Geant4/G4VSensitiveDetector.hh:33:
    In file included from /usr/local/opticks_externals/g4_1070/include/Geant4/G4Step.hh:56:
    In file included from /usr/local/opticks_externals/g4_1070/include/Geant4/G4Profiler.hh:43:
    /usr/local/opticks_externals/g4_1070/include/Geant4/G4Profiler.icc:44:12: fatal error: 'PTL/Globals.hh' file not found
    #  include "PTL/Globals.hh"
               ^~~~~~~~~~~~~~~~
    [  3%] Building CXX object CMakeFiles/ExtG4.dir/OpNoviceDetectorConstruction.cc.o
    1 error generated.
    make[2]: *** [CMakeFiles/ExtG4.dir/X4PhysicalVolume.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs....
    make[1]: *** [CMakeFiles/ExtG4.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all make : non-zero rc 2


* https://geant4-forum.web.cern.ch/t/10-7-geant4-config-problem-with-includedirs/4032


::

    epsilon:Geant4 blyth$ pwd
    /usr/local/opticks_externals/g4_1070/include/Geant4
    epsilon:Geant4 blyth$ ln -s ../PTL


1070 no longer has G4OpticalProcessIndex.hh
-----------------------------------------------

::

    === om-make-one : cfg4            /Users/francis/opticks/cfg4                                  /Users/francis/local/opticks/build/cfg4                      
    Scanning dependencies of target CFG4
    [  1%] Building CXX object CMakeFiles/CFG4.dir/PhysicsList.cc.o
    [  1%] Building CXX object CMakeFiles/CFG4.dir/CFG4_LOG.cc.o
    /Users/francis/opticks/cfg4/PhysicsList.cc:26:10: fatal error: 'G4OpticalProcessIndex.hh' file not found
    #include "G4OpticalProcessIndex.hh"
             ^~~~~~~~~~~~~~~~~~~~~~~~~~
    1 error generated.
    make[2]: *** [CMakeFiles/CFG4.dir/PhysicsList.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs....
    [  2%] Building CXX object CMakeFiles/CFG4.dir/Scintillation.cc.o
    make[1]: *** [CMakeFiles/CFG4.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all make : non-zero rc 2
    epsilon:cfg4 francis$ 


1042::

     46 #ifndef G4OpticalProcessIndex_h
     47 #define G4OpticalProcessIndex_h 1
     48 
     49 #include "globals.hh"
     50 
     51 enum G4OpticalProcessIndex {
     52   kCerenkov,      ///< Cerenkov process index
     53   kScintillation, ///< Scintillation process index
     54   kAbsorption,    ///< Absorption process index
     55   kRayleigh,      ///< Rayleigh scattering process index
     56   kMieHG,         ///< Mie scattering process index
     57   kBoundary,      ///< Boundary process index
     58   kWLS,           ///< Wave Length Shifting process index
     59   kNoProcess      ///< Number of processes, no selected process
     60 };
     61 
     62 /// Return the name for a given optical process index
     63 G4String G4OpticalProcessName(G4int );
     64 
     65 ////////////////////
     66 // Inline methods
     67 ////////////////////
     68 
     69 inline
     70 G4String G4OpticalProcessName(G4int processNumber)
     71 {
     72   switch ( processNumber ) {
     73     case kCerenkov:      return "Cerenkov";
     74     case kScintillation: return "Scintillation";
     75     case kAbsorption:    return "OpAbsorption";
     76     case kRayleigh:      return "OpRayleigh";
     77     case kMieHG:         return "OpMieHG";
     78     case kBoundary:      return "OpBoundary";
     79     case kWLS:           return "OpWLS";
     80     default:             return "NoProcess";
     81   }
     82 }
     83 
     84 #endif // G4OpticalProcessIndex_h


::

    epsilon:docs blyth$ g4-cc G4OpticalProcessIndex.hh
    epsilon:docs blyth$ g4-hh G4OpticalProcessIndex.hh
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/physics_lists/constructors/electromagnetic/include/G4OpticalPhysics.hh:#include "G4OpticalProcessIndex.hh"
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/physics_lists/constructors/electromagnetic/include/G4OpticalPhysicsMessenger.hh:#include "G4OpticalProcessIndex.hh"
    epsilon:docs blyth$ 

    097 void G4OpticalPhysics::PrintStatistics() const
     98 {
     99 // Print all processes activation and their parameters
    100 
    101   for ( G4int i=0; i<kNoProcess; i++ ) {
    102     G4cout << "  " << G4OpticalProcessName(i) << " process:  ";
    103     if ( ! fProcessUse[i] ) {
    104       G4cout << "not used" << G4endl;
    105     }




geocache-create fails
--------------------------

Presumably thats the famous G4 bug 

* :doc:`g4-1062-geocache-create-reflectivity-assert`

::

    ...
    2021-01-16 23:16:47.182 INFO  [27114104] [X4PhysicalVolume::convertMaterials@263]  num_materials 36 num_material_with_efficiency 1
    2021-01-16 23:16:47.183 INFO  [27114104] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 1
     0 :                       Bialkali
    Assertion failed: (_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "), function createStandardSurface, file /Users/francis/opticks/ggeo/GSurfaceLib.cc, line 597.
    /Users/francis/local/opticks/bin/o.sh: line 362: 63773 Abort trap: 6           /Users/francis/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /Users/francis/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml --x4polyskip 211,232 --geocenter --noviz --runfolder geocache-dx1 --runcomment sensors-gdml-review.rst
    === o-main : runline PWD /tmp/francis/opticks/geocache-create- RC 134 Sat Jan 16 23:16:47 GMT 2021
    /Users/francis/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /Users/francis/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml --x4polyskip 211,232 --geocenter --noviz --runfolder geocache-dx1 --runcomment sensors-gdml-review.rst
    echo o-postline : dummy
    o-postline : dummy
    /Users/francis/local/opticks/bin/o.sh : RC : 134
    epsilon:~ francis$ 




.opticks_config setup for easier geant4 switching
----------------------------------------------------

::

     22 ## hookup paths to access "foreign" externals 
     23 opticks-prepend-prefix /usr/local/opticks_externals/boost
     24 opticks-prepend-prefix /usr/local/opticks_externals/xercesc
     25 
     26 # leave only one of the below clhep+geant4 setup "stanzas" uncommented 
     27 # to pick the geant4 version and start a new session before doing anything 
     28 # like using the g4- functions or building opticks against this geant4 
     29 
     30 # standard 1042 
     31 #opticks-prepend-prefix /usr/local/opticks_externals/clhep
     32 #opticks-prepend-prefix /usr/local/opticks_externals/g4_1042
     33 
     34 # non-standard 1070
     35 export OPTICKS_GEANT4_VER=1070
     36 opticks-prepend-prefix /usr/local/opticks_externals/clhep_2440
     37 opticks-prepend-prefix /usr/local/opticks_externals/g4_1070
     38 
     39 # For convenient use of multiple geant4 versions with the same opticks
     40 # source create diffrent user accounts for each 
     41 #
     42 #   blyth   : 1042
     43 #   charles : 1062   has manual 2305 fix effectively eliminating mapOfMatPropVects + private/public fix for G4Cerenkov  
     44 #   francis : 1070
     45 #
     46 # where /Users/charles/opticks and /Users/francis/opticks are 
     47 # symbolic links to /Users/blyth/opticks




::

     660 g4-bug-2305-fix(){
     661   local msg="=== $FUNCNAME :"
     662 
     663   local cc=$(g4-dir)/source/persistency/gdml/src/G4GDMLReadSolids.cc
     664 
     665   if [ -f "$cc.orig" ]; then
     666      echo $msg it looks like a fix has been applied already : aborting 
     667      return 0
     668   fi
     669 
     670   local tmp=/tmp/$USER/opticks/$FUNCNAME/$(basename $cc)
     671   mkdir -p $(dirname $tmp)
     672 
     673   cp $cc $tmp
     674   echo cc $cc
     675   echo tmp $tmp
     676 
     677   perl -pi -e "s,(\s*)(mapOfMatPropVects\[Strip\(name\)\] = propvect;),\$1//\$2 //$FUNCNAME," $tmp
     678 
     679   echo diff $cc $tmp
     680   diff $cc $tmp
     681 
     682   local ans
     683   read -p "Enter YES to copy the changed cc file into location $cc "  ans
     684 
     685   if [ "$ans" == "YES" ]; then
     686      echo $msg proceeding 
     687      cp $cc $cc.orig
     688      cp $tmp $cc
     689      echo diff $cc.orig $cc
     690      diff $cc.orig $cc
     691   else
     692      echo $msg skip leaving cc untouched $cc  
     693   fi
     694 
     695 }




geocache-create::

    2021-01-17 15:17:14.074 INFO  [27361602] [GGeo::postDirectTranslationDump@590] GGeo::postDirectTranslationDump NOT --dumpsensor numSensorVolumes 672
    2021-01-17 15:17:14.078 ERROR [27361602] [OpticksHub::configure@435] FORCED COMPUTE MODE : as remote session detected 
    2021-01-17 15:17:14.079 INFO  [27361602] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2021-01-17 15:17:14.080 FATAL [27361602] [*Opticks::makeSimpleTorchStep@3584]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg 
    2021-01-17 15:17:14.082 FATAL [27361602] [OpticksResource::getDefaultFrame@207]  PLACEHOLDER ZERO 
    2021-01-17 15:17:14.082 INFO  [27361602] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname (null) nidxs.size() 0 nidx -1
    2021-01-17 15:17:14.082 ERROR [27361602] [*OpticksGen::makeTorchstep@441]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 cmdline_target [--gensteptarget] : 0 gdmlaux_target : -1 active_target : 0
    2021-01-17 15:17:14.082 ERROR [27361602] [*OpticksGen::makeTorchstep@468]  generateoverride 0 num_photons0 10000 num_photons 10000
    libc++abi.dylib: terminating with uncaught exception of type APIError
    /Users/francis/local/opticks/bin/o.sh: line 362: 80343 Abort trap: 6           /Users/francis/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /Users/francis/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml --x4polyskip 211,232 --geocenter --noviz --runfolder geocache-dx1 --runcomment sensors-gdml-review.rst
    === o-main : runline PWD /tmp/francis/opticks/geocache-create- RC 134 Sun Jan 17 15:17:14 GMT 2021
    /Users/francis/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /Users/francis/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml --x4polyskip 211,232 --geocenter --noviz --runfolder geocache-dx1 --runcomment sensors-gdml-review.rst
    echo o-postline : dummy
    o-postline : dummy
    /Users/francis/local/opticks/bin/o.sh : RC : 134
    epsilon:~ francis$ 



Huh SIGABRT from VisibleDevices, which is wierd as that is not related to Geant4.

geocache-create -D::

     tot  node :   12230 vert : 1289446 face : 2533452
    2021-01-17 15:18:53.982 INFO  [27367799] [GGeo::postDirectTranslationDump@590] GGeo::postDirectTranslationDump NOT --dumpsensor numSensorVolumes 672
    2021-01-17 15:18:53.983 ERROR [27367799] [OpticksHub::configure@435] FORCED COMPUTE MODE : as remote session detected 
    2021-01-17 15:18:53.984 INFO  [27367799] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname /dd/Geometry/AD/lvADE0xc2a78c00x3ef9140 nidxs.size() 2 nidx 3153
    2021-01-17 15:18:53.985 FATAL [27367799] [Opticks::makeSimpleTorchStep@3584]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg 
    2021-01-17 15:18:53.985 FATAL [27367799] [OpticksResource::getDefaultFrame@207]  PLACEHOLDER ZERO 
    2021-01-17 15:18:53.985 INFO  [27367799] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271]  target_lvname /dd/Geometry/AD/lvADE0xc2a78c00x3ef9140 nidxs.size() 2 nidx 3153
    2021-01-17 15:18:53.985 ERROR [27367799] [OpticksGen::makeTorchstep@441]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0 cmdline_target [--gensteptarget] : 0 gdmlaux_target : 3153 active_target : 3153
    2021-01-17 15:18:53.985 ERROR [27367799] [OpticksGen::makeTorchstep@468]  generateoverride 0 num_photons0 10000 num_photons 10000
    libc++abi.dylib: terminating with uncaught exception of type APIError
    Process 80481 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
    ...
    Process 80481 launched: '/Users/francis/local/opticks/lib/OKX4Test' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #8: 0x00000001007ad6e2 libOptiXRap.dylib`VisibleDevices::VisibleDevices(this=0x00007ffeefbfdb68) at OContext.cc:163
        frame #9: 0x00000001007966c5 libOptiXRap.dylib`VisibleDevices::VisibleDevices(this=0x00007ffeefbfdb68) at OContext.cc:162
        frame #10: 0x0000000100795d68 libOptiXRap.dylib`OContext::CheckDevices(ok=0x000000010f4c98b0) at OContext.cc:195
        frame #11: 0x00000001007977fb libOptiXRap.dylib`OContext::Create(ok=0x000000010f4c98b0, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OContext.cc:238
        frame #12: 0x00000001007b96bd libOptiXRap.dylib`OScene::OScene(this=0x0000000129ee9cd0, hub=0x000000012993bac0, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OScene.cc:85
        frame #13: 0x00000001007ba94d libOptiXRap.dylib`OScene::OScene(this=0x0000000129ee9cd0, hub=0x000000012993bac0, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OScene.cc:96
        frame #14: 0x00000001006cc416 libOKOP.dylib`OpEngine::OpEngine(this=0x00000001299408b0, hub=0x000000012993bac0) at OpEngine.cc:74
        frame #15: 0x00000001006ccced libOKOP.dylib`OpEngine::OpEngine(this=0x00000001299408b0, hub=0x000000012993bac0) at OpEngine.cc:83
        frame #16: 0x000000010022a94f libOK.dylib`OKPropagator::OKPropagator(this=0x0000000129940860, hub=0x000000012993bac0, idx=0x000000012986ce10, viz=0x0000000000000000) at OKPropagator.cc:68
        frame #17: 0x000000010022aafd libOK.dylib`OKPropagator::OKPropagator(this=0x0000000129940860, hub=0x000000012993bac0, idx=0x000000012986ce10, viz=0x0000000000000000) at OKPropagator.cc:72
        frame #18: 0x000000010020d22c libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe428, argc=15, argv=0x00007ffeefbfec80, argforced=0x0000000000000000) at OKMgr.cc:63
        frame #19: 0x000000010020d69b libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe428, argc=15, argv=0x00007ffeefbfec80, argforced=0x0000000000000000) at OKMgr.cc:65
        frame #20: 0x000000010001586f OKX4Test`main(argc=15, argv=0x00007ffeefbfec80) at OKX4Test.cc:126
        frame #21: 0x00007fff77c24015 libdyld.dylib`start + 1
    (lldb) 




The SIGABRT looks to happen at the first attempt to access the GPU::

     155 struct VisibleDevices
     156 {
     157     unsigned num_devices;
     158     unsigned version;
     159     std::vector<Device> devices ;
     160 
     161     VisibleDevices()
     162     {
     163         RT_CHECK_ERROR(rtDeviceGetDeviceCount(&num_devices));
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   
     164         RT_CHECK_ERROR(rtGetVersion(&version));
     165         for(unsigned i = 0; i < num_devices; ++i)
     166         {
     167             Device d(i);
     168             devices.push_back(d);
     169         }
     170     }



Was testing from blyth GUI account in a ssh session into francis account.
Possibly there is a GPU access permissions problem for user francis as that user does not 
have a GUI session running ? 

Yep, confirmed this. After starting GUI session for francis the geocache-create completes OK.


After setting OPTICKS_KEY opticks-t gives 3 fails
----------------------------------------------------


::

    SLOW: tests taking longer that 15 seconds
      8  /36  Test #8  : CFG4Test.CG4Test                              Passed                         16.65  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         23.34  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 Passed                         15.67  


    FAILS:  3   / 438   :  Sun Jan 17 15:35:15 2021   
      1  /1   Test #1  : OKConfTest.OKConfTest                         Child aborted***Exception:     0.03   
      1  /50  Test #1  : SysRapTest.SOKConfTest                        Child aborted***Exception:     0.03   
      6  /36  Test #6  : CFG4Test.CGDMLPropertyTest                    Child aborted***Exception:     0.13   
    epsilon:~ francis$ 


::

    epsilon:~ francis$ OKConfTest
    OKConf::Dump
                       OKConf::OpticksInstallPrefix() /Users/francis/local/opticks
                            OKConf::CMAKE_CXX_FLAGS()  -fvisibility=hidden -fvisibility-inlines-hidden -fdiagnostics-show-option -Wall -Wno-unused-function -Wno-unused-private-field -Wno-shadow
                         OKConf::CUDAVersionInteger() 9010
                   OKConf::ComputeCapabilityInteger() 30
                            OKConf::OptiXInstallDir() /usr/local/optix
                         OKCONF_OPTIX_VERSION_INTEGER 50001
                        OKConf::OptiXVersionInteger() 50001
                         OKCONF_OPTIX_VERSION_MAJOR   5
                          OKConf::OptiXVersionMajor() 5
                         OKCONF_OPTIX_VERSION_MINOR   0
                          OKConf::OptiXVersionMinor() 0
                         OKCONF_OPTIX_VERSION_MICRO   1
                          OKConf::OptiXVersionMicro() 1
                       OKConf::Geant4VersionInteger() 0
                       OKConf::ShaderDir()            /Users/francis/local/opticks/gl

     OKConf::Check() 1
    Assertion failed: (rc == 0), function main, file /Users/francis/opticks/okconf/tests/OKConfTest.cc, line 32.
    Abort trap: 6
    epsilon:~ francis$ 



okconf failing to peek at the Geant4 version with 1070 : FIXED
-----------------------------------------------------------------

::

    epsilon:okconf francis$ om-conf
    === om-one-or-all conf : okconf          /Users/francis/opticks/okconf                                /Users/francis/local/opticks/build/okconf                    
    -- Configuring OKConf
    ,,,
    -- OKCONF_OPTIX_INSTALL_DIR : 
    -- OptiX_VERSION_INTEGER : 50001
    -- OpticksCUDA_API_VERSION : 9010
    -- G4_VERSION_INTEGER      : 
    -- Configuring OKConfTest


Fixed two of the fails by generalizing the cmake/Modules/FindG4.cmake  pattern match::


    .    foreach(_line ${_contents})
    -        if (_line MATCHES "#define G4VERSION_NUMBER[ ]+([0-9]+)$")
    +        if (_line MATCHES "#[ ]*define[ ]+G4VERSION_NUMBER[ ]+([0-9]+)$")
                 set(G4_VERSION_INTEGER ${CMAKE_MATCH_1})
             endif()
         endforeach()



CGDMLPropertyTest : looks like permissions issue due to inadvertent shared path /tmp/v1.gdml
-----------------------------------------------------------------------------------------------

::

    epsilon:sysrap francis$ CGDMLPropertyTest 
    2021-01-17 16:54:04.659 INFO  [27420776] [main@146] OKConf::Geant4VersionInteger() : 1070
    2021-01-17 16:54:04.660 INFO  [27420776] [main@153]  parsing /tmp/v1.gdml
    G4GDML: Reading '/tmp/v1.gdml'...

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : InvalidRead
          issued by : G4GDMLRead::Read()
    Unable to open document: /tmp/v1.gdml
    *** Fatal Exception ***
    -------- EEEE ------- G4Exception-END -------- EEEE -------


    *** G4Exception: Aborting execution ***
    Abort trap: 6
    epsilon:sysrap francis$ 




francis 1070 down to 0/438 fails
-----------------------------------

::

    SLOW: tests taking longer that 15 seconds
      8  /36  Test #8  : CFG4Test.CG4Test                              Passed                         15.77  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         23.63  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 Passed                         15.67  


    FAILS:  0   / 438   :  Sun Jan 17 17:13:36 2021   



What about GDML writing of border surfaces, how is the order controlled ?
---------------------------------------------------------------------------

::

    epsilon:src francis$ pwd
    /Users/francis/local/opticks_externals/g4_1070.build/geant4.10.07/source/persistency/gdml/src
    epsilon:src francis$ vi G4GDMLWriteStructure.cc

    467 void G4GDMLWriteStructure::SurfacesWrite()
    468 {
    469 #ifdef G4VERBOSE
    470   G4cout << "G4GDML: Writing surfaces..." << G4endl;
    471 #endif
    472   for(auto pos = skinElementVec.cbegin();
    473            pos != skinElementVec.cend(); ++pos)
    474   {
    475     structureElement->appendChild(*pos);
    476   }
    477   for(auto pos = borderElementVec.cbegin();
    478            pos != borderElementVec.cend(); ++pos)
    479   {
    480     structureElement->appendChild(*pos);
    481   }
    482 }

    322 void G4GDMLWriteStructure::BorderSurfaceCache(
    323   const G4LogicalBorderSurface* const bsurf)
    324 { 
    325   if(bsurf == nullptr)
    326   { 
    327     return;
    328   }
    329   
    330   const G4SurfaceProperty* psurf = bsurf->GetSurfaceProperty();
    331   
    332   // Generate the new element for border-surface
    333   //
    334   const G4String& bsname             = GenerateName(bsurf->GetName(), bsurf);
    335   const G4String& psname             = GenerateName(psurf->GetName(), psurf);
    336   xercesc::DOMElement* borderElement = NewElement("bordersurface");
    337   borderElement->setAttributeNode(NewAttribute("name", bsname));
    338   borderElement->setAttributeNode(NewAttribute("surfaceproperty", psname));
    339   
    340   const G4String volumeref1 =
    341     GenerateName(bsurf->GetVolume1()->GetName(), bsurf->GetVolume1());
    342   const G4String volumeref2 =
    343     GenerateName(bsurf->GetVolume2()->GetName(), bsurf->GetVolume2());
    344   xercesc::DOMElement* volumerefElement1 = NewElement("physvolref");
    345   xercesc::DOMElement* volumerefElement2 = NewElement("physvolref");
    346   volumerefElement1->setAttributeNode(NewAttribute("ref", volumeref1));
    347   volumerefElement2->setAttributeNode(NewAttribute("ref", volumeref2));
    348   borderElement->appendChild(volumerefElement1);
    349   borderElement->appendChild(volumerefElement2);
    350   
    351   if(FindOpticalSurface(psurf))
    352   { 
    353     const G4OpticalSurface* opsurf =
    354       dynamic_cast<const G4OpticalSurface*>(psurf);
    355     if(opsurf == nullptr)
    356     { 
    357       G4Exception("G4GDMLWriteStructure::BorderSurfaceCache()", "InvalidSetup",
    358                   FatalException, "No optical surface found!");
    359       return;
    360     }
    361     OpticalSurfaceWrite(solidsElement, opsurf);
    362   }
    363   
    364   borderElementVec.push_back(borderElement);
    365 }


