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



