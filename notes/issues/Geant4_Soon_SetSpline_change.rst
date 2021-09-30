Geant4_Soon_SetSpline_change
================================


Hi Simon,

Sorry to report another compilation error while building opticks with Geant4.10.7.ref08 
(actually using today's master as ref08 has not been tagged yet)  - again due to changes 
in public APIs in the G4PhysicsVector base class of Geant4 since the beta release - 
please see the error message included below for your information. 
I also see another place,  
extg4/OpNoviceDetectorConstruction.cc where G4PhysicsVector->SetSpline is used.

If FillSecondDerivatives is not used in opticks codes (apparently, looks like the case), 
I guess that guarding with the Geant4 version number around where SetSpline is used 
probably be  enough to resolve the issue. 

Thanks for your update (and sorry again to interrupting your work).

Regards,
---Soon


::

    [ 35%] Building CXX object CMakeFiles/ExtG4.dir/X4OpNoviceMaterials.cc.o
    /work1/g4gpu/syjun/products/src/opticks.v0.1.4-g4.11.beta/extg4/X4OpNoviceMaterials.cc: In constructor ‘X4OpNoviceMaterials::X4OpNoviceMaterials()’:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.4-g4.11.beta/extg4/X4OpNoviceMaterials.cc:118:11: error: ‘G4MaterialPropertyVector’ {aka ‘class G4PhysicsFreeVector’} has no member named ‘SetSpline’; did you mean ‘GetSpline’?
             ->SetSpline(true);
               ^~~~~~~~~
               GetSpline
    /work1/g4gpu/syjun/products/src/opticks.v0.1.4-g4.11.beta/extg4/X4OpNoviceMaterials.cc:120:11: error: ‘G4MaterialPropertyVector’ {aka ‘class G4PhysicsFreeVector’} has no member named ‘SetSpline’; did you mean ‘GetSpline’?
             ->SetSpline(true);
               ^~~~~~~~~
               GetSpline
    /work1/g4gpu/syjun/products/src/opticks.v0.1.4-g4.11.beta/extg4/X4OpNoviceMaterials.cc:133:11: error: ‘G4MaterialPropertyVector’ {aka ‘class G4PhysicsFreeVector’} has no member named ‘SetSpline’; did you mean ‘GetSpline’?
             ->SetSpline(true);
               ^~~~~~~~~
               GetSpline
    /work1/g4gpu/syjun/products/src/opticks.v0.1.4-g4.11.beta/extg4/X4OpNoviceMaterials.cc:135:11: error: ‘G4MaterialPropertyVector’ {aka ‘class G4PhysicsFreeVector’} has no member named ‘SetSpline’; did you mean ‘GetSpline’?
             ->SetSpline(true);
               ^~~~~~~~~
               GetSpline
    /work1/g4gpu/syjun/products/src/opticks.v0.1.4-g4.11.beta/extg4/X4OpNoviceMaterials.cc:194:11: error: ‘G4MaterialPropertyVector’ {aka ‘class G4PhysicsFreeVector’} has no member named ‘SetSpline’; did you mean ‘GetSpline’?
             ->SetSpline(true);
               ^~~~~~~~~
               GetSpline
    gmake[2]: *** [CMakeFiles/ExtG4.dir/X4OpNoviceMaterials.cc.o] Error 1
    gmake[1]: *** [CMakeFiles/ExtG4.dir/all] Error 2
    gmake: *** [all] Error 2
    === om-one-or-all install : non-zero rc 2
    === om-all om-install : ERROR bdir /work1/g4gpu/syjun/products/build/opticks.v0.1.4-local-g4.7.r08/opticks/build/extg4 : non-zero rc 2
    === om-one-or-all install : non-zero rc 2
    === opticks-full : ERR from opticks-full-make


::

    epsilon:opticks blyth$ opticks-fl SetSpline
    ./cfg4/CPropLib.cc                 ## eliminated SetSpline 
    ./cfg4/DetectorOld.cc              ## deleted this dead code
    ./cfg4/CMPT.cc                     ## removed 
    ./extg4/X4OpNoviceMaterials.cc
    ./extg4/OpNoviceDetectorConstruction.cc
    ./examples/Geant4/GDMLMangledLVNames/DetectorConstruction.cc
    ./examples/Geant4/CerenkovMinimal/src/DetectorConstruction.cc
    ./examples/Geant4/OpNovice/src/OpNoviceDetectorConstruction.cc
    ./examples/Geant4/CerenkovStandalone/L4CerenkovTest.cc
    ./examples/Geant4/CerenkovStandalone/OpticksUtil.cc
    epsilon:opticks blyth$ 

