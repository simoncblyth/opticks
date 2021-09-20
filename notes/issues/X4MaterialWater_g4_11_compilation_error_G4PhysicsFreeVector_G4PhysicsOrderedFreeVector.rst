X4MaterialWater_g4_11_compilation_error_G4PhysicsFreeVector_G4PhysicsOrderedFreeVector
=========================================================================================


::

    Hi Hans,

    It looks that there are still remaining API issues between opticks (v0.1.3, the latest tag) 
    and Geant4 11 (beta for this testing): see compilation errors below, for an example.
    Since the CaTS example will be integrated with Geant4 11, I guess that we
    need to protect these errors with the G4VERSION guard.

    Regards,
    ---Soon

    [ 16%] Building CXX object CMakeFiles/ExtG4.dir/X4MaterialWater.cc.o
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc: In static member function ‘static G4PhysicsOrderedFreeVector* X4MaterialWater::GetProperty(G4int)’:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc:47:67: error: cannot convert ‘G4MaterialPropertyVector*’ {aka ‘G4PhysicsFreeVector*’} to ‘G4PhysicsOrderedFreeVector*’ in initialization
         G4PhysicsOrderedFreeVector* PROP = WaterMPT->GetProperty(index) ;
                                                                       ^
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc: In constructor ‘X4MaterialWater::X4MaterialWater()’:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc:55:99: error: invalid static_cast from type ‘G4MaterialPropertyVector*’ {aka ‘G4PhysicsFreeVector*’} to type ‘G4PhysicsOrderedFreeVector*’
         rayleigh0(WaterMPT ? static_cast<G4PhysicsOrderedFreeVector*>(WaterMPT->GetProperty(kRAYLEIGH)) : nullptr ),
                                                                                                       ^
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc: In member function ‘void X4MaterialWater::init()’:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc:78:90: error: invalid static_cast from type ‘G4PhysicsOrderedFreeVector*’ to type ‘G4MaterialPropertyVector*’ {aka ‘G4PhysicsFreeVector*’}
             WaterMPT->AddProperty("RAYLEIGH", static_cast<G4MaterialPropertyVector*>(rayleigh) );
                                                                                              ^
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc: In member function ‘G4double X4MaterialWater::GetMeanFreePath(G4double) const’:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc:95:31: error: invalid use of incomplete type ‘class G4PhysicsOrderedFreeVector’
         return rayleigh ? rayleigh->Value( photonMomentum ) : DBL_MAX ;
                                   ^~
    In file included from /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc:11:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4OpRayleigh.hh:9:7: note: forward declaration of ‘class G4PhysicsOrderedFreeVector’
     class G4PhysicsOrderedFreeVector ;
           ^~~~~~~~~~~~~~~~~~~~~~~~~~
    gmake[2]: *** [CMakeFiles/ExtG4.dir/X4MaterialWater.cc.o] Error 1
    gmake[1]: *** [CMakeFiles/ExtG4.dir/all] Error 2
    gmake: *** [all] Error 2
    === om-one-or-all install : non-zero rc 2
    === om-all om-install : ERROR bdir /work1/g4gpu/syjun/products/build/opticks.v0.1.3-local-g4.11.beta/opticks/build/extg4 : non-zero rc 2
    === om-one-or-all install : non-zero rc 2
    === opticks-full : ERR from opticks-full-make



::

     41 G4PhysicsOrderedFreeVector* X4MaterialWater::GetProperty(const G4int index)
     42 {
     43     G4Material* Water = G4Material::GetMaterial("Water");
     44     if(Water == nullptr) return nullptr ;
     45     G4MaterialPropertiesTable* WaterMPT = Water->GetMaterialPropertiesTable() ;
     46     if(WaterMPT == nullptr) return nullptr ;
     47     G4PhysicsOrderedFreeVector* PROP = WaterMPT->GetProperty(index) ;
     48     return PROP ;
     49 }


::

    epsilon:issues blyth$ g4-cls G4MaterialPropertiesTable
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02
    vi -R source/materials/include/G4MaterialPropertiesTable.hh source/materials/include/G4MaterialPropertiesTable.icc source/materials/src/G4MaterialPropertiesTable.cc
    3 files to edit

    109 
    110     G4MaterialPropertyVector* GetProperty(const char *key,
    111                                           G4bool warning=false);
    112     // Get the property from the table corresponding to the key-name.
    113 
    114     G4MaterialPropertyVector* GetProperty(const G4int index,
    115                                           G4bool warning=false);
    116     // Get the property from the table corresponding to the key-index.
    117 



    epsilon:issues blyth$ g4-cls G4MaterialPropertyVector
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02
    vi -R source/materials/include/G4MaterialPropertyVector.hh
    epsilon:issues blyth$ 


    56 #include "G4PhysicsOrderedFreeVector.hh"
    57 
    58 /////////////////////
    59 // Class Definition
    60 /////////////////////
    61 
    62 typedef G4PhysicsOrderedFreeVector G4MaterialPropertyVector;
    63 
    64 #endif /* G4MaterialPropertyVector_h */



1062
-----

::

    (base) [simon@localhost extg4]$ grep ^typedef /home/simon/local/opticks_externals/g4_1062.build/geant4.10.06.p02/source/materials/include/G4MaterialPropertyVector.hh
    typedef G4PhysicsOrderedFreeVector G4MaterialPropertyVector;




Huh : 1100 has lost the "Ordered"
------------------------------------


* https://github.com/Geant4/geant4/blob/master/source/materials/include/G4MaterialPropertyVector.hh
* https://github.com/Geant4/geant4


::

    typedef G4PhysicsFreeVector G4MaterialPropertyVector;




Soon Followup
----------------


* https://github.com/Geant4/geant4/blob/master/source/global/management/include/G4PhysicsTable.hh





::

    Hi Simon,

    Thank you for a quick follow up.  Included below is a new error message
    with the commit hash of opticks and geant4.11.beta.

    Regards,
    ---Soon


    [ 16%] Building CXX object CMakeFiles/ExtG4.dir/X4MaterialWater.cc.o
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc: In constructor ‘X4MaterialWater::X4MaterialWater()’:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc:71:74: error: conditional expression between distinct pointer types ‘G4MaterialPropertyVector*’ {aka ‘G4PhysicsFreeVector*’} and ‘G4PhysicsOrderedFreeVector*’ lacks a cast
         rayleigh(rayleigh0 ? rayleigh0 : X4OpRayleigh::WaterScatteringLength() )
                                                                              ^
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc: In member function ‘void X4MaterialWater::rayleigh_scan2() const’:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc:145:50: error: no matching function for call to ‘X4PhysicsOrderedFreeVector::X4PhysicsOrderedFreeVector(G4MaterialPropertyVector* const&)’
         X4PhysicsOrderedFreeVector rayleighx(rayleigh);
                                                      ^
    In file included from /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc:12:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:41:5: note: candidate: ‘X4PhysicsOrderedFreeVector::X4PhysicsOrderedFreeVector(G4PhysicsOrderedFreeVector*)’
         X4PhysicsOrderedFreeVector( G4PhysicsOrderedFreeVector* vec_ ) ;
         ^~~~~~~~~~~~~~~~~~~~~~~~~~
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:41:5: note:   no known conversion for argument 1 from ‘G4MaterialPropertyVector* const’ {aka ‘G4PhysicsFreeVector* const’} to ‘G4PhysicsOrderedFreeVector*’
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:22:15: note: candidate: ‘constexpr X4PhysicsOrderedFreeVector::X4PhysicsOrderedFreeVector(const X4PhysicsOrderedFreeVector&)’
     struct X4_API X4PhysicsOrderedFreeVector
                   ^~~~~~~~~~~~~~~~~~~~~~~~~~
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:22:15: note:   no known conversion for argument 1 from ‘G4MaterialPropertyVector* const’ {aka ‘G4PhysicsFreeVector* const’} to ‘const X4PhysicsOrderedFreeVector&’
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:22:15: note: candidate: ‘constexpr X4PhysicsOrderedFreeVector::X4PhysicsOrderedFreeVector(X4PhysicsOrderedFreeVector&&)’
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:22:15: note:   no known conversion for argument 1 from ‘G4MaterialPropertyVector* const’ {aka ‘G4PhysicsFreeVector* const’} to ‘X4PhysicsOrderedFreeVector&&’
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc: In member function ‘void X4MaterialWater::changeRayleighToMidBin()’:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc:150:50: error: no matching function for call to ‘X4PhysicsOrderedFreeVector::X4PhysicsOrderedFreeVector(G4MaterialPropertyVector*&)’
         X4PhysicsOrderedFreeVector rayleighx(rayleigh);
                                                      ^
    In file included from /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4MaterialWater.cc:12:
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:41:5: note: candidate: ‘X4PhysicsOrderedFreeVector::X4PhysicsOrderedFreeVector(G4PhysicsOrderedFreeVector*)’
         X4PhysicsOrderedFreeVector( G4PhysicsOrderedFreeVector* vec_ ) ;
         ^~~~~~~~~~~~~~~~~~~~~~~~~~
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:41:5: note:   no known conversion for argument 1 from ‘G4MaterialPropertyVector*’ {aka ‘G4PhysicsFreeVector*’} to ‘G4PhysicsOrderedFreeVector*’
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:22:15: note: candidate: ‘constexpr X4PhysicsOrderedFreeVector::X4PhysicsOrderedFreeVector(const X4PhysicsOrderedFreeVector&)’
     struct X4_API X4PhysicsOrderedFreeVector
                   ^~~~~~~~~~~~~~~~~~~~~~~~~~
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:22:15: note:   no known conversion for argument 1 from ‘G4MaterialPropertyVector*’ {aka ‘G4PhysicsFreeVector*’} to ‘const X4PhysicsOrderedFreeVector&’
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:22:15: note: candidate: ‘constexpr X4PhysicsOrderedFreeVector::X4PhysicsOrderedFreeVector(X4PhysicsOrderedFreeVector&&)’
    /work1/g4gpu/syjun/products/src/opticks.v0.1.3-g4.11.beta/extg4/X4PhysicsOrderedFreeVector.hh:22:15: note:   no known conversion for argument 1 from ‘G4MaterialPropertyVector*’ {aka ‘G4PhysicsFreeVector*’} to ‘X4PhysicsOrderedFreeVector&&’
    gmake[2]: *** [CMakeFiles/ExtG4.dir/X4MaterialWater.cc.o] Error 1
    gmake[1]: *** [CMakeFiles/ExtG4.dir/all] Error 2
    gmake: *** [all] Error 2
    === om-one-or-all install : non-zero rc 2
    === om-all om-install : ERROR bdir /work1/g4gpu/syjun/products/build/opticks.v0.1.3-local-g4.11.beta/opticks/build/extg4 : non-zero rc 2
    === om-one-or-all install : non-zero rc 2
    === opticks-full : ERR from opticks-full-make




::

    Hi Simon,

    I find that there are still many other files where
    G4PhysicsOrderedFreeVector is used:

    extg4/X4Scintillation.cc
    extg4/X4Scintillation.hh

    extg4/tests/X4ArrayTest.cc
    extg4/tests/X4PhysicsVectorTest.cc
    extg4/tests/X4ScintillationTest.cc

    and also many files under

    cfg4
    cfg4/tests

    It looks that replacing G4PhysicsOrderedFreeVector by G4MaterialPropertyVector
    will resolve all errors.   It would be great if you can update all codes accordingly
    and tag it with v0.1.4.  Thanks.

    Regards,
    ---Soon


::

    epsilon:opticks blyth$ opticks-fl G4PhysicsOrderedFreeVector
    ./cfg4/CVec.hh
    ./cfg4/Scintillation.hh
    ./cfg4/C4Cerenkov1042.cc
    ./cfg4/C4Cerenkov1042.hh
    ./cfg4/DsG4OpRayleigh.cc
    ./cfg4/DsG4Cerenkov.cc
    ./cfg4/tests/CVecTest.cc
    ./cfg4/tests/CMakeLists.txt
    ./cfg4/tests/G4PhysicsOrderedFreeVectorTest.cc
    ./cfg4/tests/WaterTest.cc
    ./cfg4/Scintillation.cc
    ./cfg4/DsG4Cerenkov.h
    ./cfg4/CVec.cc
    ./cfg4/DsG4Scintillation.cc
    ./cfg4/Cerenkov.hh
    ./cfg4/DsG4Scintillation.h
    ./cfg4/OpRayleigh.hh
    ./cfg4/CMPT.hh
    ./cfg4/G4Cerenkov1042.cc
    ./cfg4/CMPT.cc
    ./cfg4/OpRayleigh.cc
    ./cfg4/G4Cerenkov1042.hh
    ./cfg4/Cerenkov.cc
    ./cfg4/DsG4OpRayleigh.h
    ./extg4/X4MaterialWater.cc
    ./extg4/X4Scintillation.hh
    ./extg4/X4Scintillation.cc
    ./extg4/tests/X4ScintillationTest.cc
    ./extg4/tests/X4PhysicsVectorTest.cc
    ./extg4/tests/X4ArrayTest.cc
    ./extg4/X4MaterialWater.hh
    ./extg4/X4OpRayleigh.cc
    ./extg4/X4Array.hh
    ./extg4/X4PhysicsVector.cc
    ./notes/geant4/opnovice.bash
    ./examples/Geant4/CerenkovMinimal/src/CKMScintillation.h
    ./examples/Geant4/CerenkovMinimal/src/CKMScintillation.cc
    ./examples/Geant4/CerenkovMinimal/src/L4Cerenkov.cc
    ./examples/Geant4/CerenkovMinimal/src/Cerenkov.cc
    ./examples/Geant4/CerenkovMinimal/src/L4Cerenkov.hh
    ./examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.hh
    ./examples/Geant4/CerenkovStandalone/L4CerenkovTest.cc
    ./examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.cc
    epsilon:opticks blyth$ 





Kludge 1042 to preview problems with 1100.beta
---------------------------------------------------


::

    epsilon:Geant4 blyth$ cp G4MaterialPropertyVector.hh G4MaterialPropertyVector.hh.orig
    epsilon:Geant4 blyth$ vi G4MaterialPropertyVector.hh
    epsilon:Geant4 blyth$ vi G4MaterialPropertyVector.hh
    epsilon:Geant4 blyth$ 
    epsilon:Geant4 blyth$ pwd
    /usr/local/opticks_externals/g4_1042/include/Geant4
    epsilon:Geant4 blyth$ 


::

    epsilon:Geant4 blyth$ diff G4MaterialPropertyVector.hh.orig G4MaterialPropertyVector.hh
    56c56,57
    < #include "G4PhysicsOrderedFreeVector.hh"
    ---
    > //#include "G4PhysicsOrderedFreeVector.hh"
    > #include "G4PhysicsFreeVector.hh"
    62c63,64
    < typedef G4PhysicsOrderedFreeVector G4MaterialPropertyVector;
    ---
    > //typedef G4PhysicsOrderedFreeVector G4MaterialPropertyVector;
    > typedef G4PhysicsFreeVector G4MaterialPropertyVector;
    epsilon:Geant4 blyth$ 



What are these Geant4 guys smoking ?
--------------------------------------

* https://github.com/Geant4/geant4/blob/master/source/global/management/include/G4PhysicsFreeVector.hh

::

     // - 04 Feb. 2021, V.Ivanchenko moved implementation of all free vectors 
     //                 to this class



     explicit G4PhysicsFreeVector(const G4double* energies, const G4double* values,
                                   std::size_t length, G4bool spline = false);
      // The vector is filled in this constructor;
      // 'energies' and 'values' need to have the same vector length;
      // 'energies' assumed to be ordered in the user code.




