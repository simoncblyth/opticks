CerenkovMinimal_clhep_link_issue
====================================

Fixed issue by adding G4USE_STD11 macro in CMakeLists.txt of CerenkovMinimal


::

    [blyth@localhost opticks]$ ckm-
    [blyth@localhost opticks]$ ckm-c
    [blyth@localhost CerenkovMinimal]$ ./go.sh
    ...
    [100%] Linking CXX executable CerenkovMinimal
    CMakeFiles/CerenkovMinimal.dir/L4Cerenkov.cc.o: In function `L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)':
    /home/blyth/opticks/examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:365: undefined reference to `G4MTHepRandom::getTheEngine()'
    /home/blyth/opticks/examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:371: undefined reference to `G4MTHepRandom::getTheEngine()'
    /home/blyth/opticks/examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:379: undefined reference to `G4MTHepRandom::getTheEngine()'
    /home/blyth/opticks/examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:435: undefined reference to `G4MTHepRandom::getTheEngine()'
    /home/blyth/opticks/examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:438: undefined reference to `G4MTHepRandom::getTheEngine()'
    CMakeFiles/CerenkovMinimal.dir/L4Cerenkov.cc.o:/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/include/Geant4/G4Poisson.hh:59: more undefined references to `G4MTHepRandom::getTheEngine()' follow
    collect2: error: ld returned 1 exit status
    make[2]: *** [CerenkovMinimal] Error 1
    make[1]: *** [CMakeFiles/CerenkovMinimal.dir/all] Error 2
    make: *** [all] Error 2



Adding define "-DG4USE_STD11" switches to the non-MT branch in Randomize.hh and avoids undefined reference.

* /home/blyth/junotop/ExternalLibs/Build/geant4.10.04.p02/source/global/HEPRandom/include/Randomize.hh
* /home/blyth/junotop/ExternalLibs/CLHEP/2.4.1.0/include/CLHEP/Random/Randomize.h


examples/Geant4/CerenkovMinimal/CMakeLists.txt::

     43 add_executable(${name} ${name}.cc ${SOURCES} ${HEADERS})
     44 
     45 target_link_libraries(${name} Opticks::G4OK )
     46 target_compile_definitions( ${name} 
     47   PRIVATE 
     48       WITH_OPTICKS
     49       G4USE_STD11 
     50  )
     51 
     52 install(TARGETS ${name} DESTINATION lib)
     53 


::

    [blyth@localhost CerenkovMinimal]$ find /home/blyth/junotop/ExternalLibs/CLHEP/2.4.1.0/ -name Randomize.h
    /home/blyth/junotop/ExternalLibs/CLHEP/2.4.1.0/include/CLHEP/Random/Randomize.h

    [blyth@localhost geant4.10.04.p02]$ cd source
    [blyth@localhost source]$ find . -name G4MTHepRandom.hh
    ./global/HEPRandom/include/G4MTHepRandom.hh


    [blyth@localhost source]$ find . -type f -exec grep -H G4MTHepRandom.hh {} \;
    ./global/HEPRandom/src/G4MTHepRandom.cc:#include "G4MTHepRandom.hh"
    ./global/HEPRandom/sources.cmake:   G4MTHepRandom.hh 
    ./global/HEPRandom/include/G4MTRandGamma.hh:#include "G4MTHepRandom.hh"
    ./global/HEPRandom/include/G4MTRandFlat.hh:#include "G4MTHepRandom.hh"
    ./global/HEPRandom/include/G4MTRandExponential.hh:#include "G4MTHepRandom.hh"
    ./global/HEPRandom/include/G4MTRandGauss.hh:#include "G4MTHepRandom.hh"
    ./global/HEPRandom/include/Randomize.hh:#include "G4MTHepRandom.hh"
    ./global/HEPRandom/include/G4MTRandGeneral.hh:#include "G4MTHepRandom.hh"
    [blyth@localhost source]$ 


Need to consistently take the same branch here.

/home/blyth/junotop/ExternalLibs/Build/geant4.10.04.p02/source/global/HEPRandom/include/Randomize.hh::

     41 #if (defined(G4MULTITHREADED) && \
     42     (!defined(G4USE_STD11) || (defined(CLANG_NOSTDTLS) || defined(__INTEL_COMPILER))))
     43 
     44 // MT needs special Random Number distribution classes
     45 //
     46 #include "G4MTHepRandom.hh"
     47 #include "G4MTRandBit.hh"
     48 #include "G4MTRandExponential.hh"
     49 #include "G4MTRandFlat.hh"
     50 #include "G4MTRandGamma.hh"
     51 #include "G4MTRandGauss.hh"
     52 #include "G4MTRandGaussQ.hh"
     53 #include "G4MTRandGeneral.hh"
     54 
     55 // NOTE: G4RandStat MT-version is missing, but actually currently
     56 // never used in the G4 source
     57 //
     58 #define G4RandFlat G4MTRandFlat
     59 #define G4RandBit G4MTRandBit
     60 #define G4RandGamma G4MTRandGamma
     61 #define G4RandGauss G4MTRandGaussQ
     62 #define G4RandExponential G4MTRandExponential
     63 #define G4RandGeneral G4MTRandGeneral
     64 #define G4Random G4MTHepRandom
     65 
     66 #define G4UniformRand() G4MTHepRandom::getTheEngine()->flat()
     67 //
     68 //#include "G4UniformRandPool.hh"
     69 //#define G4UniformRand() G4UniformRandPool::flat()
     70 // Currently not be used in G4 source
     71 //
     72 #define G4RandFlatArray G4MTRandFlat::shootArray
     73 #define G4RandFlatInt G4MTRandFlat::shootInt
     74 #define G4RandGeneralTmp G4MTRandGeneral
     75 
     76 #else // Sequential mode or supporting C++11 standard
     77 
     78 // Distributions used ...
     79 //
     80 #include <CLHEP/Random/RandFlat.h>
     81 #include <CLHEP/Random/RandBit.h>
     82 #include <CLHEP/Random/RandGamma.h>



