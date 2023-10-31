recent_Geant4_removes_G4PhysicsOrderedFreeVector
==================================================


::

    [ 18%] Building CXX object CMakeFiles/U4.dir/Local_DsG4Scintillation.cc.o
    [ 19%] Building CXX object CMakeFiles/U4.dir/U4Physics.cc.o
    In file included from /home/simon/opticks/u4/Local_DsG4Scintillation.cc:76:
    /home/simon/opticks/u4/Local_DsG4Scintillation.hh:90:10: fatal error: G4PhysicsOrderedFreeVector.hh: No such file or directory
     #include "G4PhysicsOrderedFreeVector.hh"
              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    compilation terminated.
    make[2]: *** [CMakeFiles/U4.dir/Local_DsG4Scintillation.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs....
    In file included from /home/simon/opticks/u4/Local_G4Cerenkov_modified.cc:76:
    /home/simon/opticks/u4/Local_G4Cerenkov_modified.hh:70:10: fatal error: G4PhysicsOrderedFreeVector.hh: No such file or directory
     #include "G4PhysicsOrderedFreeVector.hh"
              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    compilation terminated.
    make[2]: *** [CMakeFiles/U4.dir/Local_G4Cerenkov_modified.cc.o] Error 1
    In file included from /home/simon/opticks/u4/U4Physics.cc:131:
    /home/simon/opticks/u4/Local_G4Cerenkov_modified.hh:70:10: fatal error: G4PhysicsOrderedFreeVector.hh: No such file or directory
     #include "G4PhysicsOrderedFreeVector.hh"
              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    compilation terminated.
    make[2]: *** [CMakeFiles/U4.dir/U4Physics.cc.o] Error 1
    make[1]: *** [CMakeFiles/U4.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all make : non-zero rc 2
    (base) [simon@localhost u4]$ 



