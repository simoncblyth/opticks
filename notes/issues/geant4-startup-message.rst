geant4-startup-message
========================

::

            ############################################
            !!! WARNING - FPE detection is activated !!!
            ############################################

    **************************************************************
     Geant4 version Name: geant4-10-04-patch-02    (25-May-2018)
                           Copyright : Geant4 Collaboration
                          References : NIM A 506 (2003), 250-303
                                     : IEEE-TNS 53 (2006), 270-278
                                     : NIM A 835 (2016), 186-225
                                 WWW : http://geant4.org/
    **************************************************************

    HepRandomEngine::put called -- no effect!



* TAB completion of symbolic bp requires all namespaces 

* documented the below in env-;gdb-

::

    (gdb) b 'HepRandom::put'
    Function "HepRandom::put" not defined.
    Make breakpoint pending on future shared library load? (y or [n]) n


    (gdb) b "CLHEP::HepRandom::put( <TAB>    DOES NOT WORK WITH DOUBLE QUOTES        

    (gdb) b 'CLHEP::HepRandom::put( <TAB>    THIS WORKS, BUT IT STOPS AT WRONG SYMBOL : SO USE SOURCE LINE BREAKPOINT

    (gdb) b 'CLHEP::HepRandom::put(std::ostream&) const' 



    (gdb) b /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/src/RandomEngine.cc:58
    Breakpoint 1 at 0x7fffe7b4d46e: file /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/src/RandomEngine.cc, line 58.



    Breakpoint 1, CLHEP::HepRandomEngine::put (this=0x6ad18a8, os=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/src/RandomEngine.cc:58
    58    std::cerr << "HepRandomEngine::put called -- no effect!\n";
    (gdb) bt
    #0  CLHEP::HepRandomEngine::put (this=0x6ad18a8, os=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/src/RandomEngine.cc:58
    #1  0x00007fffe7b4d63e in CLHEP::operator<< (os=..., e=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/src/RandomEngine.cc:99
    #2  0x00007fffe7b4c1ad in CLHEP::HepRandom::saveFullState (os=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/src/Random.cc:286
    #3  0x00007fffec6a8df8 in G4RunManager::G4RunManager (this=0x6ad2250) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:115
    #4  0x00007fffefd75d79 in CPhysics::CPhysics (this=0x614520, g4=0x6ad1ce0) at /home/blyth/opticks/cfg4/CPhysics.cc:20
    #5  0x00007fffefde738d in CG4::CG4 (this=0x6ad1ce0, hub=0x6b9590) at /home/blyth/opticks/cfg4/CG4.cc:124
    #6  0x00007ffff7bd5256 in OKG4Mgr::OKG4Mgr (this=0x7fffffffcc10, argc=35, argv=0x7fffffffcf48) at /home/blyth/opticks/okg4/OKG4Mgr.cc:76
    #7  0x000000000040399a in main (argc=35, argv=0x7fffffffcf48) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) 



    (gdb) f 7
    #7  0x000000000040399a in main (argc=35, argv=0x7fffffffcf48) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    8       OKG4Mgr okg4(argc, argv);
    (gdb) f 6
    #6  0x00007ffff7bd5256 in OKG4Mgr::OKG4Mgr (this=0x7fffffffcc10, argc=35, argv=0x7fffffffcf48) at /home/blyth/opticks/okg4/OKG4Mgr.cc:76
    76      m_g4(m_load ? NULL : new CG4(m_hub)),   // configure and initialize immediately 
    (gdb) f 5
    #5  0x00007fffefde738d in CG4::CG4 (this=0x6ad1ce0, hub=0x6b9590) at /home/blyth/opticks/cfg4/CG4.cc:124
    124     m_physics(new CPhysics(this)),
    (gdb) f 4
    #4  0x00007fffefd75d79 in CPhysics::CPhysics (this=0x614520, g4=0x6ad1ce0) at /home/blyth/opticks/cfg4/CPhysics.cc:20
    20      m_runManager(new G4RunManager),
    (gdb) f 3
    #3  0x00007fffec6a8df8 in G4RunManager::G4RunManager (this=0x6ad2250) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:115
    115   G4Random::saveFullState(oss);
    (gdb) l
    110   previousEvents = new std::list<G4Event*>;
    111   G4ParticleTable::GetParticleTable()->CreateMessenger();
    112   G4ProcessTable::GetProcessTable()->CreateMessenger();
    113   randomNumberStatusDir = "./";
    114   std::ostringstream oss;
    115   G4Random::saveFullState(oss);
    116   randomNumberStatusForThisRun = oss.str();
    117   randomNumberStatusForThisEvent = oss.str();
    118   runManagerType = sequentialRM;
    119 }
    (gdb) 


