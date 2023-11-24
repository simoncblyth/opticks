standalone_G4CXTest_gun_running_needs_more_U4Physics
=====================================================

::

    N[blyth@localhost opticks]$ ~/opticks/g4cx/tests/G4CXTest_GEOM.sh

    2023-11-24 10:46:18.812 INFO  [306965] [G4CXApp::BeamOn@322] [ OPTICKS_NUM_EVENT=3
    2023-11-24 10:47:50.347 INFO  [306965] [U4Recorder::BeginOfRunAction@253] 
    2023-11-24 10:47:50.347 INFO  [306965] [G4CXApp::GeneratePrimaries@222] [ fRunningMode 5
    2023-11-24 10:47:50.347 INFO  [306965] [G4CXApp::GeneratePrimaries@240] ]
    2023-11-24 10:47:50.347 INFO  [306965] [U4Recorder::BeginOfEventAction@288]  eventID 0
    ERROR: missing endpoint for the_energies_threshold? The size of the_energies_threshold is 1
    ERROR: missing endpoint for the_energies_threshold? The size of the_energies_threshold is 1

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : mat202
          issued by : G4MaterialPropertiesTable::GetConstProperty()
    Constant Material Property Index 8 not found.
    *** Fatal Exception *** core dump ***
    G4Track (0x222f33f0) - track ID = 1, parent ID = 0
     Particle type : e+ - creator process : not available
     Kinetic energy : 945.495 keV - Momentum direction : (0.974854,-0.190043,-0.116374)
     Step length : 346.005 um  - total energy deposit : 50.4957 keV
     Pre-step point : (0,0,0) - Physical volume : pTarget0x980a630 (LS)
     - defined by : not available
     Post-step point : (0.339839,-0.0434093,-0.0159184) - Physical volume : pTarget0x980a630 (LS)
     - defined by : eBrem - step status : 4
     *** Note: Step information might not be properly updated.

    -------- EEEE -------- G4Exception-END --------- EEEE -------



