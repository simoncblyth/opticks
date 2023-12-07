G4CXTest_GEOM_QEvent_num_seed_zero_above_1M
==============================================


::

    121 #opticks_num_photon=M10
    122 #opticks_num_photon=M1,2,3
    123 opticks_num_photon=H1:10
    124 
    125 opticks_num_event=10
    126 opticks_max_photon=H10
    127 opticks_start_index=0
    128 opticks_max_bounce=31
    129 opticks_integration_mode=3
    130 
    131 #opticks_running_mode=SRM_DEFAULT
    132 opticks_running_mode=SRM_TORCH
    133 #opticks_running_mode=SRM_INPUT_PHOTON
    134 #opticks_running_mode=SRM_INPUT_GENSTEP    ## NOT IMPLEMENTED FOR GEANT4
    135 #opticks_running_mode=SRM_GUN
    136 



::

    2023-12-07 17:03:06.188 INFO  [189391] [SEvt::add_array@3530]  k U4R.npy a (1, )
    2023-12-07 17:03:06.531 INFO  [189391] [SEvt::save@3900] /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/G4CXTest/ALL3/B008 genstep,photon,hit
    2023-12-07 17:03:06.572 INFO  [189391] [U4Recorder::EndOfEventAction_@328]  savedir -
    2023-12-07 17:03:07.015 INFO  [189391] [QSim::simulate@368]  eventID 8 dt    0.428226 ph    1000000 ph/M          1
    2023-12-07 17:03:07.254 INFO  [189391] [SEvt::save@3900] /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/G4CXTest/ALL3/A008 genstep,photon,hit
    2023-12-07 17:03:07.390 INFO  [189391] [G4CXApp::GeneratePrimaries@223] [ SEventConfig::RunningModeLabel SRM_TORCH eventID 9
    U4VPrimaryGenerator::GeneratePrimaries ph (0, 4, 4, )
    2023-12-07 17:03:07.390 INFO  [189391] [G4CXApp::GeneratePrimaries@253] ]  eventID 9
    2023-12-07 17:03:07.390 INFO  [189391] [U4Recorder::BeginOfEventAction_@292]  eventID 9
    2023-12-07 17:03:07.390 INFO  [189391] [SEvt::add_array@3530]  k TRS.npy a -
    2023-12-07 17:03:07.390 INFO  [189391] [U4Recorder::MakeMetaArray@690] U4Recorder::DescFakes  
    U4Recorder::FAKES_SKIP NO 
    U4Recorder::FAKES      YES
    FAKES.size             0

    2023-12-07 17:03:07.390 INFO  [189391] [SEvt::add_array@3530]  k U4R.npy a (1, )
    2023-12-07 17:03:07.391 INFO  [189391] [SEvt::save@3900] /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/G4CXTest/ALL3/B009 genstep,photon,hit
    2023-12-07 17:03:07.392 INFO  [189391] [U4Recorder::EndOfEventAction_@328]  savedir -
    //QEvent_count_genstep_photons_and_fill_seed_buffer  evt.seed YES  evt.num_seed 0 
    G4CXTest: /home/blyth/junotop/opticks/qudarap/QEvent.cu:205: void QEvent_count_genstep_photons_and_fill_seed_buffer(sevent*): Assertion `expect_seed' failed.
    /home/blyth/o/G4CXTest_GEOM.sh: line 220: 189391 Aborted                 (core dumped) $bin
    /home/blyth/o/G4CXTest_GEOM.sh : run error
    N[blyth@localhost ~]$ 


