PLOG_logging_from_external_libs always DEBUG, never suppressed
=================================================================

TODO: setup a test which uses FindOpticks.cmake and that reproduces 
the logging mis-behaviour to allow investigation 

* doing this in examples/UseFindOpticks/CMakeLists.txt



Logging mis-behaviour
------------------------

::

    epsilon:opticks blyth$ jcv junoSD_PMT_v2_Opticks
    2 files to edit
    ./Simulation/DetSimV2/PMTSim/include/junoSD_PMT_v2_Opticks.hh
    ./Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc

::

     44 
     45 #if (defined WITH_G4CXOPTICKS) || (defined WITH_G4OPTICKS)
     46 const plog::Severity junoSD_PMT_v2_Opticks::LEVEL = PLOG::EnvLevel("junoSD_PMT_v2_Opticks", "DEBUG") ;
     47 #endif
     48 


LOG(LEVEL) outputs from external libs always DEBUG, when would expect those would be suppressed.
The LOG(info) outputs appear as expected:: 

    dir /tmp/u4debug/ntds3/000 num_record 47
    2022-09-30 03:05:45.963 INFO  [178202] [U4Hit_Debug::Save@11]  dir /tmp/u4debug/ntds3/000 num_record 14
    dir /tmp/u4debug/ntds3/000 num_record 14
    [ junoSD_PMT_v2::EndOfEvent m_opticksMode  3
    2022-09-30 03:05:45.963 DEBUG [178202] [junoSD_PMT_v2_Opticks::EndOfEvent@169] [ eventID 0 m_opticksMode 3
    2022-09-30 03:05:45.995 INFO  [178202] [junoSD_PMT_v2_Opticks::EndOfEvent@190]  eventID 0 num_hit 27 way_enabled 0
         0 gp.x  -13840.08 gp.y   -8162.24 gp.z  -10659.09 gp.R   19281.76 pmt   13743          CK|RE|SD|BT
         1 gp.x  -13331.45 gp.y   -7860.98 gp.z  -11652.90 gp.R   19372.99 pmt   14076          CK|RE|SD|BT
         2 gp.x   -7827.26 gp.y  -16841.33 gp.z    5141.73 gp.R   19270.02 pmt    6269          CK|RE|SD|BT





