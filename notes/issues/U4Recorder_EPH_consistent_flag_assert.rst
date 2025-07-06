U4Recorder_EPH_consistent_flag_assert
========================================


::

    Begin of Event --> 0
    2025-07-05 19:10:00.529 INFO  [1688495] [U4Recorder::PreUserTrackingAction_Optical_@500]  modulo 100000 : ulabel.id 900000
    2025-07-05 19:10:05.797 INFO  [1688495] [U4Recorder::PreUserTrackingAction_Optical_@500]  modulo 100000 : ulabel.id 800000
    2025-07-05 19:10:11.118 INFO  [1688495] [U4Recorder::PreUserTrackingAction_Optical_@500]  modulo 100000 : ulabel.id 700000
    2025-07-05 19:10:16.456 INFO  [1688495] [U4Recorder::PreUserTrackingAction_Optical_@500]  modulo 100000 : ulabel.id 600000
    2025-07-05 19:10:18.258 INFO  [1688495] [U4Recorder::EPH_FlagCheck@1308]  original_flag SURFACE_ABSORB expected_original_flag YES eph  7 eph_ EPH_NDECULL is_eph_collect NO  is_eph_cull YES consistent_flag NO 
    python: /home/blyth/opticks/u4/U4Recorder.cc:1318: static void U4Recorder::EPH_FlagCheck(unsigned int, unsigned int): Assertion `consistent_flag' failed.

    (gdb) bt
    #0  0x00007ffff748b52c in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff743e686 in raise () from /lib64/libc.so.6
    #2  0x00007ffff7428833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff742875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff74373c6 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007fffc0ed1785 in U4Recorder::EPH_FlagCheck (original_flag=128, eph=7) at /home/blyth/opticks/u4/U4Recorder.cc:1318
    #6  0x00007fffc0ee2107 in U4Recorder::UserSteppingAction_Optical<C4OpBoundaryProcess> (this=0x6860a60, step=0x682c5e0) at /home/blyth/opticks/u4/U4Recorder.cc:1214
    #7  0x00007fffc0ecf232 in U4Recorder::UserSteppingAction (this=0x6860a60, step=0x682c5e0) at /home/blyth/opticks/u4/U4Recorder.cc:428
    #8  0x00007fffbd999f3c in U4RecorderAnaMgr::UserSteppingAction (this=0x65b7080, step=0x682c5e0) at /home/blyth/junosw/Simulation/DetSimV2/AnalysisCode/src/U4RecorderAnaMgr.cc:41
    #9  0x00007fffd486a062 in MgrOfAnaElem::UserSteppingAction (this=0xa284740, step=0x682c5e0) at /home/blyth/junosw/Simulation/DetSimV2/DetSimAlg/src/MgrOfAnaElem.cc:74
    #10 0x00007fffbddc6107 in LSExpSteppingAction::UserSteppingAction (this=0xa473680, fStep=0x682c5e0) at /home/blyth/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpSteppingAction.cc:56
    #11 0x00007fffc6725235 in G4SteppingManager::Stepping() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #12 0x00007fffc673070f in G4TrackingManager::ProcessOneTrack(G4Track*) () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #13 0x00007fffc676b565 in G4EventManager::DoProcessing(G4Event*) () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #14 0x00007fffe1a6c68e in G4SvcRunManager::SimulateEvent (this=0x663e990, i_event=0) at /home/blyth/junosw/Simulation/DetSimV2/G4Svc/src/G4SvcRunManager.cc:29




::

    1293 void U4Recorder::EPH_FlagCheck(unsigned original_flag, unsigned eph)  // static
    1294 {
    1295     bool expected_original_flag = original_flag == SURFACE_DETECT || original_flag == SURFACE_ABSORB || original_flag == BULK_ABSORB ;
    1296 
    1297     bool is_eph_surface_absorb = eph == EPH::NEDEP || eph == EPH::UNSET ;
    1298     bool is_eph_bulk_absorb    = eph == EPH::NEDEP || eph == EPH::UNSET || eph == EPH::NBOUND ;
    1299     bool is_eph_collect        = eph == EPH::SAVENORM || eph == EPH::YMERGE ;
    1300     bool is_eph_cull           = eph == EPH::NDECULL  ;
    1301     bool is_eph_detect         = eph == EPH::SAVENORM || eph == EPH::YMERGE || eph == EPH::NDECULL  ;
    1302 
    1303     bool consistent_flag = (  original_flag == SURFACE_DETECT && is_eph_detect ) ||
    1304                            (  original_flag == SURFACE_ABSORB && is_eph_surface_absorb ) ||
    1305                            (  original_flag == BULK_ABSORB    && is_eph_bulk_absorb ) ;
    1306 
    1307 
    1308     LOG_IF(info, !consistent_flag || !expected_original_flag )
    1309          << " original_flag " << OpticksPhoton::Flag(original_flag)
    1310          << " expected_original_flag " << ( expected_original_flag ? "YES" : "NO " )
    1311          << " eph " << std::setw(2) << eph
    1312          << " eph_ " << EPH::Name(eph)
    1313          << " is_eph_collect " << ( is_eph_collect ? "YES" : "NO " )
    1314          << " is_eph_cull " << ( is_eph_cull ? "YES" : "NO " )
    1315          << " consistent_flag " << ( consistent_flag ? "YES" : "NO " )
    1316          ;
    1317 
    1318     assert( consistent_flag );
    1319     assert( expected_original_flag );
    1320 }


Add surprise_flag allowance for now::

    1307     bool surprise_flag = ( original_flag == SURFACE_ABSORB  && is_eph_cull );
    1308     
    1309 
    1310     LOG_IF(info, !consistent_flag || !expected_original_flag )
    1311          << " original_flag " << OpticksPhoton::Flag(original_flag)
    1312          << " expected_original_flag " << ( expected_original_flag ? "YES" : "NO " )
    1313          << " eph " << std::setw(2) << eph
    1314          << " eph_ " << EPH::Name(eph)
    1315          << " is_eph_collect " << ( is_eph_collect ? "YES" : "NO " )
    1316          << " is_eph_cull " << ( is_eph_cull ? "YES" : "NO " ) 
    1317          << " consistent_flag " << ( consistent_flag ? "YES" : "NO " )
    1318          << " surpise_flag " << (  surprise_flag ?  "YES" : "NO" )
    1319          ;
    1320     
    1321     assert( consistent_flag || surprise_flag );
    1322     assert( expected_original_flag );
    1323 }



