U4SimulateTest_with_large_gdml_geometry
===========================================


::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff55664b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff5582f080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff555c01ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff555881ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010626b3d5 libSysRap.dylib`qvals(vals=size=0, key="storch_FillGenstep_pos", fallback="0,0,-90", num_expect=3) at squad.h:457
        frame #5: 0x000000010626ad12 libSysRap.dylib`qvals(v=0x0000000117e1a130, key="storch_FillGenstep_pos", fallback="0,0,-90") at squad.h:510
        frame #6: 0x0000000106268f2a libSysRap.dylib`storch::FillGenstep(gs=0x0000000117e1a120, genstep_id=0, numphoton_per_genstep=10000) at storch.h:135
        frame #7: 0x0000000106268d3b libSysRap.dylib`void SEvent::FillGensteps<storch>(gs=0x0000000117e1a040, numphoton_per_genstep=10000) at SEvent.cc:69
        frame #8: 0x0000000106268c31 libSysRap.dylib`SEvent::MakeGensteps(gentype=6) at SEvent.cc:57
        frame #9: 0x0000000106268bae libSysRap.dylib`SEvent::MakeTorchGensteps() at SEvent.cc:45
        frame #10: 0x00000001062c396d libSysRap.dylib`SEvt::AddTorchGenstep() at SEvt.cc:725
        frame #11: 0x0000000100044765 U4SimulateTest`U4App::GeneratePrimaries(this=0x00000001076c8f00, event=0x0000000117e19c50) at U4App.h:193
        frame #12: 0x0000000100044d4c U4SimulateTest`non-virtual thunk to U4App::GeneratePrimaries(this=0x00000001076c8f00, event=0x0000000117e19c50) at U4App.h:0
        frame #13: 0x00000001020e1bc0 libG4run.dylib`G4RunManager::GenerateEvent(this=0x000000010769d620, i_event=0) at G4RunManager.cc:460
        frame #14: 0x00000001020e09c6 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010769d620, i_event=0) at G4RunManager.cc:398
        frame #15: 0x00000001020e0815 libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010769d620, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #16: 0x00000001020decd1 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010769d620, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #17: 0x000000010004573d U4SimulateTest`U4App::BeamOn(this=0x00000001076c8f00) at U4App.h:272
        frame #18: 0x0000000100045b42 U4SimulateTest`main(argc=1, argv=0x00007ffeefbfe640) at U4SimulateTest.cc:33
        frame #19: 0x00007fff55514015 libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x000000010626b3d5 libSysRap.dylib`qvals(vals=size=0, key="storch_FillGenstep_pos", fallback="0,0,-90", num_expect=3) at squad.h:457
       454 	        if( (*p >= '0' && *p <= '9') || *p == '+' || *p == '-' || *p == '.') vals.push_back(strtof(p, &p)) ; 
       455 	        else p++ ;
       456 	    }   
    -> 457 	    if( num_expect > 0 ) assert( vals.size() == unsigned(num_expect) ); 
       458 	}
       459 	
       460 	inline void qvals( std::vector<long>& vals, const char* key, const char* fallback, int num_expect )
    (lldb) p num_expect
    error: libU4.dylib debug map object file '/usr/local/opticks/build/u4/CMakeFiles/U4.dir/U4VolumeMaker.cc.o' has changed (actual time is 2023-04-11 14:30:49.000000000, debug map time is 2023-04-11 14:19:24.000000000) since this executable was linked, file will be ignored
    error: libU4.dylib debug map object file '/usr/local/opticks/build/u4/CMakeFiles/U4.dir/U4Recorder.cc.o' has changed (actual time is 2023-04-11 14:30:49.000000000, debug map time is 2023-04-11 14:15:06.000000000) since this executable was linked, file will be ignored
    (int) $0 = 3
    (lldb) p num_expect
    (int) $1 = 3
    (lldb) 


::

    (lldb) f 6
    frame #6: 0x0000000106268f2a libSysRap.dylib`storch::FillGenstep(gs=0x0000000117e1a120, genstep_id=0, numphoton_per_genstep=10000) at storch.h:135
       132 	    gs.gentype = OpticksGenstep_TORCH ; 
       133 	    gs.numphoton = numphoton_per_genstep  ;   
       134 	
    -> 135 	    qvals( gs.pos , storch_FillGenstep_pos , "0,0,-90" );    
       136 	    printf("//storch::FillGenstep storch_FillGenstep_pos gs.pos (%10.4f %10.4f %10.4f) \n", gs.pos.x, gs.pos.y, gs.pos.z ); 
       137 	
       138 	    qvals( gs.time, storch_FillGenstep_time, "0.0" ); 
    (lldb) 


This was due to blank strings when no config envvars being non-null 
but not handing over to fallback. 


