FIXED : SEvt_save_creating_empty_NPFold_dirs_when_config_nothing_to_save
===========================================================================


::

    LOG=1 BP=mkdir ~/o/qudarap/tests/QEvent_Lifecycle_Test.sh

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x00007fff57e06d78 libsystem_kernel.dylib`mkdir
        frame #1: 0x000000010066eae9 libSysRap.dylib`sdirectory::MakeDirs(dirpath_="/data/blyth/opticks/GEOM/TEST/QEvent_Lifecycle_Test/ALL0/A000", mode_=0) at sdirectory.h:60
        frame #2: 0x0000000100837f72 libSysRap.dylib`SEvt::getOutputDir(this=0x000000010181ac00, base_="$DefaultOutputDir") const at SEvt.cc:3760
        frame #3: 0x000000010081b5b5 libSysRap.dylib`SEvt::save(this=0x000000010181ac00, dir_="$DefaultOutputDir") at SEvt.cc:3948
        frame #4: 0x000000010081acba libSysRap.dylib`SEvt::save(this=0x000000010181ac00) at SEvt.cc:3658
        frame #5: 0x000000010081d849 libSysRap.dylib`SEvt::endOfEvent(this=0x000000010181ac00, eventID=0) at SEvt.cc:1602
        frame #6: 0x00000001000472df QEvent_Lifecycle_Test`QEvent_Lifecycle_Test::Test() at QEvent_Lifecycle_Test.cc:63
        frame #7: 0x0000000100047533 QEvent_Lifecycle_Test`main(argc=1, argv=0x00007ffeefbfe850) at QEvent_Lifecycle_Test.cc:86
        frame #8: 0x00007fff57cb5015 libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 3
    frame #3: 0x000000010081b5b5 libSysRap.dylib`SEvt::save(this=0x000000010181ac00, dir_="$DefaultOutputDir") at SEvt.cc:3948
       3945	        // NPFold::add does nothing with nullptr array 
       3946	    }
       3947	
    -> 3948	    const char* dir = getOutputDir(dir_); 
       3949	    LOG(info) << dir << " [" << save_comp << "]"  ; 
       3950	    LOG(LEVEL) << descSaveDir(dir_) ; 
       3951	
    (lldb) p dir_
    (const char *) $1 = 0x000000010098628a "$DefaultOutputDir"
    (lldb) f 2
    frame #2: 0x0000000100837f72 libSysRap.dylib`SEvt::getOutputDir(this=0x000000010181ac00, base_="$DefaultOutputDir") const at SEvt.cc:3760
       3757	    const char* reldir = GetReldir() ; 
       3758	    const char* sidx = hasIndex() ? getIndexString(nullptr) : nullptr ; 
       3759	    const char* path = sidx ? spath::Resolve(base,reldir,sidx ) : spath::Resolve(base, reldir) ; 
    -> 3760	    sdirectory::MakeDirs(path,0); 
       3761	
       3762	    LOG(LEVEL)
       3763	        << std::endl  
    (lldb) p path 
    (const char *) $2 = 0x0000000102483db0 "/data/blyth/opticks/GEOM/TEST/QEvent_Lifecycle_Test/ALL0/A000"
    (lldb) 

