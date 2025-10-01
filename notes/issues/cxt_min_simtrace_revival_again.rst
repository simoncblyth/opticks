cxt_min_simtrace_revival_again
=================================

Overview
-----------

* Again, it runs but no simtrace array.

* HMM, recent genstep rejig for server-client : could potentially have broken simtrace.

* NOPE, nothing complicated : just bash level stomping by EVT envvar caused looking in wrong place




trace check basics, like genstep setup
---------------------------------------

::

    P[blyth@localhost CSGOptiX]$ LOG=1 BP=SEvt::SEvt ~/o/cxt_min.sh 

    Breakpoint 1, SEvt::SEvt (this=0x1608dcf0) at /home/blyth/opticks/sysrap/SEvt.cc:223
    223	    addGenstep_array(0)
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64
    (gdb) bt
    #0  SEvt::SEvt (this=0x1608dcf0) at /home/blyth/opticks/sysrap/SEvt.cc:223
    #1  0x00007ffff5949be2 in SEvt::Create (ins=0) at /home/blyth/opticks/sysrap/SEvt.cc:1270
    #2  0x00007ffff5949f4d in SEvt::CreateOrReuse (idx=0) at /home/blyth/opticks/sysrap/SEvt.cc:1328
    #3  0x00007ffff594a1db in SEvt::CreateOrReuse () at /home/blyth/opticks/sysrap/SEvt.cc:1372
    #4  0x00007ffff7652548 in CSGFoundry::AfterLoadOrCreate () at /home/blyth/opticks/CSG/CSGFoundry.cc:3702
    #5  0x00007ffff764f9ce in CSGFoundry::Load () at /home/blyth/opticks/CSG/CSGFoundry.cc:3063
    #6  0x00007ffff7e324a8 in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:155
    #7  0x0000000000404a95 in main (argc=1, argv=0x7fffffffb258) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 

    (gdb) b SEvt::beginOfEvent
    Breakpoint 2 at 0x7ffff594b451: file /home/blyth/opticks/sysrap/SEvt.cc, line 1759.
    (gdb) c
    Continuing.

    Thread 1 "CSGOptiXTMTest" hit Breakpoint 2, SEvt::beginOfEvent (this=0x1608dcf0, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1759
    1759	    if(isFirstEvtInstance() && eventID == 0) BeginOfRun() ;
    Missing separate debuginfos, use: dnf debuginfo-install libnvidia-gpucomp-580.82.07-1.el9.x86_64 libnvidia-ml-580.82.07-1.el9.x86_64 nvidia-driver-cuda-libs-580.82.07-1.el9.x86_64 nvidia-driver-libs-580.82.07-1.el9.x86_64
    (gdb) bt
    #0  SEvt::beginOfEvent (this=0x1608dcf0, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1759
    #1  0x00007ffff5e732f2 in QSim::simtrace (this=0x186387d0, eventID=0) at /home/blyth/opticks/qudarap/QSim.cc:722
    #2  0x00007ffff7e35cdd in CSGOptiX::simtrace (this=0x1864ccf0, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:755
    #3  0x00007ffff7e324cd in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:157
    #4  0x0000000000404a95 in main (argc=1, argv=0x7fffffffb258) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) f 0
    #0  SEvt::beginOfEvent (this=0x1608dcf0, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1759
    1759	    if(isFirstEvtInstance() && eventID == 0) BeginOfRun() ;
    (gdb) p addGenstep_array
    $1 = 0
    (gdb) 

    (gdb) b SEvt::addInputGenstep
    Breakpoint 3 at 0x7ffff59484e3: file /home/blyth/opticks/sysrap/SEvt.cc, line 1002.
    (gdb) c
    Continuing.
    2025-10-01 14:14:22.173 INFO  [204562] [SEvt::beginOfEvent@1768] SEvt::id EGPU (0)  GSV NO  SEvt__beginOfEvent
    2025-10-01 14:14:22.174 INFO  [204562] [SEvt::clear_output@2040] SEvt::id EGPU (0)  GSV NO  SEvt__OTHER BEFORE clear_output_vector 
    2025-10-01 14:14:22.174 INFO  [204562] [SEvt::clear_output@2050] SEvt::id EGPU (0)  GSV NO  SEvt__OTHER AFTER clear_output_vector 

    Thread 1 "CSGOptiXTMTest" hit Breakpoint 3, SEvt::addInputGenstep (this=0x1608dcf0) at /home/blyth/opticks/sysrap/SEvt.cc:1002
    1002	    LOG_IF(info, LIFECYCLE) << id() ;
    (gdb) bt
    #0  SEvt::addInputGenstep (this=0x1608dcf0) at /home/blyth/opticks/sysrap/SEvt.cc:1002
    #1  0x00007ffff594b6ec in SEvt::beginOfEvent (this=0x1608dcf0, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1773
    #2  0x00007ffff5e732f2 in QSim::simtrace (this=0x186387d0, eventID=0) at /home/blyth/opticks/qudarap/QSim.cc:722
    #3  0x00007ffff7e35cdd in CSGOptiX::simtrace (this=0x1864ccf0, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:755
    #4  0x00007ffff7e324cd in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:157
    #5  0x0000000000404a95 in main (argc=1, argv=0x7fffffffb258) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 



    1000 void SEvt::addInputGenstep()
    1001 {
    1002     LOG_IF(info, LIFECYCLE) << id() ;
    1003     LOG(LEVEL);
    1004 
    1005     NP* igs = createInputGenstep_configured();
    1006     addGenstep(igs);
    1007 }

     970 NP* SEvt::createInputGenstep_configured()
     971 {
     972     NP* igs = nullptr ;
     973     if(SEventConfig::IsRGModeSimtrace())
     974     {
     975         igs = createInputGenstep_simtrace();
     976     }
     977     else if(SEventConfig::IsRGModeSimulate())
     978     {
     979         igs = createInputGenstep_simulate();
     980     }
     981     return igs ;
     982 }




    (gdb) b SEvt::addGenstep
    Breakpoint 4 at 0x7ffff594d78e: SEvt::addGenstep. (2 locations)
    (gdb) b SEvt::createInputGenstep_configured
    Breakpoint 5 at 0x7ffff594848e: file /home/blyth/opticks/sysrap/SEvt.cc, line 972.
    (gdb) c
    Continuing.
    2025-10-01 14:16:39.266 INFO  [204562] [SEvt::addInputGenstep@1002] SEvt::id EGPU (0)  GSV NO  SEvt__OTHER

    Thread 1 "CSGOptiXTMTest" hit Breakpoint 5, SEvt::createInputGenstep_configured (this=0x1608dcf0) at /home/blyth/opticks/sysrap/SEvt.cc:972
    972	    NP* igs = nullptr ;
    (gdb) 


    (gdb) b SEvt::createInputGenstep_simtrace
    Breakpoint 6 at 0x7ffff5947b75: file /home/blyth/opticks/sysrap/SEvt.cc, line 857.
    (gdb) c
    Continuing.

    Thread 1 "CSGOptiXTMTest" hit Breakpoint 6, SEvt::createInputGenstep_simtrace (this=0x1608dcf0) at /home/blyth/opticks/sysrap/SEvt.cc:857
    857	    NP* igs = nullptr ;
    (gdb) bt
    #0  SEvt::createInputGenstep_simtrace (this=0x1608dcf0) at /home/blyth/opticks/sysrap/SEvt.cc:857
    #1  0x00007ffff59484ab in SEvt::createInputGenstep_configured (this=0x1608dcf0) at /home/blyth/opticks/sysrap/SEvt.cc:975
    #2  0x00007ffff594868b in SEvt::addInputGenstep (this=0x1608dcf0) at /home/blyth/opticks/sysrap/SEvt.cc:1005
    #3  0x00007ffff594b6ec in SEvt::beginOfEvent (this=0x1608dcf0, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1773
    #4  0x00007ffff5e732f2 in QSim::simtrace (this=0x186387d0, eventID=0) at /home/blyth/opticks/qudarap/QSim.cc:722
    #5  0x00007ffff7e35cdd in CSGOptiX::simtrace (this=0x1864ccf0, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:755
    #6  0x00007ffff7e324cd in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:157
    #7  0x0000000000404a95 in main (argc=1, argv=0x7fffffffb258) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 



    (gdb) bt
    #0  SEvt::addGenstep (this=0x1608dcf0, a=0x1afcb190) at /home/blyth/opticks/sysrap/SEvt.cc:2239
    #1  0x00007ffff59486a5 in SEvt::addInputGenstep (this=0x1608dcf0) at /home/blyth/opticks/sysrap/SEvt.cc:1006
    #2  0x00007ffff594b6ec in SEvt::beginOfEvent (this=0x1608dcf0, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1773
    #3  0x00007ffff5e732f2 in QSim::simtrace (this=0x186387d0, eventID=0) at /home/blyth/opticks/qudarap/QSim.cc:722
    #4  0x00007ffff7e35cdd in CSGOptiX::simtrace (this=0x1864ccf0, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:755
    #5  0x00007ffff7e324cd in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:157
    #6  0x0000000000404a95 in main (argc=1, argv=0x7fffffffb258) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) f 0
    #0  SEvt::addGenstep (this=0x1608dcf0, a=0x1afcb190) at /home/blyth/opticks/sysrap/SEvt.cc:2239
    2239	    sgs s = {} ;
    (gdb) p a->sstr()
    $3 = {static npos = 18446744073709551615, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0x7fffffff9890 "(7077, 6, 4, )"}, _M_string_length = 14, {
        _M_local_buf = "(7077, 6, 4, )\000", _M_allocated_capacity = 3900165892963579688}}
    (gdb) 



HMM, maybe just script looking in wrong dir::

    A[blyth@localhost qudarap]$ l /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXTMTest/ALL0_no_opticks_event_name/A000/
    total 885320
         4 -rw-r--r--. 1 blyth blyth       677 Oct  1 14:28 NPFold_meta.txt
         0 -rw-r--r--. 1 blyth blyth         0 Oct  1 14:28 NPFold_names.txt
         4 -rw-r--r--. 1 blyth blyth       133 Oct  1 14:28 sframe_meta.txt
         4 -rw-r--r--. 1 blyth blyth       384 Oct  1 14:28 sframe.npy
         4 -rw-r--r--. 1 blyth blyth        25 Oct  1 14:28 NPFold_index.txt
    884632 -rw-r--r--. 1 blyth blyth 905856128 Oct  1 14:28 simtrace.npy
       664 -rw-r--r--. 1 blyth blyth    679520 Oct  1 14:28 genstep.npy
         4 drwxr-xr-x. 2 blyth blyth      4096 Jul 10 16:30 .
         4 drwxr-xr-x. 3 blyth blyth      4096 Jul 10 16:30 ..
    A[blyth@localhost qudarap]$ 


LOG is going to some other dir with MOI in path, but nothing else written there::

    A[blyth@localhost qudarap]$ l /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXTMTest/Waterdistributor_2:0:-2/
    total 740
    732 -rw-r--r--. 1 blyth blyth 745932 Oct  1 14:28 CSGOptiXTMTest.log
      4 drwxr-xr-x. 2 blyth blyth   4096 Oct  1 11:37 .
      4 drwxr-xr-x. 6 blyth blyth   4096 Sep 30 20:48 ..
    A[blyth@localhost qudarap]$ 





Adding points suspected as being overlapped
--------------------------------------------------

::


    (ok) A[blyth@localhost sysrap]$ BP=SFrameGenstep::StandardizeCEGS cxt_min.sh


    (gdb) bt
    #0  SFrameGenstep::StandardizeCEGS (cegs=...) at /home/blyth/opticks/sysrap/SFrameGenstep.cc:467
    #1  0x00007ffff5905f47 in SFrameGenstep::GetGridConfig (cegs=..., ekey=0x7ffff5aa0904 "CEGS", delim=58 ':', fallback=0x7ffff5aa08f8 "16:0:9:1000") at /home/blyth/opticks/sysrap/SFrameGenstep.cc:132
    #2  0x00007ffff59061b9 in SFrameGenstep::MakeCenterExtentGenstep_FromFrame (fr=...) at /home/blyth/opticks/sysrap/SFrameGenstep.cc:180
    #3  0x00007ffff5947e97 in SEvt::createInputGenstep_simtrace (this=0x1608dd10) at /home/blyth/opticks/sysrap/SEvt.cc:880
    #4  0x00007ffff59484ab in SEvt::createInputGenstep_configured (this=0x1608dd10) at /home/blyth/opticks/sysrap/SEvt.cc:975
    #5  0x00007ffff594868b in SEvt::addInputGenstep (this=0x1608dd10) at /home/blyth/opticks/sysrap/SEvt.cc:1005
    #6  0x00007ffff594b6ec in SEvt::beginOfEvent (this=0x1608dd10, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1773
    #7  0x00007ffff5e732f2 in QSim::simtrace (this=0x186387d0, eventID=0) at /home/blyth/opticks/qudarap/QSim.cc:722
    #8  0x00007ffff7e35cdd in CSGOptiX::simtrace (this=0x1864cb50, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:755
    #9  0x00007ffff7e324cd in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:157
    #10 0x0000000000404a95 in main (argc=1, argv=0x7fffffffb2f8) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 


