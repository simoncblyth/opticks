CRandomEngineTest_SEGV_on_Ubuntu
======================================


Investigate
-------------

1. reimplemented as CRandomEngine::CurrentGeant4ProcessName 


To Check 
----------

::

   CRandomEngine=INFO CRandomEngineTest 
   


Issue still present with CRandomEngine::CurrentGeant4ProcessName
------------------------------------------------------------------

Notice the G4VProcess address looks like a dummy : 0x303030303030303

::

    2020-11-12 20:47:58.559 INFO  [90592] [CRandomEngine::init@128] [
    2020-11-12 20:47:58.559 INFO  [90592] [CRandomEngine::initCurand@177]  DYNAMIC_CURAND 100000,16,16 curand_ni 100000 curand_nv 256
    2020-11-12 20:47:58.559 INFO  [90592] [CRandomEngine::init@131] ]
    2020-11-12 20:47:58.559 INFO  [90592] [CRandomEngine::setupTranche@239]  DYNAMIC_CURAND  m_tranche_id 0 m_tranche_size 100000 m_tranche_ibase 0
    2020-11-12 20:47:58.703 INFO  [90592] [CRandomEngine::setupCurandSequence@298]  record_id 0 m_tranche_id 0 m_tranche_size 100000 m_tranche_index 0 m_curand_ni 100000 m_curand_nv 256
    2020-11-12 20:47:58.704 INFO  [90592] [CRandomEngineTest::print@59] record_id 0
    2020-11-12 20:47:58.704 INFO  [90592] [CRandomEngine::CurrentGeant4ProcessName@345] [ 0x303030303030303

    Thread 1 "CRandomEngineTe" received signal SIGSEGV, Segmentation fault.
    0x00007ffff6be2738 in std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) ()
       from /lib/x86_64-linux-gnu/libstdc++.so.6
    (gdb) bt
    #0  0x00007ffff6be2738 in std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) ()
       from /lib/x86_64-linux-gnu/libstdc++.so.6
    #1  0x00007ffff7f4070a in CRandomEngine::CurrentGeant4ProcessName[abi:cxx11]() ()
        at /home/lab110/opticks/cfg4/CRandomEngine.cc:348
    #2  0x00007ffff7f40b74 in CRandomEngine::flat (this=0x7fffffffccb8) at /home/lab110/opticks/cfg4/CRandomEngine.cc:424
    #3  0x000055555555b277 in CRandomEngineTest::print (this=0x7fffffffccb0, record_id=0)
        at /home/lab110/opticks/cfg4/tests/CRandomEngineTest.cc:63
    #4  0x0000555555559ccd in main (argc=1, argv=0x7fffffffcfb8)
        at /home/lab110/opticks/cfg4/tests/CRandomEngineTest.cc:113



Issue SEGV in CRandomEngine::CurrentProcessName on Ubuntu reported by Fan Hu
--------------------------------------------------------------------------------

::

    Sorry, I use backtrack now, the output is:
    lab110@lab110-MS-7B89:~/local/opticks/build/cfg4/tests$ gdb CRandomEngineTest 
    GNU gdb (Ubuntu 9.2-0ubuntu1~20.04) 9.2
    Copyright (C) 2020 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.
    Type "show copying" and "show warranty" for details.
    This GDB was configured as "x86_64-linux-gnu".


    2020-11-11 22:37:54.367 INFO  [6006] [CGDMLDetector::standardizeGeant4MaterialProperties@242] [
    2020-11-11 22:37:54.367 INFO  [6006] [X4MaterialLib::init@127]     0 ok pmap_name                  G4_PLEXIGLASS g4 m4_name                 G4_PLEXIGLASS g4 m4_name_base                   G4_PLEXIGLASS has_prefix 0
    2020-11-11 22:37:54.367 INFO  [6006] [X4MaterialLib::init@127]     1 ok pmap_name                       G4_WATER g4 m4_name                 G4_WATER g4 m4_name_base                        G4_WATER has_prefix 0
    2020-11-11 22:37:54.367 INFO  [6006] [CGDMLDetector::standardizeGeant4MaterialProperties@244] ]
    2020-11-11 22:37:54.367 ERROR [6006] [CDetector::attachSurfaces@376]  some surfaces were found : so assume there is nothing to do 
    2020-11-11 22:37:54.368 FATAL [6006] [CTorchSource::configure@163] CTorchSource::configure _t 0.1 _radius 0 _pos 0.0000,0.0000,0.0000 _dir 0.0000,0.0000,1.0000 _zeaz 0.0000,1.0000,0.0000,1.0000 _pol 0.0000,0.0000,1.0000
       0                         DomSurface           DomSurface0x7fe41a882000 lv DOM_LV0x7fe41a881890
    [New Thread 0x7fffe7d20700 (LWP 6013)]
    [New Thread 0x7fffe751f700 (LWP 6014)]
    [New Thread 0x7fffe6b9b700 (LWP 6015)]
    2020-11-11 22:37:54.667 INFO  [6006] [CRandomEngineTest::print@59] record_id 0

    Thread 1 "CRandomEngineTe" received signal SIGSEGV, Segmentation fault.
    0x00007ffff6be19e4 in std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::c_str() const
       () from /lib/x86_64-linux-gnu/libstdc++.so.6
    (gdb) bt
    #0  0x00007ffff6be19e4 in std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::c_str() const () from /lib/x86_64-linux-gnu/libstdc++.so.6
    #1  0x00007ffff7f4061b in CRandomEngine::CurrentProcessName[abi:cxx11]() ()
       at /home/lab110/opticks/cfg4/CRandomEngine.cc:339
    #2  0x00007ffff7f4098e in CRandomEngine::flat (this=0x7fffffffd008) at /home/lab110/opticks/cfg4/CRandomEngine.cc:414
    #3  0x000055555555b277 in CRandomEngineTest::print (this=0x7fffffffd000, record_id=0)
       at /home/lab110/opticks/cfg4/tests/CRandomEngineTest.cc:63
    #4  0x0000555555559ccd in main (argc=1, argv=0x7fffffffd308)
       at /home/lab110/opticks/cfg4/tests/CRandomEngineTest.cc:113
    (gdb) c
    Continuing.
    Couldn't get registers: No such process.
    Couldn't get registers: No such process.
    (gdb) [Thread 0x7fffe6b9b700 (LWP 6015) exited]
    [Thread 0x7fffe751f700 (LWP 6014) exited]
    [Thread 0x7fffe7d20700 (LWP 6013) exited]

    Program terminated with signal SIGSEGV, Segmentation fault.
    The program no longer exists.


    Hope they can help.
    Fan




Hmm this assumes running under Geant4 
--------------------------------------

::

    411 
    412 double CRandomEngine::flat()
    413 {
    414     if(!m_internal) m_location = CurrentProcessName();
    415     assert( m_current_record_flat_count < m_curand_nv );
    416 

    335 std::string CRandomEngine::CurrentProcessName()
    336 {
    337     G4VProcess* proc = CProcess::CurrentProcess() ;
    338     std::stringstream ss ;
    339     ss <<  ( proc ? proc->GetProcessName().c_str() : "NoProc" )  ;
    340     return ss.str();
    341 }

    059 G4VProcess* CProcess::CurrentProcess()
     60 {
     61     G4EventManager* evtMgr = G4EventManager::GetEventManager() ;
     62     G4TrackingManager* trkMgr = evtMgr->GetTrackingManager() ;
     63     G4SteppingManager* stepMgr = trkMgr->GetSteppingManager() ;
     64     return stepMgr->GetfCurrentProcess() ;
     65 }


