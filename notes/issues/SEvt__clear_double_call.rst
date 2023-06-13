SEvt__clear_double_call
=========================



::

    /**
    SEvt::Clear
    -------------

    SEvt::Clear is invoked in two situations:

    1. from QEvent::setGenstep after SEvt::GatherGenstep to prepare 
       for further genstep collection, this is done immediately prior to 
       simulation launches

    2. from SEvt::EndOfEvent immediately after SEvt::Save

    **/

    void SEvt::Clear(){ Check() ; INSTANCE->clear();  }



Check Potential Over Clearing
--------------------------------

::

     963 void SEvt::clear_()
     964 {
     965     numphoton_collected = 0u ;
     966     numphoton_genstep_max = 0u ;
     967     clear_count += 1 ;
     968 
     969     if(DEBUG_CLEAR > 0)
     970     {
     971         LOG(info)
     972            << " DEBUG_CLEAR " << DEBUG_CLEAR
     973            << " clear_count " << clear_count
     974            ;
     975         std::raise(SIGINT);
     976     }







::

    2023-06-13 18:23:17.898 INFO  [431000] [SEvt::clear_@971]  DEBUG_CLEAR 1 clear_count 1

    Program received signal SIGINT, Interrupt.
    (gdb) bt
    #0  0x00007ffff501d4fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff53ae812 in SEvt::clear_ (this=0x66c040) at /data/blyth/junotop/opticks/sysrap/SEvt.cc:975
    #2  0x00007ffff53aea7d in SEvt::clear (this=0x66c040) at /data/blyth/junotop/opticks/sysrap/SEvt.cc:1016
    #3  0x00007ffff53adf6d in SEvt::Clear () at /data/blyth/junotop/opticks/sysrap/SEvt.cc:823
    #4  0x00007ffff5808693 in QEvent::setGenstep (this=0xe230b0) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:151
    #5  0x00007ffff57df682 in QSim::simulate (this=0xeb9570) at /data/blyth/junotop/opticks/qudarap/QSim.cc:300
    #6  0x00007ffff7aae9d7 in CSGOptiX::SimulateMain () at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:186
    #7  0x000000000040fea1 in main (argc=1, argv=0x7fffffff5268) at /data/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) 
    (gdb) c
    Continuing.
    2023-06-13 18:25:22.556 INFO  [431000] [CSGOptiX::simulate@914] 
    2023-06-13 18:25:22.568 INFO  [431000] [SEvt::save@2657]  dir /home/blyth/.opticks/GEOM/V1J009/CSGOptiXSMTest/ALL/000
    2023-06-13 18:25:22.705 INFO  [431000] [SEvt::clear_@971]  DEBUG_CLEAR 1 clear_count 2

    Program received signal SIGINT, Interrupt.
    0x00007ffff501d4fb in raise () from /lib64/libpthread.so.0
    (gdb) bt
    #0  0x00007ffff501d4fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff53ae812 in SEvt::clear_ (this=0x66c040) at /data/blyth/junotop/opticks/sysrap/SEvt.cc:975
    #2  0x00007ffff53aea7d in SEvt::clear (this=0x66c040) at /data/blyth/junotop/opticks/sysrap/SEvt.cc:1016
    #3  0x00007ffff53adf6d in SEvt::Clear () at /data/blyth/junotop/opticks/sysrap/SEvt.cc:823
    #4  0x00007ffff53ae369 in SEvt::EndOfEvent (index=0) at /data/blyth/junotop/opticks/sysrap/SEvt.cc:906
    #5  0x00007ffff7aae9e1 in CSGOptiX::SimulateMain () at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:189
    #6  0x000000000040fea1 in main (argc=1, argv=0x7fffffff5268) at /data/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) 



