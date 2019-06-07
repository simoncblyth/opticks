old-environment-still-being-setup
=================================

::

    Continuing.
    2019-06-07 21:04:28.078 INFO  [204046] [SSys::setenvvar@278]  ekey OPTICKSDATA_DAEPATH_DFAR ekv OPTICKSDATA_DAEPATH_DFAR=/home/blyth/local/opticks/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae overwrite 1 prior NULL value /home/blyth/local/opticks/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae after /home/blyth/local/opticks/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae rc 0
    
    Program received signal SIGINT, Interrupt.
    0x00007ffff44f649b in raise () from /lib64/libpthread.so.0
    (gdb) bt
    #0  0x00007ffff44f649b in raise () from /lib64/libpthread.so.0
    #1  0x00007fffe492a378 in SSys::setenvvar (ekey=0x6c6d48 "OPTICKSDATA_DAEPATH_DFAR", value=0x6cd2b8 "/home/blyth/local/opticks/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae", overwrite=true) at /home/blyth/opticks/sysrap/SSys.cc:288
    #2  0x00007fffe4c2c584 in BEnv::setEnvironment (this=0x6c7800, overwrite=true, native=true) at /home/blyth/opticks/boostrap/BEnv.cc:209
    #3  0x00007fffe5672cb1 in OpticksResource::readOpticksEnvironment (this=0x6c2f80) at /home/blyth/opticks/optickscore/OpticksResource.cc:504
    #4  0x00007fffe56713d6 in OpticksResource::init (this=0x6c2f80) at /home/blyth/opticks/optickscore/OpticksResource.cc:230
    #5  0x00007fffe5670d82 in OpticksResource::OpticksResource (this=0x6c2f80, ok=0x69f570) at /home/blyth/opticks/optickscore/OpticksResource.cc:94
    #6  0x00007fffe564fc35 in Opticks::initResource (this=0x69f570) at /home/blyth/opticks/optickscore/Opticks.cc:596
    #7  0x00007fffe56540ce in Opticks::configure (this=0x69f570) at /home/blyth/opticks/optickscore/Opticks.cc:1723
    #8  0x00007fffe70feb83 in OpticksHub::configure (this=0x6b8100) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:276
    #9  0x00007fffe70fe4de in OpticksHub::init (this=0x6b8100) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:213
    #10 0x00007fffe70fe356 in OpticksHub::OpticksHub (this=0x6b8100, ok=0x69f570) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:195
    #11 0x00007ffff7bd51ad in OKG4Mgr::OKG4Mgr (this=0x7fffffffcce0, argc=29, argv=0x7fffffffd018) at /home/blyth/opticks/okg4/OKG4Mgr.cc:71
    #12 0x0000000000403998 in main (argc=29, argv=0x7fffffffd018) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) 
    (gdb) c



    2019-06-07 21:08:41.216 INFO  [204046] [SSys::setenvvar@278]  ekey IDPATH ekv IDPATH=/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae overwrite 1 prior NULL value /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae after /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae rc 0
    
    Program received signal SIGINT, Interrupt.
    0x00007ffff44f649b in raise () from /lib64/libpthread.so.0
    (gdb) bt
    #0  0x00007ffff44f649b in raise () from /lib64/libpthread.so.0
    #1  0x00007fffe492a378 in SSys::setenvvar (ekey=0x7fffe4c5d169 "IDPATH", value=0x6ce020 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae", overwrite=true) at /home/blyth/opticks/sysrap/SSys.cc:288
    #2  0x00007fffe4c31098 in BOpticksResource::setupViaSrc (this=0x6c2f80, srcpath=0x6cd428 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae", srcdigest=0x6c60f0 "96ff965744a2f6b78c24e33c80d3a4cd")
        at /home/blyth/opticks/boostrap/BOpticksResource.cc:701
    #3  0x00007fffe5673229 in OpticksResource::readEnvironment (this=0x6c2f80) at /home/blyth/opticks/optickscore/OpticksResource.cc:588
    #4  0x00007fffe5671409 in OpticksResource::init (this=0x6c2f80) at /home/blyth/opticks/optickscore/OpticksResource.cc:238
    #5  0x00007fffe5670d82 in OpticksResource::OpticksResource (this=0x6c2f80, ok=0x69f570) at /home/blyth/opticks/optickscore/OpticksResource.cc:94
    #6  0x00007fffe564fc35 in Opticks::initResource (this=0x69f570) at /home/blyth/opticks/optickscore/Opticks.cc:596
    #7  0x00007fffe56540ce in Opticks::configure (this=0x69f570) at /home/blyth/opticks/optickscore/Opticks.cc:1723
    #8  0x00007fffe70feb83 in OpticksHub::configure (this=0x6b8100) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:276
    #9  0x00007fffe70fe4de in OpticksHub::init (this=0x6b8100) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:213
    #10 0x00007fffe70fe356 in OpticksHub::OpticksHub (this=0x6b8100, ok=0x69f570) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:195
    #11 0x00007ffff7bd51ad in OKG4Mgr::OKG4Mgr (this=0x7fffffffcce0, argc=29, argv=0x7fffffffd018) at /home/blyth/opticks/okg4/OKG4Mgr.cc:71
    #12 0x0000000000403998 in main (argc=29, argv=0x7fffffffd018) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) 

