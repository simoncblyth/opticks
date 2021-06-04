tboolean_box_fail
====================

::

    O[blyth@localhost tests]$ ./tboolean_box.sh 

    2021-06-05 01:46:38.501 INFO  [65473] [CDetector::traverse@124] [
    2021-06-05 01:46:38.501 INFO  [65473] [CDetector::traverse@132] ]
    2021-06-05 01:46:38.501 FATAL [65473] [Opticks::setSpaceDomain@3263]  changing w 60000 -> 451
    OKG4Test: /home/blyth/opticks/cfg4/CMaterialBridge.cc:101: void CMaterialBridge::initMap(): Assertion `m_g4toix.size() == nmat_mlib' failed.

    (gdb) bt
    #3  0x00007fffe5734252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff4ab0320 in CMaterialBridge::initMap (this=0xa279480) at /home/blyth/opticks/cfg4/CMaterialBridge.cc:101
    #5  0x00007ffff4aafc14 in CMaterialBridge::CMaterialBridge (this=0xa279480, mlib=0x8e85ef0) at /home/blyth/opticks/cfg4/CMaterialBridge.cc:41
    #6  0x00007ffff4a85477 in CGeometry::postinitialize (this=0x91a8c60) at /home/blyth/opticks/cfg4/CGeometry.cc:143
    #7  0x00007ffff4af5d94 in CG4::postinitialize (this=0x8fb2a40) at /home/blyth/opticks/cfg4/CG4.cc:249
    #8  0x00007ffff4af5aef in CG4::initialize (this=0x8fb2a40) at /home/blyth/opticks/cfg4/CG4.cc:225
    #9  0x00007ffff4af583a in CG4::init (this=0x8fb2a40) at /home/blyth/opticks/cfg4/CG4.cc:195
    #10 0x00007ffff4af55e0 in CG4::CG4 (this=0x8fb2a40, hub=0x703c50) at /home/blyth/opticks/cfg4/CG4.cc:186
    #11 0x00007ffff7baf89d in OKG4Mgr::OKG4Mgr (this=0x7fffffff4440, argc=33, argv=0x7fffffff4788) at /home/blyth/opticks/okg4/OKG4Mgr.cc:107
    #12 0x00000000004038ba in main (argc=33, argv=0x7fffffff4788) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:27
    (gdb) 



On material different ?::

    2021-06-05 02:01:44.672 INFO  [89720] [CMaterialBridge::initMap@106] 
     nmat (G4Material::GetNumberOfMaterials) 3 nmat_mlib (GMaterialLib::getNumMaterials) materials used by geometry 4
     i   0 name                                Rock shortname                                Rock abbr                                Rock index     2 mlib_unset     0
     i   1 name                              Vacuum shortname                              Vacuum abbr                              Vacuum index     3 mlib_unset     0
     i   2 name                       GlassSchottF2 shortname                       GlassSchottF2 abbr                       GlassSchottF2 index     0 mlib_unset     0
     nmat 3 nmat_mlib 4 m_g4toix.size() 3 m_ixtoname.size() 3 m_ixtoabbr.size() 3

    OKG4Test: /home/blyth/opticks/cfg4/CMaterialBridge.cc:112: void CMaterialBridge::initMap(): Assertion `m_g4toix.size() == nmat_mlib' failed.

    Program received signal SIGABRT, Aborted.


This issue looks to be very specific to this test geometry, so its non urgent.


