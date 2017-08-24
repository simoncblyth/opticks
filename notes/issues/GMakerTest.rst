GMakerTest fail on Linux reported by YL
==========================================

And here's the details for GMakerTest:


::

    gdb GMakerTest
    (gdb) r
    Starting program: /home/roy/opticks/lib/GMakerTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2017-08-24 10:34:20.316 WARN  [30613] [GMaker::init@176] GMaker::init booting from cache
    2017-08-24 10:34:20.316 INFO  [30613] [GMergedMesh::load@631] GMergedMesh::load dir /home/roy/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /home/roy/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-08-24 10:34:20.483 INFO  [30613] [GMergedMesh::load@631] GMergedMesh::load dir /home/roy/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /home/roy/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-08-24 10:34:20.489 INFO  [30613] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-08-24 10:34:20.489 INFO  [30613] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-08-24 10:34:20.489 INFO  [30613] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-08-24 10:34:20.492 INFO  [30613] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-08-24 10:34:20.497 INFO  [30613] [NSceneConfig::NSceneConfig@48] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0]
            check_surf_containment :                    0
            check_aabb_containment :                    0

    Program received signal SIGSEGV, Segmentation fault.
    0x0000000000000000 in ?? ()
    (gdb) bt
    #0  0x0000000000000000 in ?? ()
    #1  0x00007ffff7231c76 in NNodeDump::dump_label (this=0x1353460, pfx=0x7ffff735958e "du") at /home/roy/opticks/opticksnpy/NNodeDump.cpp:37
    #2  0x00007ffff7231d67 in NNodeDump::dump_base (this=0x1353460) at /home/roy/opticks/opticksnpy/NNodeDump.cpp:43
    #3  0x00007ffff7231bee in NNodeDump::dump (this=0x1353460) at /home/roy/opticks/opticksnpy/NNodeDump.cpp:22
    #4  0x00007ffff72253ac in nnode::dump (this=0x1353480, msg=0x0) at /home/roy/opticks/opticksnpy/NNode.cpp:1097
    #5  0x0000000000405599 in GMakerTest::makeFromCSG (this=0x7fffffffd7d0) at /home/roy/opticks/ggeo/tests/GMakerTest.cc:70
    #6  0x0000000000405915 in main (argc=1, argv=0x7fffffffda18) at /home/roy/opticks/ggeo/tests/GMakerTest.cc:99

