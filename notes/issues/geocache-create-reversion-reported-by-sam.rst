geocache-create-reversion-reported-by-sam
===========================================


Reproduce the issue::

   op.sh -G -D

::

    (gdb) bt
    #0  0x00007fffeaf7d207 in raise () from /lib64/libc.so.6
    #1  0x00007fffeaf7e8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffeaf76026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffeaf760d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff50ef639 in GMeshLib::saveMeshes (this=0x665320, 
        idpath=0x6525a0 "/home/blyth/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/101") at /home/blyth/opticks/ggeo/GMeshLib.cc:493
    #5  0x00007ffff50ed72f in GMeshLib::save (this=0x665320) at /home/blyth/opticks/ggeo/GMeshLib.cc:66
    #6  0x00007ffff50e2239 in GGeo::save (this=0x653ff0) at /home/blyth/opticks/ggeo/GGeo.cc:717
    #7  0x00007ffff50e0ec1 in GGeo::loadGeometry (this=0x653ff0) at /home/blyth/opticks/ggeo/GGeo.cc:553
    #8  0x00007ffff64e9c71 in OpticksGeometry::loadGeometryBase (this=0x653170) at /home/blyth/opticks/opticksgeo/OpticksGeometry.cc:136
    #9  0x00007ffff64e96a9 in OpticksGeometry::loadGeometry (this=0x653170) at /home/blyth/opticks/opticksgeo/OpticksGeometry.cc:86
    #10 0x00007ffff64ee2e0 in OpticksHub::loadGeometry (this=0x63f1b0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:503
    #11 0x00007ffff64ecd4e in OpticksHub::init (this=0x63f1b0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:232
    #12 0x00007ffff64eca69 in OpticksHub::OpticksHub (this=0x63f1b0, ok=0x626250) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:198
    #13 0x00007ffff7bd58bf in OKMgr::OKMgr (this=0x7fffffffd980, argc=7, argv=0x7fffffffdaf8, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:35
    #14 0x0000000000402ead in main (argc=7, argv=0x7fffffffdaf8) at /home/blyth/opticks/ok/tests/OKTest.cc:13
    (gdb) 


Try just removing the solid assert, completes the geocache creation but a subsequent opticks-t 
(actually opticks-t1 for me on multiple GPU machine) gives 28 fails::

    FAILS:  28  / 406   :  Fri Jun 28 10:07:10 2019   
      39 /53  Test #39 : GGeoTest.GGeoLibTest                          Child aborted***Exception:     0.23   
      40 /53  Test #40 : GGeoTest.GGeoTest                             Child aborted***Exception:     0.25   
      52 /53  Test #52 : GGeoTest.GSceneTest                           Child aborted***Exception:     0.25   
      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 Child aborted***Exception:     0.27   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 Child aborted***Exception:     0.27   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.93   
      12 /24  Test #12 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     0.36   
      17 /24  Test #17 : OptiXRapTest.eventTest                        Child aborted***Exception:     0.36   
      18 /24  Test #18 : OptiXRapTest.interpolationTest                Child aborted***Exception:     0.39   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.26   
      1  /5   Test #1  : OKOPTest.OpIndexerTest                        Child aborted***Exception:     0.36   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     0.36   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     0.36   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     0.36   
      3  /5   Test #3  : OKTest.OTracerTest                            Child aborted***Exception:     0.39   
      1  /34  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     0.34   
      2  /34  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     0.34   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     0.34   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     0.33   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     0.33   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     0.35   
      22 /34  Test #22 : CFG4Test.CGenstepCollectorTest                Child aborted***Exception:     1.28   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     0.33   
      25 /34  Test #25 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     0.82   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     0.36   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     0.35   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     0.34   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     0.48   
    [blyth@localhost opticks]$ 




::

    [blyth@localhost ggeo]$ gdb GGeoLibTest 

    GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-114.el7
    ...
    (gdb) r
    Starting program: /home/blyth/local/opticks/lib/GGeoLibTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    2019-06-28 10:14:27.493 INFO  [138359] [Opticks::init@313] INTEROP_MODE
    2019-06-28 10:14:27.494 INFO  [138359] [Opticks::configure@1776]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
    GGeoLibTest: /home/blyth/opticks/ggeo/GPts.cc:81: void GPts::import(): Assertion `num_pt = m_ipt_buffer->getShape(0)' failed.
    
    Program received signal SIGABRT, Aborted.
    ...
    #3  0x00007ffff3e540d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7afc3c0 in GPts::import (this=0x762730) at /home/blyth/opticks/ggeo/GPts.cc:81
    #5  0x00007ffff7afc011 in GPts::Load (dir=0x765378 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/0") at /home/blyth/opticks/ggeo/GPts.cc:27
    #6  0x00007ffff7b164b0 in GGeoLib::loadConstituents (this=0x6bd450, idpath=0x6317f0 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae") at /home/blyth/opticks/ggeo/GGeoLib.cc:193
    #7  0x00007ffff7b15c28 in GGeoLib::loadFromCache (this=0x6bd450) at /home/blyth/opticks/ggeo/GGeoLib.cc:102
    #8  0x00007ffff7b15a16 in GGeoLib::Load (opticks=0x7fffffffd770, analytic=false, bndlib=0x6337f0) at /home/blyth/opticks/ggeo/GGeoLib.cc:54
    #9  0x0000000000404372 in main (argc=1, argv=0x7fffffffda28) at /home/blyth/opticks/ggeo/tests/GGeoLibTest.cc:143
    (gdb) 

    (gdb) f 6
    #6  0x00007ffff7b164b0 in GGeoLib::loadConstituents (this=0x6bd450, idpath=0x6317f0 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae") at /home/blyth/opticks/ggeo/GGeoLib.cc:193
    193         GPts*        pts = BFile::ExistsDir(ptspath) ? GPts::Load( ptspath ) : NULL ; 
    (gdb) p ptspath
    $1 = 0x765378 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/0"
    (gdb) 




Persisted GPts directory exists but the arrays are empty::

    [blyth@localhost opticks]$ np.py /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/0
    /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/0
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/0/GPts.txt :                    0 : d41d8cd98f00b204e9800998ecf8427e : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/0/iptBuffer.npy :               (0, 4) : f26f5c6534cf0b611f7f4ebbb4df67cb : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/0/plcBuffer.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190628-1005 
    [blyth@localhost opticks]$ 




::
    
    (gdb) f 4
    #4  0x00007ffff7afc3c0 in GPts::import (this=0x762730) at /home/blyth/opticks/ggeo/GPts.cc:81
    81      assert( num_pt = m_ipt_buffer->getShape(0)) ; 
                      ^^^^^^^^^^^^^ should be == , but its going to fail in other tests later with no GPts
    (gdb) p num_pt
    $2 = 0
    (gdb) p m_ipt_buffer->getShape(0)
    $3 = 0
    (gdb) p m_ipt_buffer
    $4 = (NPY<int> *) 0x9be570
    (gdb) 




After fixing that, get back to normal two fails and a visual check of analytic geometry looks fine::

   OKTest --xanalytic --gltf 1


Huh, how is it managing to work with no GPts ?::

    [blyth@localhost ggeo]$ np.py /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts
    /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/0/GPts.txt :                    0 : d41d8cd98f00b204e9800998ecf8427e : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/0/iptBuffer.npy :               (0, 4) : f26f5c6534cf0b611f7f4ebbb4df67cb : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/0/plcBuffer.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/1/GPts.txt :                    0 : d41d8cd98f00b204e9800998ecf8427e : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/1/iptBuffer.npy :               (0, 4) : f26f5c6534cf0b611f7f4ebbb4df67cb : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/1/plcBuffer.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/2/GPts.txt :                    0 : d41d8cd98f00b204e9800998ecf8427e : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/2/iptBuffer.npy :               (0, 4) : f26f5c6534cf0b611f7f4ebbb4df67cb : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/2/plcBuffer.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/3/GPts.txt :                    0 : d41d8cd98f00b204e9800998ecf8427e : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/3/iptBuffer.npy :               (0, 4) : f26f5c6534cf0b611f7f4ebbb4df67cb : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/3/plcBuffer.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/4/GPts.txt :                    0 : d41d8cd98f00b204e9800998ecf8427e : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/4/iptBuffer.npy :               (0, 4) : f26f5c6534cf0b611f7f4ebbb4df67cb : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/4/plcBuffer.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/5/GPts.txt :                    0 : d41d8cd98f00b204e9800998ecf8427e : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/5/iptBuffer.npy :               (0, 4) : f26f5c6534cf0b611f7f4ebbb4df67cb : 20190628-1005 
    . : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPts/5/plcBuffer.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190628-1005 
    [blyth@localhost ggeo]$ 



Notice GScene is still being instanciated, when thats now vestigial. 

* nope it's still being used, in legacy workflow the analytic comes in separately 
  via the GDML parse into GScene  

For a reminder on legacy workflow see :doc:`plan-removal-of-legacy-geometry-workflow-packages-and-externals`


