opticks-t-10-of-402-fails-2019-06-13
========================================


Up from 2 to 10 fails
---------------------------

::

    totals  10  / 402 


    FAILS:
      53 /120 Test #53 : NPYTest.NSlabTest                             ***Exception: SegFault         0.08   
      88 /120 Test #88 : NPYTest.NCSGRoundTripTest                     ***Exception: SegFault         0.09   
      38 /50  Test #38 : GGeoTest.GMakerTest                           ***Exception: SegFault         0.19   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.90   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.15   
      3  /18  Test #3  : ExtG4Test.X4SolidTest                         ***Exception: SegFault         0.14   
      11 /18  Test #11 : ExtG4Test.X4PhysicalVolumeTest                ***Exception: SegFault         0.15   
      12 /18  Test #12 : ExtG4Test.X4PhysicalVolume2Test               ***Exception: SegFault         0.17   
      16 /18  Test #16 : ExtG4Test.X4CSGTest                           ***Exception: SegFault         0.16   
      26 /34  Test #26 : CFG4Test.CMakerTest                           ***Exception: SegFault         0.16   



Many from lack of NPolygonizer metadata, from recent NCSG::postchange modification
---------------------------------------------------------------------------------------------

::

    nnode::composite_bbox  left [ 0:sp] P  right [ 0:sl] P  bb  mi (   -500.000  -500.000  -500.000) mx (    500.000   500.000   500.000) si (   1000.000  1000.000  1000.000)
    
    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff7a7058a in nlohmann::basic_json<std::map, std::vector, std::string, bool, long, unsigned long, double, std::allocator>::is_object (this=0x0)
        at /home/blyth/local/opticks/externals/include/YoctoGL/ext/json.hpp:2462
    2462            return m_type == value_t::object;
    (gdb) bt
    #0  0x00007ffff7a7058a in nlohmann::basic_json<std::map, std::vector, std::string, bool, long, unsigned long, double, std::allocator>::is_object (this=0x0)
        at /home/blyth/local/opticks/externals/include/YoctoGL/ext/json.hpp:2462
    #1  0x00007ffff7a95acc in nlohmann::basic_json<std::map, std::vector, std::string, bool, long, unsigned long, double, std::allocator>::count (this=0x0, 
        key="verbosity") at /home/blyth/local/opticks/externals/include/YoctoGL/ext/json.hpp:4243
    #2  0x00007ffff7a95a1f in NMeta::get<std::string> (this=0x0, name=0x7ffff7ade6ce "verbosity", fallback=0x7ffff7afd77e "0")
        at /home/blyth/opticks/npy/NMeta.cpp:207
    #3  0x00007ffff7a940ae in NMeta::getIntFromString (this=0x0, name=0x7ffff7ade6ce "verbosity", fallback=0x7ffff7ade6cc "0")
        at /home/blyth/opticks/npy/NMeta.cpp:232
    #4  0x00007ffff7a0cb0d in NPolygonizer::NPolygonizer (this=0x7fffffffd620, csg=0x6122c0) at /home/blyth/opticks/npy/NPolygonizer.cpp:77
    #5  0x00007ffff79dee47 in NCSG::polygonize (this=0x6122c0) at /home/blyth/opticks/npy/NCSG.cpp:1075
    #6  0x00007ffff79daf62 in NCSG::postchange (this=0x6122c0) at /home/blyth/opticks/npy/NCSG.cpp:165
    #7  0x00007ffff79daeaf in NCSG::Adopt (root=0x6115c0, config=0x611cb0, soIdx=0, lvIdx=0) at /home/blyth/opticks/npy/NCSG.cpp:140
    #8  0x0000000000404c22 in test_slab_sphere_intersection () at /home/blyth/opticks/npy/tests/NSlabTest.cc:103
    #9  0x0000000000404f35 in main (argc=1, argv=0x7fffffffda18) at /home/blyth/opticks/npy/tests/NSlabTest.cc:146
    (gdb) 


Polygonizer failing for lack of metadata::

    (gdb) f 6
    #6  0x00007ffff79daf62 in NCSG::postchange (this=0x6122c0) at /home/blyth/opticks/npy/NCSG.cpp:165
    165     if(m_config->polygonize) polygonize();
    (gdb) l
    160     collect_global_transforms();  // also sets the gtransform_idx onto the tree
    161     export_();                    // node tree -> complete binary tree m_nodes buffer
    162     export_srcidx();              // identity indices into srcidx buffer  : formerly was not done by NCSG::Load only NCSG::Adopt
    163 
    164     if(m_config->verbosity > 1) dump("NCSG::postchange");
    165     if(m_config->polygonize) polygonize();
    166 
    167     assert( getGTransformBuffer() );
    168     collect_surface_points();
    169 }
    (gdb) 


After change default polygonize OFF in NSceneConfig, down to 4 fails
----------------------------------------------------------------------------

::

    totals  4   / 402 


    FAILS:
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.89   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.23   
      11 /18  Test #11 : ExtG4Test.X4PhysicalVolumeTest                ***Exception: SegFault         0.16   
      12 /18  Test #12 : ExtG4Test.X4PhysicalVolume2Test               ***Exception: SegFault         0.18   


Same cause from last two, X4PhysicalVolumeTest::

    2019-06-13 15:44:34.916 INFO  [218992] [X4PhysicalVolume::convertSolid@500]  [ 0 Bubble

    Program received signal SIGSEGV, Segmentation fault.
    0x00007fffeeb9df2c in NCSG::postchange (this=0x719430) at /home/blyth/opticks/npy/NCSG.cpp:164
    164     if(m_config->verbosity > 1) dump("NCSG::postchange");
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffeeb9df2c in NCSG::postchange (this=0x719430) at /home/blyth/opticks/npy/NCSG.cpp:164
    #1  0x00007fffeeb9deaf in NCSG::Adopt (root=0x718c20, config=0x0, soIdx=0, lvIdx=0) at /home/blyth/opticks/npy/NCSG.cpp:140
    #2  0x00007ffff7b9c008 in X4PhysicalVolume::convertSolid (this=0x7fffffffd5e0, lvIdx=0, soIdx=0, solid=0x6c7d00, lvname="Bubble") at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:530
    #3  0x00007ffff7b9bac9 in X4PhysicalVolume::convertSolids_r (this=0x7fffffffd5e0, pv=0x6c7e90, depth=2) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:478
    #4  0x00007ffff7b9b974 in X4PhysicalVolume::convertSolids_r (this=0x7fffffffd5e0, pv=0x6c7c60, depth=1) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:464
    #5  0x00007ffff7b9b974 in X4PhysicalVolume::convertSolids_r (this=0x7fffffffd5e0, pv=0x6bcd90, depth=0) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:464
    #6  0x00007ffff7b9b719 in X4PhysicalVolume::convertSolids (this=0x7fffffffd5e0) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:436
    #7  0x00007ffff7b9a0a1 in X4PhysicalVolume::init (this=0x7fffffffd5e0) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:139
    #8  0x00007ffff7b99ec8 in X4PhysicalVolume::X4PhysicalVolume (this=0x7fffffffd5e0, ggeo=0x6ecdb0, top=0x6bcd90) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:123
    #9  0x00007ffff7b99bc8 in X4PhysicalVolume::Convert (top=0x6bcd90) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:89
    #10 0x0000000000402b64 in main (argc=1, argv=0x7fffffffda08) at /home/blyth/opticks/extg4/tests/X4PhysicalVolumeTest.cc:19
    (gdb) 



After checking m_config in NCSG::postchange are back to 2/402 
------------------------------------------------------------------

::

    totals  2   / 402 
    FAILS:
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.98   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.33  



