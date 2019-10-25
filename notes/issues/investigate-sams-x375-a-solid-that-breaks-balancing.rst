investigate-sams-x375-a-solid-that-breaks-balancing
====================================================


Issue
-------

* https://github.com/63990/Opticks_install_guide
* https://groups.io/g/opticks/topic/geometry_balancing/32989642



Sam added backtrace
----------------------

* immediately see from the idx values that are trying and failing to export the unbalanced tree
  
  * given that the tree has more than a billion nodes, that is not too surprising
 
::

   In [17]: (1 << 30) - 1
   Out[17]: 1073741823



Opticks_install_guide/geometry_balancing_backtrace.txt::

     01 (gdb) bt
      2 #0  0x00007fffe20d12c7 in __GI_raise (sig=sig@entry=6) at ../nptl/sysdeps/unix/sysv/linux/raise.c:55
      3 #1  0x00007fffe20d29b8 in __GI_abort () at abort.c:90
      4 #2  0x00007fffe20ca0e6 in __assert_fail_base (
      5     fmt=0x7fffe2225020 "%s%s%s:%u: %s%sAssertion `%s' failed.\n%n",
      6     assertion=assertion@entry=0x7fffe9c4d30c "idx < getNumNodes()",
      7     file=file@entry=0x7fffe9c4ce88 "/home/opc/opticks/npy/NCSG.cpp", line=line@entry=899,
      8     function=function@entry=0x7fffe9c4dba0 <NCSG::export_node(nnode*, unsigned int)::__PRETTY_FUNCTION__> "void NCSG::export_node(nnode*, unsigned int)") at assert.c:92
      9 #3  0x00007fffe20ca192 in __GI___assert_fail (assertion=0x7fffe9c4d30c "idx < getNumNodes()",
     10     file=0x7fffe9c4ce88 "/home/opc/opticks/npy/NCSG.cpp", line=899,
     11     function=0x7fffe9c4dba0 <NCSG::export_node(nnode*, unsigned int)::__PRETTY_FUNCTION__> "void NCSG::export_node(nnode*, unsigned int)") at assert.c:101
     12 #4  0x00007fffe9b55f22 in NCSG::export_node (this=0xa65ba50, node=0x9a98c50, idx=1073741823)
     13     at /home/opc/opticks/npy/NCSG.cpp:899
     14 #5  0x00007fffe9b55e6f in NCSG::export_r (this=0xa65ba50, node=0x9a98c50, idx=1073741823)
     15     at /home/opc/opticks/npy/NCSG.cpp:875
     16 #6  0x00007fffe9b55ea8 in NCSG::export_r (this=0xa65ba50, node=0x9a996e0, idx=536870911)
     17     at /home/opc/opticks/npy/NCSG.cpp:879
     18 #7  0x00007fffe9b55ea8 in NCSG::export_r (this=0xa65ba50, node=0x9a9a170, idx=268435455)
     19     at /home/opc/opticks/npy/NCSG.cpp:879
     20 #8  0x00007fffe9b55ea8 in NCSG::export_r (this=0xa65ba50, node=0x9a9ac00, idx=134217727)

     ...

     71     at /home/opc/opticks/npy/NCSG.cpp:879
     72 #34 0x00007fffe9b55ea8 in NCSG::export_r (this=0xa65ba50, node=0x9aabe90, idx=1)
     73     at /home/opc/opticks/npy/NCSG.cpp:879
     74 #35 0x00007fffe9b55ea8 in NCSG::export_r (this=0xa65ba50, node=0x9aac920, idx=0)
     75     at /home/opc/opticks/npy/NCSG.cpp:879
     76 #36 0x00007fffe9b55d2f in NCSG::export_ (this=0xa65ba50) at /home/opc/opticks/npy/NCSG.cpp:845
     77 #37 0x00007fffe9b52f3a in NCSG::postchange (this=0xa65ba50) at /home/opc/opticks/npy/NCSG.cpp:162
     78 #38 0x00007fffe9b52ed1 in NCSG::Adopt (root=0x9aac920, config=0x0, soIdx=375, lvIdx=375)
     79     at /home/opc/opticks/npy/NCSG.cpp:141
     80 #39 0x00007ffff4950635 in X4PhysicalVolume::convertSolid (this=0x7fffffffd110, lvIdx=375, soIdx=375,
     81     solid=0x8d6570, lvname="topTitaniumPlate_log0x607bf70", balance_deep_tree=false)
                                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
     82     at /home/opc/opticks/extg4/X4PhysicalVolume.cc:605
     83 #40 0x00007ffff495019a in X4PhysicalVolume::convertSolids_r (this=0x7fffffffd110, pv=0xe26620,
     84     depth=10) at /home/opc/opticks/extg4/X4PhysicalVolume.cc:529
     85 #41 0x00007ffff494ffcc in X4PhysicalVolume::convertSolids_r (this=0x7fffffffd110, pv=0xed0a30, depth=9)
     86     at /home/opc/opticks/extg4/X4PhysicalVolume.cc:506
     87 #42 0x00007ffff494ffcc in X4PhysicalVolume::convertSolids_r (this=0x7fffffffd110, pv=0x1146f70,
     88     depth=8) at /home/opc/opticks/extg4/X4PhysicalVolume.cc:506
     89 #43 0x00007ffff494ffcc in X4PhysicalVolume::convertSolids_r (this=0x7fffffffd110, pv=0x1147d80,
     90     depth=7) at /home/opc/opticks/extg4/X4PhysicalVolume.cc:506
     91 #44 0x00007ffff494ffcc in X4PhysicalVolume::convertSolids_r (this=0x7fffffffd110, pv=0x1148290,
     92     depth=6) at /home/opc/opticks/extg4/X4PhysicalVolume.cc:506
     93 #45 0x00007ffff494ffcc in X4PhysicalVolume::convertSolids_r (this=0x7fffffffd110, pv=0x11487b0,



This is from the 2nd unbalanced convert for setting of the alt (which is used for back conversion from opticks geometry to geant4)::

     545 void X4PhysicalVolume::convertSolids_r(const G4VPhysicalVolume* const pv, int depth)
     546 {
     547     const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
     548     for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
     549     {
     550         const G4VPhysicalVolume* const daughter_pv = lv->GetDaughter(i);
     551         convertSolids_r( daughter_pv , depth + 1 );
     552     }
     553 
     554     // for newly encountered lv record the tail/postorder idx for the lv
     555     if(std::find(m_lvlist.begin(), m_lvlist.end(), lv) == m_lvlist.end())
     556     {
     557         int lvIdx = m_lvlist.size();
     558         int soIdx = lvIdx ; // when converting in postorder soIdx is the same as lvIdx
     559         m_lvidx[lv] = lvIdx ;
     560         m_lvlist.push_back(lv);
     561 
     562         const G4VSolid* const solid = lv->GetSolid();
     563         const std::string& lvname = lv->GetName() ;
     564 
     565         bool balance_deep_tree = true ;
     566         GMesh* mesh = convertSolid( lvIdx, soIdx, solid, lvname, balance_deep_tree ) ;
     567         mesh->setIndex( lvIdx ) ;
     568 
     569         const NCSG* csg = mesh->getCSG();
     570 
     571         if( csg->is_balanced() )  // when balancing done, also convert without it 
     572         {
     573             balance_deep_tree = false ;
     574             GMesh* rawmesh = convertSolid( lvIdx, soIdx, solid, lvname, balance_deep_tree ) ;
     575             rawmesh->setIndex( lvIdx ) ;
     576 
     577             const NCSG* rawcsg = rawmesh->getCSG();
     578             assert( rawmesh->getIndex() == rawcsg->getIndex() ) ;
     579 
     580             mesh->setAlt(rawmesh);  // <-- this association is preserved (and made symmetric) thru metadata by GMeshLib 
     581         }
     582 
     583         const nnode* root = mesh->getRoot();
     584         assert( root );
     585 
     586         if( root->has_torus() )
     587         {
     588             LOG(fatal) << " has_torus lvIdx " << lvIdx << " " << lvname ;
     589             m_lv_with_torus.push_back( lvIdx );
     590             m_lvname_with_torus.push_back( lvname );
     591         }
     592 
     593         m_ggeo->add( mesh ) ;
     594     }
     595 }




Try to reproduce
----------------------

Change X4GDMLBalanceTest to export unbalanced::

    (gdb) f 11
    #11 0x00007fffeeb88354 in NCSGData::init_buffers (this=0x7f2840, height=253) at /home/blyth/opticks/npy/NCSGData.cpp:94
    94      m_npy->initBuffer( (int)SRC_NODES     ,  m_num_nodes, zero , msg ); 
    (gdb) p m_num_nodes
    $4 = 1073741823
    (gdb) bt
    #0  0x00007ffff586000f in std::__fill_n_a<float*, unsigned long, float> (__first=0x7ff304492000, __n=17179869168, __value=@0x7fffffffc13c: 0) at /usr/include/c++/4.8.2/bits/stl_algobase.h:751
    #1  0x00007ffff585d6bb in std::fill_n<float*, unsigned long, float> (__first=0x7fefe0b0f010, __n=17179869168, __value=@0x7fffffffc13c: 0) at /usr/include/c++/4.8.2/bits/stl_algobase.h:786
    #2  0x00007ffff585cf8a in std::__uninitialized_default_n_1<true>::__uninit_default_n<float*, unsigned long> (__first=0x7fefe0b0f010, __n=17179869168) at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:515
    #3  0x00007ffff58578db in std::__uninitialized_default_n<float*, unsigned long> (__first=0x7fefe0b0f010, __n=17179869168) at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:543
    #4  0x00007ffff584ef12 in std::__uninitialized_default_n_a<float*, unsigned long, float> (__first=0x7fefe0b0f010, __n=17179869168) at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:605
    #5  0x00007ffff583f14e in std::vector<float, std::allocator<float> >::_M_default_append (this=0x7f3e80, __n=17179869168) at /usr/include/c++/4.8.2/bits/vector.tcc:557
    #6  0x00007ffff582c7f7 in std::vector<float, std::allocator<float> >::resize (this=0x7f3e80, __new_size=17179869168) at /usr/include/c++/4.8.2/bits/stl_vector.h:667
    #7  0x00007fffeea49186 in NPY<float>::allocate (this=0x7f3dc0) at /home/blyth/opticks/npy/NPY.cpp:106
    #8  0x00007fffeea49116 in NPY<float>::zero (this=0x7f3dc0) at /home/blyth/opticks/npy/NPY.cpp:95
    #9  0x00007fffeea34f8e in NPYBase::Make (ni=1073741823, itemspec=0x60fe40, zero=true) at /home/blyth/opticks/npy/NPYBase.cpp:117
    #10 0x00007fffeeabcea5 in NPYList::initBuffer (this=0x7f4930, bid=0, ni=1073741823, zero=true, msg=0x7fffeec7d913 "init_buffer.adopt") at /home/blyth/opticks/npy/NPYList.cpp:212
    #11 0x00007fffeeb88354 in NCSGData::init_buffers (this=0x7f2840, height=253) at /home/blyth/opticks/npy/NCSGData.cpp:94
    #12 0x00007fffeeb80989 in NCSG::NCSG (this=0x7f3420, root=0x7f32c0) at /home/blyth/opticks/npy/NCSG.cpp:282
    #13 0x00007fffeeb80347 in NCSG::Adopt (root=0x7f32c0, config=0x0, soIdx=0, lvIdx=0) at /home/blyth/opticks/npy/NCSG.cpp:153
    #14 0x0000000000404b20 in main (argc=1, argv=0x7fffffffd968) at /home/blyth/opticks/extg4/tests/X4GDMLBalanceTest.cc:86
    (gdb) 


Taking ages initing the ginormous buffers::

    (gdb) c
    Continuing.
    ^C
    Program received signal SIGINT, Interrupt.
    0x00007fffeb4cffe0 in __memset_sse2 () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007fffeb4cffe0 in __memset_sse2 () from /lib64/libc.so.6
    #1  0x00007fffeea4913f in NPY<float>::zero (this=0x7f3dc0) at /home/blyth/opticks/npy/NPY.cpp:96
    #2  0x00007fffeea34f8e in NPYBase::Make (ni=1073741823, itemspec=0x60fe40, zero=true) at /home/blyth/opticks/npy/NPYBase.cpp:117
    #3  0x00007fffeeabcea5 in NPYList::initBuffer (this=0x7f4930, bid=0, ni=1073741823, zero=true, msg=0x7fffeec7d913 "init_buffer.adopt") at /home/blyth/opticks/npy/NPYList.cpp:212
    #4  0x00007fffeeb88354 in NCSGData::init_buffers (this=0x7f2840, height=253) at /home/blyth/opticks/npy/NCSGData.cpp:94
    #5  0x00007fffeeb80989 in NCSG::NCSG (this=0x7f3420, root=0x7f32c0) at /home/blyth/opticks/npy/NCSG.cpp:282
    #6  0x00007fffeeb80347 in NCSG::Adopt (root=0x7f32c0, config=0x0, soIdx=0, lvIdx=0) at /home/blyth/opticks/npy/NCSG.cpp:153
    #7  0x0000000000404b20 in main (argc=1, argv=0x7fffffffd968) at /home/blyth/opticks/extg4/tests/X4GDMLBalanceTest.cc:86
    (gdb) 


    (gdb) c
    Continuing.


Then eventually out of memory::

    terminate called after throwing an instance of 'std::bad_alloc'
      what():  std::bad_alloc
    
    Program received signal SIGABRT, Aborted.
    0x00007fffeb477207 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007fffeb477207 in raise () from /lib64/libc.so.6
    #1  0x00007fffeb4788f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffebd867d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007fffebd84746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffebd84773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffebd84993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007fffebd84f2d in operator new(unsigned long) () from /lib64/libstdc++.so.6
    #7  0x00007ffff7484146 in __gnu_cxx::new_allocator<float>::allocate (this=0x7fa190, __n=17179869168) at /usr/include/c++/4.8.2/ext/new_allocator.h:104
    #8  0x00007ffff74816db in std::_Vector_base<float, std::allocator<float> >::_M_allocate (this=0x7fa190, __n=17179869168) at /usr/include/c++/4.8.2/bits/stl_vector.h:168
    #9  0x00007ffff583f0f2 in std::vector<float, std::allocator<float> >::_M_default_append (this=0x7fa190, __n=17179869168) at /usr/include/c++/4.8.2/bits/vector.tcc:549
    #10 0x00007ffff582c7f7 in std::vector<float, std::allocator<float> >::resize (this=0x7fa190, __new_size=17179869168) at /usr/include/c++/4.8.2/bits/stl_vector.h:667
    #11 0x00007fffeea49186 in NPY<float>::allocate (this=0x7fa0d0) at /home/blyth/opticks/npy/NPY.cpp:106
    #12 0x00007fffeea49116 in NPY<float>::zero (this=0x7fa0d0) at /home/blyth/opticks/npy/NPY.cpp:95
    #13 0x00007fffeea34f8e in NPYBase::Make (ni=1073741823, itemspec=0x6118a0, zero=true) at /home/blyth/opticks/npy/NPYBase.cpp:117
    #14 0x00007fffeeabcea5 in NPYList::initBuffer (this=0x7f4930, bid=6, ni=1073741823, zero=true, msg=0x7fffeec7da4c "prepareForExport") at /home/blyth/opticks/npy/NPYList.cpp:212
    #15 0x00007fffeeb88bae in NCSGData::prepareForExport (this=0x7f2840) at /home/blyth/opticks/npy/NCSGData.cpp:203
    #16 0x00007fffeeb83009 in NCSG::export_ (this=0x7f3420) at /home/blyth/opticks/npy/NCSG.cpp:850
    #17 0x00007fffeeb80412 in NCSG::postchange (this=0x7f3420) at /home/blyth/opticks/npy/NCSG.cpp:181
    #18 0x00007fffeeb803a9 in NCSG::Adopt (root=0x7f32c0, config=0x0, soIdx=0, lvIdx=0) at /home/blyth/opticks/npy/NCSG.cpp:160
    #19 0x0000000000404b20 in main (argc=1, argv=0x7fffffffd968) at /home/blyth/opticks/extg4/tests/X4GDMLBalanceTest.cc:86
    (gdb) 

    (gdb) f 11
    #11 0x00007fffeea49186 in NPY<float>::allocate (this=0x7fa0d0) at /home/blyth/opticks/npy/NPY.cpp:106
    106     m_data.resize(getNumValues(0));
    (gdb) p getNumValues(0)
    $5 = 17179869168
    (gdb) 




::

    In [21]:  (1 << 34) - 1
    Out[21]: 17179869183





Investigation
----------------

* Added GDML snippet reading capabaility to X4GDMLParser

::

   [blyth@localhost extg4]$ X4GDMLParserTest > /tmp/out

    ## tree is very big, so using nowrap in vim is handy 


::

    Hi Sam,

    I had a look at x375  with X4GDMLParserTest. x375 is a height 253 tree with 507 nodes. 
    I hope this is not an important piece of geometry for your photons because even if I
    succeed to convert it to a balanced GPU tree I expect the performance will be terrible.
    I expect your Geant4 performance with this will be really terrible also. 

       2019-09-26 21:30:38.255 INFO  [182477] [X4SolidStore::Dump@49] NTreeAnalyse height 253 count 507

    di : differernce
    cy : cylinder

                                                                                                                di
     599
     600                                                                                               di          cy
     601
     602                                                                                       di          cy
     603
     604                                                                               di          cy
     605
     606                                                                       di          cy
     607
     608                                                               di          cy
     609
     610                                                       di          cy
     611
     612                                               di          cy
     613
     614                                       di          cy

    ~ 500 lines like this

    1038                                                                               di          cy
    1039
    1040                                                                       di          cy
    1041
    1042                                                               di          cy
    1043
    1044                                                       di          cy
    1045
    1046                                               di          cy
    1047
    1048                                       di          cy
    1049
    1050                               di          cy
    1051
    1052                       di          cy
    1053
    1054               di          cy
    1055
    1056       di          cy
    1057
    1058   cy      cy
    1059


    Regards the number of solids, I mean how many that are used in logical volumes.
    Most of the many thousands of solids are just constituents of booleans, its the number
    of roots of the trees that matters. 

    Simon



Sam asked why such a big tree ?, I asked why only 1 Billion ? Answer it cycled 32bit unsigned int several times
------------------------------------------------------------------------------------------------------------------


So what to do:

* need to disallow exports of trees of heights exceeding say height 16 



::

     254 NCSG::NCSG(nnode* root )
     255     :
     256     m_treedir(NULL),
     257     m_index(0),
     258     m_surface_epsilon(SURFACE_EPSILON),
     259     m_verbosity(root->verbosity),
     260     m_usedglobally(true),   // changed to true : June 2018, see notes/issues/subtree_instances_missing_transform.rst
     261     m_root(root),
     262     m_points(NULL),
     263     m_uncoincide(make_uncoincide()),
     264     m_nudger(make_nudger("Adopt root ctor")),
     265     m_csgdata(new NCSGData),
     266     m_meta(new NPYMeta),
     267     m_adopted(true),
     268     m_boundary(NULL),
     269     m_config(NULL),
     270     m_gpuoffset(0,0,0),
     271     m_proxylv(-1),
     272     m_container(0),
     273     m_containerscale(2.f),
     274     m_containerautosize(-1),
     275     m_tris(NULL),
     276     m_soIdx(0),
     277     m_lvIdx(0),
     278     m_other(NULL)
     279 {
     280     setBoundary( root->boundary );  // boundary spec
     281     LOG(debug) << "[" ;
     282     m_csgdata->init_buffers(root->maxdepth()) ;
     //                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   
     283     LOG(debug) << "]" ;
     284 }
     285 


::

     267 unsigned nnode::maxdepth() const
     268 {
     269     return _maxdepth(0);
     270 }
     271 unsigned nnode::_maxdepth(unsigned depth) const   // recursive 
     272 {
     273     return left && right ? nmaxu( left->_maxdepth(depth+1), right->_maxdepth(depth+1)) : depth ;
     274 }



::

    085 void NCSGData::init_buffers(unsigned height)  // invoked from NCSG::NCSG(nnode* root ) ie when adopting 
     86 {
     87     m_height = height ;
     88     unsigned num_nodes = NumNodes(height); // number of nodes for a complete binary tree of the needed height, with no balancing 
     89     m_num_nodes = num_nodes ;
     90 
     91     bool zero = true ;
     92     const char* msg = "init_buffer.adopt" ;
     93 
     94     m_npy->initBuffer( (int)SRC_NODES     ,  m_num_nodes, zero , msg );
     95     m_npy->initBuffer( (int)SRC_TRANSFORMS,            0, zero , msg );
     96     m_npy->initBuffer( (int)SRC_PLANES    ,            0, zero , msg );
     97     m_npy->initBuffer( (int)SRC_IDX       ,            1, zero , msg );
     98 
     99     m_npy->initBuffer( (int)SRC_VERTS     ,            0, zero , msg );
    100     m_npy->initBuffer( (int)SRC_FACES     ,            0, zero , msg );
    101 }

::

    039 #define TREE_NODES(height) ( (0x1 << (1+(height))) - 1 )


    360 unsigned NCSGData::NumNodes(unsigned height)
    361 {
    362    return TREE_NODES(height);
    363 }



::

    [blyth@localhost npy]$ NCSGDataTest 
     h          0 NumNodes                    1
     h          1 NumNodes                    3
     h          2 NumNodes                    7
     h          3 NumNodes                   15
     h          4 NumNodes                   31
     h          5 NumNodes                   63
     h          6 NumNodes                  127
     h          7 NumNodes                  255
     h          8 NumNodes                  511
     h          9 NumNodes                 1023
     h         10 NumNodes                 2047
     h         11 NumNodes                 4095
     h         12 NumNodes                 8191
     h         13 NumNodes                16383
     h         14 NumNodes                32767
     h         15 NumNodes                65535
     h         16 NumNodes               131071
     h         17 NumNodes               262143
     h         18 NumNodes               524287
     h         19 NumNodes              1048575
     h         20 NumNodes              2097151
     h         21 NumNodes              4194303
     h         22 NumNodes              8388607
     h         23 NumNodes             16777215
     h         24 NumNodes             33554431
     h         25 NumNodes             67108863
     h         26 NumNodes            134217727
     h         27 NumNodes            268435455
     h         28 NumNodes            536870911
     h         29 NumNodes           1073741823
     h         30 NumNodes           2147483647
     h         31 NumNodes                    0
     h         32 NumNodes                    1
     h         33 NumNodes                    3
     h         34 NumNodes                    7
     h         35 NumNodes                   15
     h         36 NumNodes                   31
     h         37 NumNodes                   63
     h         38 NumNodes                  127
     h         39 NumNodes                  255
     h         40 NumNodes                  511
     h         41 NumNodes                 1023
     h         42 NumNodes                 2047
     h         43 NumNodes                 4095
     h         44 NumNodes                 8191
     h         45 NumNodes                16383
     h         46 NumNodes                32767
     h         47 NumNodes                65535
     h         48 NumNodes               131071
     h         49 NumNodes               262143
     h         50 NumNodes               524287
     h         51 NumNodes              1048575
     h         52 NumNodes              2097151
     h         53 NumNodes              4194303
     h         54 NumNodes              8388607
     h         55 NumNodes             16777215
     h         56 NumNodes             33554431
     h         57 NumNodes             67108863
     h         58 NumNodes            134217727
     h         59 NumNodes            268435455
     h         60 NumNodes            536870911
     h         61 NumNodes           1073741823
     h         62 NumNodes           2147483647
     h         63 NumNodes                    0
     h         64 NumNodes                    1
     h         65 NumNodes                    3
     h         66 NumNodes                    7
     h         67 NumNodes                   15
     h         68 NumNodes                   31
     h         69 NumNodes                   63
     h         70 NumNodes                  127
     h         71 NumNodes                  255
     h         72 NumNodes                  511
     h         73 NumNodes                 1023
     h         74 NumNodes                 2047
     h         75 NumNodes                 4095
     h         76 NumNodes                 8191
     h         77 NumNodes                16383
     h         78 NumNodes                32767
     h         79 NumNodes                65535
     h         80 NumNodes               131071
     h         81 NumNodes               262143
     h         82 NumNodes               524287
     h         83 NumNodes              1048575
     h         84 NumNodes              2097151
     h         85 NumNodes              4194303
     h         86 NumNodes              8388607
     h         87 NumNodes             16777215
     h         88 NumNodes             33554431
     h         89 NumNodes             67108863
     h         90 NumNodes            134217727
     h         91 NumNodes            268435455
     h         92 NumNodes            536870911
     h         93 NumNodes           1073741823
     h         94 NumNodes           2147483647
     h         95 NumNodes                    0
     h         96 NumNodes                    1
     h         97 NumNodes                    3
     h         98 NumNodes                    7
     h         99 NumNodes                   15
     h        100 NumNodes                   31
     h        101 NumNodes                   63
     h        102 NumNodes                  127
     h        103 NumNodes                  255
     h        104 NumNodes                  511
     h        105 NumNodes                 1023
     h        106 NumNodes                 2047
     h        107 NumNodes                 4095
     h        108 NumNodes                 8191
     h        109 NumNodes                16383
     h        110 NumNodes                32767
     h        111 NumNodes                65535
     h        112 NumNodes               131071
     h        113 NumNodes               262143
     h        114 NumNodes               524287
     h        115 NumNodes              1048575
     h        116 NumNodes              2097151
     h        117 NumNodes              4194303
     h        118 NumNodes              8388607
     h        119 NumNodes             16777215
     h        120 NumNodes             33554431
     h        121 NumNodes             67108863
     h        122 NumNodes            134217727
     h        123 NumNodes            268435455
     h        124 NumNodes            536870911
     h        125 NumNodes           1073741823
     h        126 NumNodes           2147483647
     h        127 NumNodes                    0
     h        128 NumNodes                    1
     h        129 NumNodes                    3
     h        130 NumNodes                    7
     h        131 NumNodes                   15
     h        132 NumNodes                   31
     h        133 NumNodes                   63
     h        134 NumNodes                  127
     h        135 NumNodes                  255
     h        136 NumNodes                  511
     h        137 NumNodes                 1023
     h        138 NumNodes                 2047
     h        139 NumNodes                 4095
     h        140 NumNodes                 8191
     h        141 NumNodes                16383
     h        142 NumNodes                32767
     h        143 NumNodes                65535
     h        144 NumNodes               131071
     h        145 NumNodes               262143
     h        146 NumNodes               524287
     h        147 NumNodes              1048575
     h        148 NumNodes              2097151
     h        149 NumNodes              4194303
     h        150 NumNodes              8388607
     h        151 NumNodes             16777215
     h        152 NumNodes             33554431
     h        153 NumNodes             67108863
     h        154 NumNodes            134217727
     h        155 NumNodes            268435455
     h        156 NumNodes            536870911
     h        157 NumNodes           1073741823
     h        158 NumNodes           2147483647
     h        159 NumNodes                    0
     h        160 NumNodes                    1
     h        161 NumNodes                    3
     h        162 NumNodes                    7
     h        163 NumNodes                   15
     h        164 NumNodes                   31
     h        165 NumNodes                   63
     h        166 NumNodes                  127
     h        167 NumNodes                  255
     h        168 NumNodes                  511
     h        169 NumNodes                 1023
     h        170 NumNodes                 2047
     h        171 NumNodes                 4095
     h        172 NumNodes                 8191
     h        173 NumNodes                16383
     h        174 NumNodes                32767
     h        175 NumNodes                65535
     h        176 NumNodes               131071
     h        177 NumNodes               262143
     h        178 NumNodes               524287
     h        179 NumNodes              1048575
     h        180 NumNodes              2097151
     h        181 NumNodes              4194303
     h        182 NumNodes              8388607
     h        183 NumNodes             16777215
     h        184 NumNodes             33554431
     h        185 NumNodes             67108863
     h        186 NumNodes            134217727
     h        187 NumNodes            268435455
     h        188 NumNodes            536870911
     h        189 NumNodes           1073741823
     h        190 NumNodes           2147483647
     h        191 NumNodes                    0
     h        192 NumNodes                    1
     h        193 NumNodes                    3
     h        194 NumNodes                    7
     h        195 NumNodes                   15
     h        196 NumNodes                   31
     h        197 NumNodes                   63
     h        198 NumNodes                  127
     h        199 NumNodes                  255
     h        200 NumNodes                  511
     h        201 NumNodes                 1023
     h        202 NumNodes                 2047
     h        203 NumNodes                 4095
     h        204 NumNodes                 8191
     h        205 NumNodes                16383
     h        206 NumNodes                32767
     h        207 NumNodes                65535
     h        208 NumNodes               131071
     h        209 NumNodes               262143
     h        210 NumNodes               524287
     h        211 NumNodes              1048575
     h        212 NumNodes              2097151
     h        213 NumNodes              4194303
     h        214 NumNodes              8388607
     h        215 NumNodes             16777215
     h        216 NumNodes             33554431
     h        217 NumNodes             67108863
     h        218 NumNodes            134217727
     h        219 NumNodes            268435455
     h        220 NumNodes            536870911
     h        221 NumNodes           1073741823
     h        222 NumNodes           2147483647
     h        223 NumNodes                    0
     h        224 NumNodes                    1
     h        225 NumNodes                    3
     h        226 NumNodes                    7
     h        227 NumNodes                   15
     h        228 NumNodes                   31
     h        229 NumNodes                   63
     h        230 NumNodes                  127
     h        231 NumNodes                  255
     h        232 NumNodes                  511
     h        233 NumNodes                 1023
     h        234 NumNodes                 2047
     h        235 NumNodes                 4095
     h        236 NumNodes                 8191
     h        237 NumNodes                16383
     h        238 NumNodes                32767
     h        239 NumNodes                65535
     h        240 NumNodes               131071
     h        241 NumNodes               262143
     h        242 NumNodes               524287
     h        243 NumNodes              1048575
     h        244 NumNodes              2097151
     h        245 NumNodes              4194303
     h        246 NumNodes              8388607
     h        247 NumNodes             16777215
     h        248 NumNodes             33554431
     h        249 NumNodes             67108863
     h        250 NumNodes            134217727
     h        251 NumNodes            268435455
     h        252 NumNodes            536870911
     h        253 NumNodes           1073741823
     h        254 NumNodes           2147483647
     h        255 NumNodes                    0
     h        256 NumNodes                    1
     h        257 NumNodes                    3
     h        258 NumNodes                    7
     h        259 NumNodes                   15






