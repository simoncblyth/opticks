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




