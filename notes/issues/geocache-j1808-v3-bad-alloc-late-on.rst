geocache-j1808-v3-bad-alloc-late-on
====================================


Workaround : copy over from workstation
------------------------------------------

::

    [blyth@localhost~ ]$ geocache-cd
    [blyth@localhost geocache]$ scp -r OKX4Test_lWorld0x4bc2710_PV_g4live L7:g/local/opticks/geocache/


:google:`linux profile virtual memory from within process`
--------------------------------------------------------------

* https://stackoverflow.com/search?q=reading+%2Fproc%2Fself%2Fstatus+

Reading /proc/self/status

* https://github.com/TysonRayJones/CTools/blob/master/memory/memorymeasure.c 

See:

* sysrap/SProc.cc 
* sysrap/tests/SProcTest.cc 



Issue : geocache creation runs of of memory on lxslc L7
----------------------------------------------------------


::

    2019-04-28 22:34:29.572 INFO  [19731] [GGeo::prepare@681] prepareVolumes
    2019-04-28 22:34:43.793 INFO  [19731] [GInstancer::findRepeatCandidates@283]  nall 40 repeat_min 400 vertex_min 0 num_repcan 5
    2019-04-28 22:34:43.794 INFO  [19731] [GInstancer::findRepeatCandidates@293]  num_repcan 5 dmax 20
     (**) candidates fulfil repeat/vert cuts   
     (##) selected survive contained-repeat disqualification 
     **  ##  idx   0 pdig 6d5590df6216ccec831fe5d8d53bacf2 ndig  36572 nprog      4 nvert    805 n PMT_3inch_log_phys0x510ddb0
     **      idx   1 pdig 4e732ced3463d06de0ca9a15b6153677 ndig  36572 nprog      0 nvert     50 n PMT_3inch_cntr_phys0x510c010
     **      idx   2 pdig 24280eae25dd029943692e7b55f69cec ndig  36572 nprog      2 nvert    489 n PMT_3inch_body_phys0x510be30
     **      idx   3 pdig 37693cfc748049e45d87b8c7d8b9aacd ndig  36572 nprog      0 nvert    123 n PMT_3inch_inner1_phys0x510beb0
     **      idx   4 pdig 1ff1de774005f8da13f42943881c655f ndig  36572 nprog      0 nvert    147 n PMT_3inch_inner2_phys0x510bf60
     **      idx   5 pdig a87ff679a2f3e71d9181a67b7542122c ndig  30720 nprog      0 nvert      8 n pBar0x5b3a400
     **      idx   6 pdig 4a891f9f9571ecb440f34c6f1d7a0ed1 ndig  30720 nprog      1 nvert     16 n pCoating00x5b37960
     **      idx   7 pdig 5d176f7783ecaca7f7b787b393ed52d6 ndig  20046 nprog      3 nvert   1353 n PMT_20inch_log_phys0x4ca16b0
     **      idx   8 pdig 70efdf2ec9b086079795c442636b55fb ndig  20046 nprog      0 nvert    484 n pMask0x4c3bf20
     **  ##  idx   9 pdig 91bdcb2ca344aad4e19053c9ac2e756e ndig  20046 nprog      5 nvert   1887 n lMaskVirtual_phys0x4c9a510
     **      idx  10 pdig 35eb2482ddcb49a50b1a085175fe3bf1 ndig  20046 nprog      2 nvert    919 n PMT_20inch_body_phys0x4c9a7f0
     **      idx  11 pdig 6f4922f45568161a8cdf4ad2299f6d23 ndig  20046 nprog      0 nvert    171 n PMT_20inch_inner1_phys0x4c9a870
     **      idx  12 pdig 1f0e3dad99908345f7439f8ffabdffc4 ndig  20046 nprog      0 nvert    314 n PMT_20inch_inner2_phys0x4c9a920
     **      idx  13 pdig 26bb84e49645032e36b4255eca5f8baa ndig    480 nprog    128 nvert   1032 n pModuleTape0x5b378c0
     **  ##  idx  14 pdig 36986ef0e48b2696d09c93846c6b69d0 ndig    480 nprog    129 nvert   1040 n pModule00x5b37610
     **  ##  idx  15 pdig 9bf31c7ff062936a96d3c8bd1f8f2ff3 ndig    480 nprog      0 nvert     96 n lSteel_phys0x4bd4d60
     **  ##  idx  16 pdig c74d97b01eae257e44aa9d5bade97baf ndig    480 nprog      0 nvert    914 n lFasteners_phys0x4c01450
             idx  17 pdig e9cb249cd3ac991974a004095df669bf ndig    120 nprog    520 nvert   4168 n pPlane00x5b37480
             idx  18 pdig babb1e17850363013d520fb097b7ebc0 ndig     60 nprog   1042 nvert   8344 n pWall00x5b34c40
             idx  19 pdig e920f80b0f270231875a7d7f69b3d37b ndig      1 nprog      3 nvert    292 n lLowerChimney_phys0x5b32c20
    2019-04-28 22:34:43.794 INFO  [19731] [GInstancer::dumpRepeatCandidates@353]  num_repcan 5 dmax 20
     pdig 6d5590df6216ccec831fe5d8d53bacf2 ndig  36572 nprog      4 placements  36572 n PMT_3inch_log_phys0x510ddb0
     pdig 91bdcb2ca344aad4e19053c9ac2e756e ndig  20046 nprog      5 placements  20046 n lMaskVirtual_phys0x4c9a510
     pdig 36986ef0e48b2696d09c93846c6b69d0 ndig    480 nprog    129 placements    480 n pModule00x5b37610
     pdig 9bf31c7ff062936a96d3c8bd1f8f2ff3 ndig    480 nprog      0 placements    480 n lSteel_phys0x4bd4d60
     pdig c74d97b01eae257e44aa9d5bade97baf ndig    480 nprog      0 placements    480 n lFasteners_phys0x4c01450
    2019-04-28 22:34:44.667 FATAL [19731] [GInstancer::labelTree@432]  m_labels (count of non-zero setRepeatIndex) 366496 m_csgskiplv_count 20046 m_repeats_count 366496 m_globals_count 201 total_count : 366697
    terminate called after throwing an instance of 'std::bad_alloc'
      what():  std::bad_alloc

    Program received signal SIGABRT, Aborted.
    0x00007fffe207b207 in raise () from /usr/lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-260.el7.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-34.el7.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXi-1.7.9-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-12.el7_5.x86_64 libcurl-7.29.0-51.el7.x86_64 libgcc-4.8.5-28.el7_5.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-15.el7.x86_64 libidn-1.28-4.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.4.3-12.el7_6.2.x86_64 libstdc++-4.8.5-28.el7_5.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.el7_5.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.el7_5.x86_64 openldap-2.4.44-15.el7_5.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-17.el7.x86_64
    (gdb) 
    (gdb) bt
    #0  0x00007fffe207b207 in raise () from /usr/lib64/libc.so.6
    #1  0x00007fffe207c8f8 in abort () from /usr/lib64/libc.so.6
    #2  0x00007fffe298a7d5 in __gnu_cxx::__verbose_terminate_handler() () from /usr/lib64/libstdc++.so.6
    #3  0x00007fffe2988746 in ?? () from /usr/lib64/libstdc++.so.6
    #4  0x00007fffe2988773 in std::terminate() () from /usr/lib64/libstdc++.so.6
    #5  0x00007fffe2988993 in __cxa_throw () from /usr/lib64/libstdc++.so.6
    #6  0x00007fffe2988f2d in operator new(unsigned long) () from /usr/lib64/libstdc++.so.6
    #7  0x00007fffe5cfeab3 in __gnu_cxx::new_allocator<glm::vec<3, float, (glm::qualifier)0> >::allocate (this=0x7fffffff9230, __n=64) at /usr/include/c++/4.8.2/ext/new_allocator.h:104
    #8  0x00007fffe5cfe59d in std::_Vector_base<glm::vec<3, float, (glm::qualifier)0>, std::allocator<glm::vec<3, float, (glm::qualifier)0> > >::_M_allocate (this=0x7fffffff9230, __n=64) at /usr/include/c++/4.8.2/bits/stl_vector.h:168
    #9  0x00007fffe5cfdf88 in std::vector<glm::vec<3, float, (glm::qualifier)0>, std::allocator<glm::vec<3, float, (glm::qualifier)0> > >::_M_emplace_back_aux<glm::vec<3, float, (glm::qualifier)0> const&> (this=0x7fffffff9230)
        at /usr/include/c++/4.8.2/bits/vector.tcc:404
    #10 0x00007fffe5cfdc45 in std::vector<glm::vec<3, float, (glm::qualifier)0>, std::allocator<glm::vec<3, float, (glm::qualifier)0> > >::push_back (this=0x7fffffff9230, __x=...) at /usr/include/c++/4.8.2/bits/stl_vector.h:911
    #11 0x00007fffe5cf9619 in GMesh::findBBox (vertices=0xd8503210, num_vertices=484) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMesh.cc:1435
    #12 0x00007fffe5d273bb in GMergedMesh::mergeVolumeBBox (this=0x7fff9e3c4390, vertices=0xd8503210, nvert=484) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:543
    #13 0x00007fffe5d2670c in GMergedMesh::mergeVolume (this=0x7fff9e3c4390, volume=0x1d41cd60, selected=false, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:414
    #14 0x00007fffe5d25df7 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x1d41cd60, depth=7, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:291
    #15 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x1d41b560, depth=6, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    #16 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x13b22ed0, depth=5, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    #17 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x13b1fb10, depth=4, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    #18 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x13b1e420, depth=3, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    #19 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x13b1ce80, depth=2, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    #20 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x13b1b9a0, depth=1, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    #21 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x3bbd440, depth=0, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    #22 0x00007fffe5d258f7 in GMergedMesh::create (ridx=0, base=0x0, root=0x3bbd440, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:238
    #23 0x00007fffe5d07df5 in GGeoLib::makeMergedMesh (this=0x37786a0, index=0, base=0x0, root=0x3bbd440, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GGeoLib.cc:264
    #24 0x00007fffe5d1b74e in GInstancer::makeMergedMeshAndInstancedBuffers (this=0x37838a0, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GInstancer.cc:550
    #25 0x00007fffe5d196a0 in GInstancer::createInstancedMergedMeshes (this=0x37838a0, delta=true, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GInstancer.cc:102
    #26 0x00007fffe5d31b58 in GGeo::prepareVolumes (this=0x377b000) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GGeo.cc:1274
    #27 0x00007fffe5d2ef56 in GGeo::prepare (this=0x377b000) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GGeo.cc:682
    #28 0x0000000000404fab in main (argc=5, argv=0x7fffffffd768) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/okg4/tests/OKX4Test.cc:119
    (gdb) 
    
   (gdb) f 28
    #28 0x0000000000404fab in main (argc=5, argv=0x7fffffffd768) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/okg4/tests/OKX4Test.cc:119
    119     gg2->prepare();   // merging meshes, closing libs
    (gdb) f 27
    #27 0x00007fffe5d2ef56 in GGeo::prepare (this=0x377b000) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GGeo.cc:682
    warning: Source file is more recent than executable.
    682     LOG(info) << "prepareVolumes" ;  
    (gdb) f 26
    #26 0x00007fffe5d31b58 in GGeo::prepareVolumes (this=0x377b000) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GGeo.cc:1274
    1274                  ;
    (gdb) f 25
    #25 0x00007fffe5d196a0 in GInstancer::createInstancedMergedMeshes (this=0x37838a0, delta=true, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GInstancer.cc:102
    102     makeMergedMeshAndInstancedBuffers(verbosity);
    (gdb) f 24
    #24 0x00007fffe5d1b74e in GInstancer::makeMergedMeshAndInstancedBuffers (this=0x37838a0, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GInstancer.cc:550
    550     GMergedMesh* mm0 = m_geolib->makeMergedMesh(0, base, root, verbosity );
    (gdb) f 23
    #23 0x00007fffe5d07df5 in GGeoLib::makeMergedMesh (this=0x37786a0, index=0, base=0x0, root=0x3bbd440, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GGeoLib.cc:264
    264         m_merged_mesh[index] = GMergedMesh::create(index, base, root, verbosity );
    (gdb) p index
    $1 = 0
    (gdb) f 22
    #22 0x00007fffe5d258f7 in GMergedMesh::create (ridx=0, base=0x0, root=0x3bbd440, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:238
    238     mm->traverse_r( start, 0, PASS_MERGE, verbosity );  
    (gdb) f 21
    #21 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x3bbd440, depth=0, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    295     for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1, pass, verbosity );
    (gdb) f 20
    #20 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x13b1b9a0, depth=1, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    295     for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1, pass, verbosity );
    (gdb) f 19
    #19 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x13b1ce80, depth=2, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    295     for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1, pass, verbosity );
    (gdb) f 18
    #18 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x13b1e420, depth=3, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    295     for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1, pass, verbosity );
    (gdb) f 17
    #17 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x13b1fb10, depth=4, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    295     for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1, pass, verbosity );
    (gdb) f 16
    #16 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x13b22ed0, depth=5, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    295     for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1, pass, verbosity );
    (gdb) f 15
    #15 0x00007fffe5d25e66 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x1d41b560, depth=6, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:295
    295     for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1, pass, verbosity );
    (gdb) f 14
    #14 0x00007fffe5d25df7 in GMergedMesh::traverse_r (this=0x7fff9e3c4390, node=0x1d41cd60, depth=7, pass=1, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:291
    291        case PASS_MERGE:    mergeVolume(volume, selected, verbosity)  ;break;
    (gdb) p volume
    $2 = (GVolume .) 0x1d41cd60
    (gdb) p .volume
    $3 = {<GNode> = {_vptr.GNode = 0x7fffe5fb6150 <vtable for GVolume+16>, m_selfdigest = true, m_selected = true, m_index = 75370, m_parent = 0x1d41b560, m_children = std::vector of length 0, capacity 0, m_description = 0x0, 
        m_transform = 0x1d41cce0, m_ltransform = 0x1d41cad0, m_gtriple = 0x1d41cc10, m_ltriple = 0x1d41ca00, m_mesh = 0x397f700, m_low = 0x1d41cb50, m_high = 0x1d41cea0, m_boundary_indices = 0x1d41ece0, m_sensor_indices = 0x1d41ddd0, 
        m_node_indices = 0x1d41cec0, m_name = 0x1d41cbd0 "pMask0x4c3bf20", m_local_digest = "", m_progeny_digest = "70efdf2ec9b086079795c442636b55fb", m_progeny = std::vector of length 0, capacity 0, 
        m_ancestors = std::vector of length 7, capacity 8 = {0x3bbd440, 0x13b1b9a0, 0x13b1ce80, 0x13b1e420, 0x13b1fb10, 0x13b22ed0, 0x1d41b560}, m_progeny_count = 0, m_repeat_index = 2, m_progeny_num_vertices = 0, 
        m_distinct_boundary_indices = std::vector of length 0, capacity 0}, m_boundary = 15, m_csgflag = CSG_PARTLIST, m_csgskip = false, m_sensor = 0x0, m_pvname = 0x1d41cb90 "pMask0x4c3bf20", m_lvname = 0x1d41cbb0 "lMask0x4ca3960", 
      m_sensor_surface_index = 0, m_parts = 0x1d41c910, m_parallel_node = 0x1d41c760}
    (gdb) 
     


Rerunning same executable, the issue repeats at the same volume::

    (gdb) p .volume
    $1 = {<GNode> = {_vptr.GNode = 0x7fffe5fb6150 <vtable for GVolume+16>, m_selfdigest = true, m_selected = true, m_index = 75370, m_parent = 0x1d41b560, m_children = std::vector of length 0, capacity 0, m_description = 0x0, 
        m_transform = 0x1d41cce0, m_ltransform = 0x1d41cad0, m_gtriple = 0x1d41cc10, m_ltriple = 0x1d41ca00, m_mesh = 0x397f700, m_low = 0x1d41cb50, m_high = 0x1d41cea0, m_boundary_indices = 0x1d41ece0, m_sensor_indices = 0x1d41ddd0, 
        m_node_indices = 0x1d41cec0, m_name = 0x1d41cbd0 "pMask0x4c3bf20", m_local_digest = "", m_progeny_digest = "70efdf2ec9b086079795c442636b55fb", m_progeny = std::vector of length 0, capacity 0, 
        m_ancestors = std::vector of length 7, capacity 8 = {0x3bbd440, 0x13b1b9a0, 0x13b1ce80, 0x13b1e420, 0x13b1fb10, 0x13b22ed0, 0x1d41b560}, m_progeny_count = 0, m_repeat_index = 2, m_progeny_num_vertices = 0, 
        m_distinct_boundary_indices = std::vector of length 0, capacity 0}, m_boundary = 15, m_csgflag = CSG_PARTLIST, m_csgskip = false, m_sensor = 0x0, m_pvname = 0x1d41cb90 "pMask0x4c3bf20", m_lvname = 0x1d41cbb0 "lMask0x4ca3960", 
      m_sensor_surface_index = 0, m_parts = 0x1d41c910, m_parallel_node = 0x1d41c760}
    (gdb) 



Rerunning with changed executable, removing the gltf saving to save some memory, the issue happens again but at a later volume::

    (gdb) f 13
    #13 0x00007fffe5d2670c in GMergedMesh::mergeVolume (this=0xe4778340, volume=0x4b9a83a0, selected=false, verbosity=0) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GMergedMesh.cc:414
    414     mergeVolumeBBox(vertices, num_vert);
    (gdb) p *volume
    $1 = {<GNode> = {_vptr.GNode = 0x7fffe5fb6150 <vtable for GVolume+16>, m_selfdigest = true, m_selected = true, m_index = 138969, m_parent = 0x13b22ed0, m_children = std::vector of length 2, capacity 2 = {0x4b9a9bb0, 0x4b9adb60}, 
        m_description = 0x0, m_transform = 0x4b9a8320, m_ltransform = 0x4b9a8120, m_gtriple = 0x4b9a8250, m_ltriple = 0x4b9a8050, m_mesh = 0x3a6afe0, m_low = 0x4b9a81e0, m_high = 0x4b9a84e0, m_boundary_indices = 0x4b9a8820, 
        m_sensor_indices = 0x4b9a8690, m_node_indices = 0x4b9a8500, m_name = 0x4b9a89d0 "lMaskVirtual_phys0x4fa2bb0", m_local_digest = "", m_progeny_digest = "91bdcb2ca344aad4e19053c9ac2e756e", 
        m_progeny = std::vector of length 5, capacity 8 = {0x4b9a9bb0, 0x4b9adb60, 0x4b9b16c0, 0x4b9b5110, 0x4b9b74c0}, m_ancestors = std::vector of length 6, capacity 8 = {0x3bbd440, 0x13b1b9a0, 0x13b1ce80, 0x13b1e420, 0x13b1fb10, 
          0x13b22ed0}, m_progeny_count = 5, m_repeat_index = 2, m_progeny_num_vertices = 0, m_distinct_boundary_indices = std::vector of length 0, capacity 0}, m_boundary = 19, m_csgflag = CSG_PARTLIST, m_csgskip = true, m_sensor = 0x0, 
      m_pvname = 0x4b9a8220 "lMaskVirtual_phys0x4fa2bb0", m_lvname = 0x4b9a89b0 "lMaskVirtual0x4c803b0", m_sensor_surface_index = 0, m_parts = 0x4b9a7f60, m_parallel_node = 0x4b9a81c0}
    (gdb) 






::

    1423 gbbox* GMesh::findBBox(gfloat3* vertices, unsigned int num_vertices)
    1424 {
    1425     if(num_vertices == 0) return NULL ;
    1426 
    1427 
    1428 
    1429     std::vector<glm::vec3> points ;
    1430 
    1431     for( unsigned int i = 0; i < num_vertices ;++i )
    1432     {
    1433         gfloat3& v = vertices[i];
    1434         glm::vec3 p(v.x,v.y,v.z);
    1435         points.push_back(p);
    1436     }
    1437 
    1438     unsigned verbosity = 0 ;
    1439     nbbox nbb = nbbox::from_points(points, verbosity);
    1440     gbbox* bb = new gbbox(nbb);
    1441 
    1442 /*
    1443     gbbox* bb = new gbbox (gfloat3(FLT_MAX), gfloat3(-FLT_MAX)) ; 
    1444     for( unsigned int i = 0; i < num_vertices ;++i )
    1445     {
    1446         gfloat3& v = vertices[i];
    1447 
    1448         bb->min.x = std::min( bb->min.x, v.x);
    1449         bb->min.y = std::min( bb->min.y, v.y);
    1450         bb->min.z = std::min( bb->min.z, v.z);
    1451 
    1452         bb->max.x = std::max( bb->max.x, v.x);
    1453         bb->max.y = std::max( bb->max.y, v.y);
    1454         bb->max.z = std::max( bb->max.z, v.z);
    1455     }
    1456 */
    1457 
    1458     return bb ;
    1459 }

