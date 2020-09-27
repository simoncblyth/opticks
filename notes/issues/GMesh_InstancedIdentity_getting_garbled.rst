GMesh_InstancedIdentity_getting_garbled
========================================


::

    epsilon:ggeo blyth$ GMergedMesh=INFO GGeoTest 
    PLOG::EnvLevel adjusting loglevel by envvar   key GMergedMesh level INFO fallback DEBUG
    2020-09-27 15:02:13.317 INFO  [576837] [BOpticksKey::SetKey@75]  spec OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
    2020-09-27 15:02:13.320 INFO  [576837] [Opticks::init@405] INTEROP_MODE hostname epsilon.local
    2020-09-27 15:02:13.320 INFO  [576837] [Opticks::init@414]  non-legacy mode : ie mandatory keyed access to geometry, opticksaux 
    2020-09-27 15:02:13.323 INFO  [576837] [BOpticksResource::setupViaKey@832] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
                     exename  : OKX4Test
             current_exename  : GGeoTest
                       class  : X4PhysicalVolume
                     volname  : World0xc15cfc00x40f7000_PV
                      digest  : 50a18baaf29b18fae8c1642927003ee3
                      idname  : OKX4Test_World0xc15cfc00x40f7000_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-09-27 15:02:13.324 INFO  [576837] [Opticks::loadOriginCacheMeta@1849]  cachemetapath /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/cachemeta.json
    2020-09-27 15:02:13.324 INFO  [576837] [NMeta::dump@199] Opticks::loadOriginCacheMeta
    {
        "argline": "/usr/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v0.gdml --x4polyskip 211,232 --geocenter --noviz --runfolder geocache-dx-v0 --runcomment export-dyb-gdml-from-g4-10-4-2-to-support-geocache-creation.rst ",
        "location": "Opticks::updateCacheMeta",
        "runcomment": "export-dyb-gdml-from-g4-10-4-2-to-support-geocache-creation.rst",
        "rundate": "20200926_190048",
        "runfolder": "geocache-dx-v0",
        "runlabel": "R0_cvd_",
        "runstamp": 1601143248
    }
    2020-09-27 15:02:13.324 INFO  [576837] [Opticks::loadOriginCacheMeta@1853]  gdmlpath /usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v0.gdml
    2020-09-27 15:02:13.330 INFO  [576837] [*GMergedMesh::Load@983]  dir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/0 -> cachedir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/0 index 0 version (null) existsdir 1
    2020-09-27 15:02:13.395 INFO  [576837] [GMesh::loadNPYBuffer@1854]  loading iidentity /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/0/iidentity.npy
    2020-09-27 15:02:13.432 INFO  [576837] [*GMergedMesh::Load@983]  dir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/1 -> cachedir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/1 index 1 version (null) existsdir 1
    2020-09-27 15:02:13.433 INFO  [576837] [GMesh::loadNPYBuffer@1854]  loading iidentity /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/1/iidentity.npy
    2020-09-27 15:02:13.434 INFO  [576837] [*GMergedMesh::Load@983]  dir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/2 -> cachedir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/2 index 2 version (null) existsdir 1
    2020-09-27 15:02:13.435 INFO  [576837] [GMesh::loadNPYBuffer@1854]  loading iidentity /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/2/iidentity.npy
    2020-09-27 15:02:13.435 INFO  [576837] [*GMergedMesh::Load@983]  dir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/3 -> cachedir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/3 index 3 version (null) existsdir 1
    2020-09-27 15:02:13.436 INFO  [576837] [GMesh::loadNPYBuffer@1854]  loading iidentity /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/3/iidentity.npy
    2020-09-27 15:02:13.436 INFO  [576837] [*GMergedMesh::Load@983]  dir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/4 -> cachedir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/4 index 4 version (null) existsdir 1
    2020-09-27 15:02:13.437 INFO  [576837] [GMesh::loadNPYBuffer@1854]  loading iidentity /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/4/iidentity.npy
    2020-09-27 15:02:13.437 INFO  [576837] [*GMergedMesh::Load@983]  dir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/5 -> cachedir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/5 index 5 version (null) existsdir 1
    2020-09-27 15:02:13.438 INFO  [576837] [GMesh::loadNPYBuffer@1854]  loading iidentity /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/5/iidentity.npy
    2020-09-27 15:02:13.439 INFO  [576837] [*GMergedMesh::Load@983]  dir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/6 -> cachedir /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/6 index 6 version (null) existsdir 1
    2020-09-27 15:02:13.497 INFO  [576837] [GMesh::loadNPYBuffer@1854]  loading iidentity /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/6/iidentity.npy
    2020-09-27 15:02:16.981 INFO  [576837] [NMeta::dump@199] GGeo::loadCacheMeta.lv2sd
    2020-09-27 15:02:16.981 INFO  [576837] [NMeta::dump@199] GGeo::loadCacheMeta.lv2mt
    GGeo::dumpStats
     mm  0 : vertices  247718 faces  480972 transforms   12230 itransforms       1 
     mm  1 : vertices       8 faces      12 transforms       1 itransforms    1792 
     mm  2 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  3 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  4 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  5 : vertices    1498 faces    2976 transforms       5 itransforms     672 
     mm  6 : vertices  247718 faces  480972 transforms    4486 itransforms       1 
       totVertices    496966  totFaces    964968 
      vtotVertices   1537164 vtotFaces   3014424 (virtual: scaling by transforms)
      vfacVertices     3.093 vfacFaces     3.124 (virtual to total ratio)
    2020-09-27 15:02:16.982 INFO  [576837] [test_GGeo_identity@74]  numVolumes 12230 edgeitems 20 modulo 500
     NodeInfo  nface     12 nvert      8 node      0 parent      0 Identity  (      0   248     0     0 )  InstancedIdentity  (          0        248          0          0 ) 
     NodeInfo  nface     12 nvert      8 node      1 parent      0 Identity  (      1   247     1     0 )  InstancedIdentity  (          1        247          1          0 ) 
     NodeInfo  nface     36 nvert     20 node      2 parent      1 Identity  (      2    21     2     0 )  InstancedIdentity  (          2         21          2          0 ) 
     NodeInfo  nface     64 nvert     34 node      3 parent      2 Identity  (      3     0     3     0 )  InstancedIdentity  (          3          0          3          0 ) 
     NodeInfo  nface     12 nvert      8 node      4 parent      2 Identity  (      4     7     4     0 )  InstancedIdentity  (          4          7          4          0 ) 
     NodeInfo  nface     12 nvert      8 node      5 parent      4 Identity  (      5     6     5     0 )  InstancedIdentity  (          5          6          5          0 ) 
     NodeInfo  nface     12 nvert      8 node      6 parent      5 Identity  (      6     3     6     0 )  InstancedIdentity  (          6          3          6          0 ) 
     NodeInfo  nface     12 nvert      8 node      7 parent      6 Identity  (      7     2     7     0 )  InstancedIdentity  (          7          2          7          0 ) 
     NodeInfo  nface      0 nvert      0 node      8 parent      7 Identity  (      8     1     8     0 )  InstancedIdentity  (         16          3          6          0 ) 
     NodeInfo  nface      0 nvert      0 node      9 parent      7 Identity  (      9     1     8     0 )  InstancedIdentity  (         17          2          7          0 ) 
     NodeInfo  nface      0 nvert      0 node     10 parent      7 Identity  (     10     1     8     0 )  InstancedIdentity  (         26          5          6          0 ) 
     NodeInfo  nface      0 nvert      0 node     11 parent      7 Identity  (     11     1     8     0 )  InstancedIdentity  (         27          4          7          0 ) 
     NodeInfo  nface      0 nvert      0 node     12 parent      7 Identity  (     12     1     8     0 )  InstancedIdentity  (         36          5          6          0 ) 
     NodeInfo  nface      0 nvert      0 node     13 parent      7 Identity  (     13     1     8     0 )  InstancedIdentity  (         37          4          7          0 ) 
     NodeInfo  nface      0 nvert      0 node     14 parent      7 Identity  (     14     1     8     0 )  InstancedIdentity  (         46          7          4          0 ) 
     NodeInfo  nface      0 nvert      0 node     15 parent      7 Identity  (     15     1     8     0 )  InstancedIdentity  (         47          6          5          0 ) 
     NodeInfo  nface     12 nvert      8 node     16 parent      5 Identity  (     16     3     6     0 )  InstancedIdentity  (         48          3          6          0 ) 
     NodeInfo  nface     12 nvert      8 node     17 parent     16 Identity  (     17     2     7     0 )  InstancedIdentity  (         49          2          7          0 ) 
     NodeInfo  nface      0 nvert      0 node     18 parent     17 Identity  (     18     1     8     0 )  InstancedIdentity  (         58          3          6          0 ) 
     NodeInfo  nface      0 nvert      0 node     19 parent     17 Identity  (     19     1     8     0 )  InstancedIdentity  (         59          2          7          0 ) 
     NodeInfo  nface     12 nvert      8 node    500 parent    499 Identity  (    500     4     7     0 )  InstancedIdentity  (       2076          2          7          0 ) 
     NodeInfo  nface      0 nvert      0 node   1000 parent    994 Identity  (   1000     1     8     0 )  InstancedIdentity  (       2792          9         10          0 ) 
     NodeInfo  nface      0 nvert      0 node   1500 parent   1498 Identity  (   1500     1     8     0 )  InstancedIdentity  (       3762         48         31          0 ) 
     NodeInfo  nface      0 nvert      0 node   2000 parent   1992 Identity  (   2000     1     8     0 )  InstancedIdentity  (       4752        126         58          0 ) 
     NodeInfo  nface     24 nvert     16 node   2500 parent   2431 Identity  (   2500    12    10     0 )  InstancedIdentity  (       6212        111         43          0 ) 
     NodeInfo  nface     24 nvert     16 node   3000 parent   2968 Identity  (   3000    11    10     0 )  InstancedIdentity  (       7826        196         84          0 ) 
     NodeInfo  nface      0 nvert      0 node   3500 parent   3499 Identity  (   3500    46    28     0 )  InstancedIdentity  (      10398        196        104          0 ) 
     NodeInfo  nface      0 nvert      0 node   4000 parent   3998 Identity  (   4000    44    30     0 )  InstancedIdentity  (      11744        219        112          0 ) 
     NodeInfo  nface     28 nvert     16 node   4500 parent   3155 Identity  (   4500    74    20     0 )  InstancedIdentity  (        192         96       8591       3150 ) 
     NodeInfo  nface      0 nvert      0 node   5000 parent   4998 Identity  (   5000    44    30     0 )  InstancedIdentity  ( 3212836864 1140981760 3267559424 3212836864 ) 
     NodeInfo  nface    192 nvert     96 node   5500 parent   4815 Identity  (   5500    48    31     0 )  InstancedIdentity  (          0     524289     524291          0 ) 
     NodeInfo  nface      0 nvert      0 node   6000 parent   5999 Identity  (   6000    46    28     0 )  InstancedIdentity  ( 3317659007 3252158464 1172473601 3316889855 ) 
     NodeInfo  nface     36 nvert     20 node   6500 parent   3152 Identity  (   6500   196    84     0 )  InstancedIdentity  ( 1104674816 1169396994 1167867904 1104674816 ) 
     NodeInfo  nface      0 nvert      0 node   7000 parent   3152 Identity  (   7000    47    80     0 )  InstancedIdentity  (          2     524292     524290          1 ) 
     NodeInfo  nface      0 nvert      0 node   7500 parent   3152 Identity  (   7500   195    83     0 )  InstancedIdentity  ( 1167867904 3249537024 1157234688 1167867904 ) 
     NodeInfo  nface      0 nvert      0 node   8000 parent   3152 Identity  (   8000   198    86     0 )  InstancedIdentity  (          4     524288     524289          1 ) 
     NodeInfo  nface      0 nvert      0 node   8500 parent   8497 Identity  (   8500    45    30     0 )  InstancedIdentity  (          3     524294     524291          2 ) 
     NodeInfo  nface    192 nvert     96 node   9000 parent   3150 Identity  (   9000   194   102     0 )  InstancedIdentity  ( 3324384191 1167098752 3235381248 3323999615 ) 
     NodeInfo  nface      0 nvert      0 node   9500 parent   3150 Identity  (   9500   197   105     0 )  InstancedIdentity  (          4     524290     524294          3 ) 
     NodeInfo  nface      0 nvert      0 node  10000 parent   9998 Identity  (  10000    44    30     0 )  InstancedIdentity  ( 3376740936 3298651698 3330323143 3376722787 ) 
     NodeInfo  nface     36 nvert     20 node  10500 parent   3150 Identity  (  10500   196   104     0 )  InstancedIdentity  (          5     524295     524296          4 ) 



The shape from the cache (1, 4486, 4) doesnt match GMesh::getInstanceIdentity treatment that assumes (12230,4)::

    In [2]: a = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/0/iidentity.npy")                                    

    In [3]: a                                                                                                                                                                                                 
    Out[3]: 
    array([[[    0,   248,     0,     0],
            [    1,   247,     1,     0],
            [    2,    21,     2,     0],
            ...,
            [12227,   243,   126,     0],
            [12228,   244,   126,     0],
            [12229,   245,   126,     0]]], dtype=uint32)

    In [4]: a.shape                                                                                                                                                                                           
    Out[4]: (1, 4486, 4)



Shape change to iidentity from Aug 2020 is not fully baked::

    080 Prior to Aug 2020 this returned an iidentity buffer with all nodes 
     81 when invoked on the root node, eg::  
     82 
     83     GMergedMesh/0/iidentity.npy :       (1, 316326, 4)
     84 
     85 This was because of a fundamental difference between the repeated instances and the 
     86 global ridx 0 volumes. The volumes of the instances are all together in a subtree 
     87 whereas the global remainder volumes with ridx 0 are scattered all over the full tree.
     88 
     89 Due to this a separate getGlobalProgeny is now used which selects the collected
     90 nodes based on the ridx (getRepeatIndex()) being zero.
     91 
     92 **/
     93 
     94 NPY<unsigned int>* GTree::makeInstanceIdentityBuffer(const std::vector<GNode*>& placements)  // static
     95 {
     96     unsigned int numInstances = placements.size() ;
     97     GNode* base0 = placements[0] ;
     98 
     99     unsigned ridx0 = base0->getRepeatIndex() ;
    100     bool is_global = ridx0 == 0 ;



