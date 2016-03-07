Geometry Review
================

Minature Dev Cycle, without the offset
---------------------------------------

Export PmtInBox geometry 

   cfg4-dpib

   # ie: ggv-pmt-test --cdetector --export --exportconfig $path

Whats happening here

* cfg4-/CDetector creates G4 geometry from the GPmt/GCSG derived 
  analytic PMT description (which came from the standalone detdesc parse).

* G4 geometry then exported by g4d-/G4DAE (built against G4 10.2) 


Using Assimp/AssimpGGeo to parse the .dae creating geocache GMergedMesh etc::

    ggv --dpib -G 
    ggv --dpib -G --loaderverbosity 3 

Load geocache GMergedMesh etc and push to GPU for OpenGL etc..::

    ggv --dpib 


The fact that this does not have the problem with Vacuum vertex sagging
might indicate a problem with older G4 (or G4DAE) not present in current one.





GGV test running ggeo-/GGeoTest
---------------------------------

::

    ggv-;ggv-pmt-test --tracer


ggv running with "--test" option loads standard geometry
(in order to have all materials etc.. available) but 
then modifies the geometry based on GGeoTestConfig 
command line args.

**BUT** crucially this grabs PMT triangulated geometry from the standard IDP source.

::

     339 void GGeo::modifyGeometry(const char* config)
     340 {
     341     // NB only invoked with test option : "ggv --test" 
     342     GGeoTestConfig* gtc = new GGeoTestConfig(config);
     343 
     344     LOG(debug) << "GGeo::modifyGeometry"
     345               << " config [" << ( config ? config : "" ) << "]" ;
     346 
     347     assert(m_geotest == NULL);
     348 
     349     m_geotest = new GGeoTest(m_cache, gtc);
     350     m_geotest->modifyGeometry();
     351 }


cfg4-/CDetector 
----------------

::

   G4VPhysicalVolume* CDetector::Construct() 


Test with::

   ggv-;ggv-pmt-test --cdetector


cfg4-/CMaker
-------------

CMaker is a constitent of CDetector used
to convert GCSG geometry into G4 geometry.


G4DAE Exports
--------------

cfg4.dae exported with::

    cfg4-dpib () 
    { 
        local msg="=== $FUNCNAME ";
        export-;
        local base=$(export-base dpib);
        local path=$base.dae;
        [ -f "$path" ] && echo $msg path $path exists already : delete and rerun to recreate && return;
        ggv-;
        ggv-pmt-test --cdetector --export --exportconfig $path
    }


::

    1521     <geometry id="sphere-i-150x109063930" name="sphere-i-150x109063930">
    1522       <mesh>
    1523         <source id="sphere-i-150x109063930-Pos">
    1524           <float_array count="864" id="sphere-i-150x109063930-Pos-array">
    1525                 98.1428 0 -13
    1526                 94.7986 25.4012 -13
    1527                 84.9941 49.0714 -13
    1528                 69.3974 69.3974 -13
    1529                 49.0714 84.9941 -13
    1530                 25.4012 94.7986 -13
    ....
    1810                 2.20269e-06 -2.20269e-06 -98
    1811                 2.69774e-06 -1.55754e-06 -98
    1812                 3.00893e-06 -8.06241e-07 -98
    1813 </float_array>

    ....

    4557     <node id="lvPmtHemiVacuum0x10905e140">
    4558       <instance_geometry url="#union-ab-i-6-fc-7-lc-110x10905e000">
    4559         <bind_material>
    4560           <technique_common>
    4561             <instance_material symbol="Vacuum" target="#Vacuum0x10905af00"/>
    4562           </technique_common>
    4563         </bind_material>
    4564       </instance_geometry>

    ////
    ////  nodes: pvPmtHemiCathode pvPmtHemiBottom pvPmtHemiDynode
    ////  are contained within lvPmtHemiVacuum
    //// 

    4565       <node id="pvPmtHemiCathode0x10905eca0">
    4566         <matrix>
    4567                 1 0 0 0
    4568 0 1 0 0
    4569 0 0 1 0
    4570 0.0 0.0 0.0 1.0
    4571 </matrix>
    4572         <instance_node url="#lvPmtHemiCathode0x10905ec10"/>
    4573         <extra>
    4574           <meta id="pvPmtHemiCathode0x10905eca0">
    4575             <copyNo>0</copyNo>
    4576             <ModuleName></ModuleName>
    4577           </meta>
    4578         </extra>
    4579       </node>

    4580       <node id="pvPmtHemiBottom0x1090620b0">
    4581         <matrix>
    4582                 1 0 0 0
    4583 0 1 0 0
    4584 0 0 1 69
    4585 0.0 0.0 0.0 1.0
    4586 </matrix>

    ////
    ////  initially surprised by the +69 Z translation, 
    ////  but looking at pmt-ecd/plot.py the radius is rather large
    ////  with restricted theta range so it makes sense that need to 
    ////  translate to the front of PMT
    ///


    4587         <instance_node url="#lvPmtHemiBottom0x109063fe0"/>
    4588         <extra>
    4589           <meta id="pvPmtHemiBottom0x1090620b0">
    4590             <copyNo>0</copyNo>
    4591             <ModuleName></ModuleName>
    4592           </meta>
    4593         </extra>
    4594       </node>

    4595       <node id="pvPmtHemiDynode0x1090622a0">
    4596         <matrix>
    4597                 1 0 0 0
    4598 0 1 0 0
    4599 0 0 1 -81.5
    4600 0.0 0.0 0.0 1.0
    4601 </matrix>
    4602         <instance_node url="#lvPmtHemiDynode0x1090621e0"/>
    4603         <extra>
    4604           <meta id="pvPmtHemiDynode0x1090622a0">
    4605             <copyNo>0</copyNo>
    4606             <ModuleName></ModuleName>
    4607           </meta>
    4608         </extra>
    4609       </node>




pmt-ecd/plot.py
-----------------

Presents GPmt 

assimp-/ColladaParser
-----------------------

Reads in the nodes


assimpwrap-/AssimpGGeo
------------------------    

Z transforms come thru as expected::

    delta:assimpwrap blyth$ ggv --dpib -G --loaderverbosity 3 


    [2016-Mar-06 13:59:23.627085]:info: AssimpGGeo::convertStructureVisit nodeIndex      4 ( mti    2 mt 0x7f98e2771710 ) OpaqueVacuum0x7fd599d5f1e0 ( mti_p    4 mt_p 0x7f98e27776f0 ) Vacuum0x7fd599d5b3b0 ( msi    1 mesh 0x7f98e2788c40 ) sphere-i-150x7fd599d63de0
    AssimpGGeo::convertStructureVisit gtransform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000     69.000 
     d      0.000      0.000      0.000      1.000 
    AssimpGGeo::convertStructureVisit ltransform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000     69.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 13:59:23.627358]:info: AssimpGGeo::convertStructureVisit nodeIndex      5 ( mti    2 mt 0x7f98e2771710 ) OpaqueVacuum0x7fd599d5f1e0 ( mti_p    4 mt_p 0x7f98e27776f0 ) Vacuum0x7fd599d5b3b0 ( msi    2 mesh 0x7f98e2782410 ) tubs-i-160x7fd599d625b0
    AssimpGGeo::convertStructureVisit gtransform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000    -81.500 
     d      0.000      0.000      0.000      1.000 
    AssimpGGeo::convertStructureVisit ltransform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000    -81.500 
     d      0.000      0.000      0.000      1.000 



ggeo-/GTreeCheck
-------------------

Finds repeated geometry and creates GMergedMesh instances for them and for the global leftovers.



ggeo-/GMergedMesh
------------------

::

    097 GMergedMesh* GGeoLib::makeMergedMesh(GGeo* ggeo, unsigned int index, GNode* base)
     98 {
     99     if(m_merged_mesh.find(index) == m_merged_mesh.end())
    100     {
    101         m_merged_mesh[index] = GMergedMesh::create(index, ggeo, base);
    102     }
    103     return m_merged_mesh[index] ;
    104 }


::

     ggv --dpib -G --meshverbosity 3 


    [2016-Mar-06 14:57:54.158467]:info: GMergedMesh::mergeSolid idx 0 id  (  0,  5,  0,  0)  pv - lv - bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000      0.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 14:57:54.158698]:info: GMergedMesh::mergeSolid idx 1 id  (  1,  4,  1,  0)  pv - lv - bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000      0.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 14:57:54.158924]:info: GMergedMesh::mergeSolid idx 2 id  (  2,  3,  2,  0)  pv - lv - bb bb min    -97.288    -97.288   -164.495  max     97.288     97.288    128.000 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000      0.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 14:57:54.159146]:info: GMergedMesh::mergeSolid idx 3 id  (  3,  0,  3,  0)  pv - lv - bb bb min    -98.138    -98.139     55.996  max     98.148     98.147    128.000 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000      0.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 14:57:54.159343]:info: GMergedMesh::mergeSolid idx 4 id  (  4,  1,  4,  0)  pv - lv - bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000     69.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 14:57:54.159511]:info: GMergedMesh::mergeSolid idx 5 id  (  5,  2,  4,  0)  pv - lv - bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000    -81.500 
     d      0.000      0.000      0.000      1.000 



::

    ggv -G --meshverbosity 3 


    [2016-Mar-06 15:04:42.447428]:info: GMergedMesh::create index 1 numVertices 1474 numFaces 2928 numSolids 5 numSolidsSelected 5
    [2016-Mar-06 15:04:42.447638]:info: GMergedMesh::mergeSolid idx 3199 id  (3199, 47, 27,  0)  pv __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xc2a6b40 lv __dd__Geometry__PMT__lvPmtHemi0xc133740 bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000      0.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 15:04:42.447953]:info: GMergedMesh::mergeSolid idx 3200 id  (3200, 46, 28,  0)  pv __dd__Geometry__PMT__lvPmtHemi--pvPmtHemiVacuum0xc1340e8 lv __dd__Geometry__PMT__lvPmtHemiVacuum0xc2c7cc8 bb bb min    -98.995    -99.003   -164.504  max     99.005     98.997    128.000 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000      0.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 15:04:42.448235]:info: GMergedMesh::mergeSolid idx 3201 id  (3201, 43, 29,  3)  pv __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode0xc02c380 lv __dd__Geometry__PMT__lvPmtHemiCathode0xc2cdca0 bb bb min    -98.138    -98.147     55.996  max     98.148     98.139    128.000 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000      0.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 15:04:42.448495]:info: GMergedMesh::mergeSolid idx 3202 id  (3202, 44, 30,  0)  pv __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiBottom0xc21de78 lv __dd__Geometry__PMT__lvPmtHemiBottom0xc12ad60 bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000     69.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 15:04:42.448748]:info: GMergedMesh::mergeSolid idx 3203 id  (3203, 45, 30,  0)  pv __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiDynode0xc04ad28 lv __dd__Geometry__PMT__lvPmtHemiDynode0xc02b280 bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000    -81.500 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 15:04:42.449049]:info: GTreeCheck::createInstancedMergedMeshes dumpSolids




Above bb z range looks correct -30 to 56.131, but the offset is stubbornly still there::


With testverbosity enabled, it looks like GGeoTest::createPmtInBox is stomping on 
preexisting solid 0. Yep, but this doesnt explain the offset.

::

    ggv-;ggv-pmt-test --tracer


    [2016-Mar-06 16:57:34.902350]:info: App:: loadGeometryBase
    [2016-Mar-06 16:57:34.902596]:info: GGeoTest::createPmtInBox B : Rock//perfectAbsorbSurface/MineralOil 0.0000,0.0000,0.0000,300.0000
    [2016-Mar-06 16:57:34.902749]:info: GGeoLib::getMergedMesh index 1 m_ggeo 0x7fc673736100 mm 0x7fc6735bc000 meshverbosity 3
    [2016-Mar-06 16:57:34.902869]:info: GGeoTest::createPmtInBox verbosity 3
    [2016-Mar-06 16:57:34.902965]:info: GGeoTest::createPmtInBox GMergedMesh::dumpSolids (before:mmpmt) 
        0 ce             gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
        1 ce             gfloat4      0.005     -0.003    -18.252    146.252  bb bb min    -98.995    -99.003   -164.504  max     99.005     98.997    128.000 
        2 ce             gfloat4      0.005     -0.004     91.998     98.143  bb bb min    -98.138    -98.147     55.996  max     98.148     98.139    128.000 
        3 ce             gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
        4 ce             gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
    [2016-Mar-06 16:57:34.904071]:info: GMergedMesh::combine making new mesh  index 1 solids 1 verbosity 3
    [2016-Mar-06 16:57:34.904192]:info: GMergedMesh::count other GMergedMesh   selected true num_solids 5 num_solids_selected 1
    [2016-Mar-06 16:57:34.904336]:info: GMergedMesh::count GSolid  selected true num_solids 6 num_solids_selected 2ar-06 16:57:34.904465]:info: GMesh::allocate numVertices 1498 numFaces 2940 numSolids 6
    [2016-Mar-06 16:57:34.904642]:info: GMesh::setCenterExtent (creates buffer)  m_center_extent 0x7fc678232fc0 m_num_solids 6
    [2016-Mar-06 16:57:34.904760]:info: GMesh::allocate DONE 
    [2016-Mar-06 16:57:34.904833]:info: GMergedMesh::mergeMergedMesh m_cur_solid 0 m_cur_vertices 0 m_cur_faces 0 other nsolid 5 selected true
    [2016-Mar-06 16:57:34.904959]:info: GMergedMesh::mergeMergedMesh m_cur_solid 0 i 0 ce gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
    [2016-Mar-06 16:57:34.905163]:info: GMergedMesh::mergeMergedMesh m_cur_solid 1 i 1 ce gfloat4      0.005     -0.003    -18.252    146.252  bb bb min    -98.995    -99.003   -164.504  max     99.005     98.997    128.000 
    [2016-Mar-06 16:57:34.905363]:info: GMergedMesh::mergeMergedMesh m_cur_solid 2 i 2 ce gfloat4      0.005     -0.004     91.998     98.143  bb bb min    -98.138    -98.147     55.996  max     98.148     98.139    128.000 
    [2016-Mar-06 16:57:34.905560]:info: GMergedMesh::mergeMergedMesh m_cur_solid 3 i 3 ce gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
    [2016-Mar-06 16:57:34.905756]:info: GMergedMesh::mergeMergedMesh m_cur_solid 4 i 4 ce gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
    [2016-Mar-06 16:57:34.906038]:info: GMergedMesh::mergeSolid m_cur_solid 5 idx 0 id  (  0,1000,123,  0)  pv - lv - bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000 
    GMergedMesh::mergeSolid transform
     a      1.000      0.000      0.000      0.000 
     b      0.000      1.000      0.000      0.000 
     c      0.000      0.000      1.000      0.000 
     d      0.000      0.000      0.000      1.000 
    [2016-Mar-06 16:57:34.906210]:fatal: GMergedMesh::mergeSolid mismatch  nodeIndex 0 m_cur_solid 5
    [2016-Mar-06 16:57:34.906420]:info: GGeoTest::createPmtInBox GMergedMesh::dumpSolids (after:tri) 
        0 ce             gfloat4      0.000      0.000      0.000    300.000  bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000 
        1 ce             gfloat4      0.005     -0.003    -18.252    146.252  bb bb min    -98.995    -99.003   -164.504  max     99.005     98.997    128.000 
        2 ce             gfloat4      0.005     -0.004     91.998     98.143  bb bb min    -98.138    -98.147     55.996  max     98.148     98.139    128.000 
        3 ce             gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
        4 ce             gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
        5 ce             gfloat4      0.000      0.000      0.000    300.000  bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000 
    [2016-Mar-06 16:57:34.906980]:info: App:: modifyGeometry
    [2016-Mar-06 16:57:34.907054]:info: App::registerGeometry
    [2016-Mar-06 16:57:34.907133]:info: GGeoLib::getMergedMesh index 0 m_ggeo 0x7fc673736100 mm 0x7fc678232b40 meshverbosity 3


With offset::

    [2016-Mar-06 16:57:34.902965]:info: GGeoTest::createPmtInBox GMergedMesh::dumpSolids (before:mmpmt) 
        0 ce             gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
        1 ce             gfloat4      0.005     -0.003    -18.252    146.252  bb bb min    -98.995    -99.003   -164.504  max     99.005     98.997    128.000 
        2 ce             gfloat4      0.005     -0.004     91.998     98.143  bb bb min    -98.138    -98.147     55.996  max     98.148     98.139    128.000 
        3 ce             gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
        4 ce             gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
 
Without offset::

    ggv --dpib --meshverbosity 3

    [2016-Mar-06 17:42:35.481308]:info: App::loadGeometryBase mesh0
        0 ce             gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
        1 ce             gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
        2 ce             gfloat4      0.000      0.000    -18.247    146.247  bb bb min    -97.288    -97.288   -164.495  max     97.288     97.288    128.000 
        3 ce             gfloat4      0.005      0.004     91.998     98.143  bb bb min    -98.138    -98.139     55.996  max     98.148     98.147    128.000 
        4 ce             gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
        5 ce             gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
    [2016-Mar-06 17:42:35.481514]:info: App:: loadGeometryBase

    delta:ggeoview blyth$ mv /tmp/vbuf.npy /tmp/dpib_vbuf.npy



Is there an offset by 1 mismatch ?


::

    In [1]: a = np.load("/tmp/dpib_vbuf.npy")

    In [2]: b = np.load("/tmp/vbuf_modifyGeometry.npy")

    In [3]: a.shape
    Out[3]: (1494, 3)

    In [4]: b.shape
    Out[4]: (1498, 3)

    In [5]: a
    Out[5]: 
    array([[   0.   ,    0.   ,  131.   ],
           [  33.905,    0.   ,  126.536],
           [  32.75 ,    8.775,  126.536],
           ..., 
           [   0.   ,   -0.   ,  -29.   ],
           [   0.   ,   -0.   ,  -29.   ],
           [   0.   ,   -0.   ,  -29.   ]], dtype=float32)

    In [6]: b
    Out[6]: 
    array([[   0.   ,    0.   ,  131.   ],
           [  33.905,    0.   ,  126.536],
           [  32.75 ,    8.775,  126.536],
           ..., 
           [ 300.   ,  300.   , -300.   ],
           [ 300.   , -300.   , -300.   ],
           [-300.   , -300.   , -300.   ]], dtype=float32)



npy-/mesh.py GMergedMesh check
--------------------------------

Combines PMT analytic plotting with vertices rz plotting, from GMergedMesh vertices 
loaded from::

    /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1/vertices.npy 
     
Contrary to prior, the problem is with the vacuum (solid 1), not the PMT bottom.



dump the base and modified meshes from pmt test
--------------------------------------------------

::

    ggv-;ggv-pmt-test --tracer --meshverbosity 3


::

    134     #base = os.path.expandvars("$IDPATH/GMergedMesh/1")
    135     #base = "/tmp/GMergedMesh/baseGeometry"
    136     #base = "/tmp/GMergedMesh/modifyGeometry"
    137     base = os.path.expandvars("$IDPATH_DPIB/GMergedMesh/0")
    138 
    139     mm = MergedMesh(base=base)
    140 
    141     pmt = Pmt()
    142     ALL, PYREX, VACUUM, CATHODE, BOTTOM, DYNODE = None,0,1,2,3,4
    143     pts = pmt.parts(ALL)
    144 
    145     fig = plt.figure()
    146     
    147     #one_plot(fig, pmt, pts, axes=ZX, clip=True)
    148     
    149     solids_plot(fig, pmt, mm, solids=range(5))
    150     
    151     #plot_vertices(fig, mm)
    152     
    153     plt.show()


Only "$IDPATH_DPIB/GMergedMesh/0" does not have the vacuum sagging vertices problem, 
but needed to offset that by one. Plus it has other nodeinfo issues::

::

    In [4]: mm.nodeinfo
    Out[4]: 
    array([[         0,          0,          0, 4294967295],
           [       720,        362,          1,          0],
           [       720,        362,          2,          1],
           [       960,        482,          3,          2],
           [       576,        288,          4,          2],
           [         0,          0,          5,          2]], dtype=uint32)

    In [16]: mm.nodeinfo.view(np.int32)
    Out[16]: 
    array([[  0,   0,   0,  -1],
           [720, 362,   1,   0],
           [720, 362,   2,   1],
           [960, 482,   3,   2],
           [576, 288,   4,   2],
           [  0,   0,   5,   2]], dtype=int32)




