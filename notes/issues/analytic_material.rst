Full Analytic Route Needs Material Hookup
============================================

As GDML does not contain optical props will need 
to interface to G4DAE material props.  

* better to think of this as geometry merging between the two routes


TODO : C++ alignment  + cross accessors 
--------------------------------------------
 
* need partial alignment check
* cross accessors : to get to corresponding objects in "other" world, just need the right index

* verify oxrap can continue unchanged

* COMPLICATION : tri when operating from cache, does not have the volume tree persisted/saved
  only the merged mesh and its solid related buffers are persisted ?

* Hmm but due to instancing splits of the GMergedMesh this aint so easy ?
  Unless the global one retains all solid info ?


Yep mm0 is holding full solid info::

    In [1]: run mergedmesh.py

    [2017-06-21 12:59:49,242] p1196 {/Users/blyth/opticks/ana/mergedmesh.py:51} INFO - ready buffers from /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 
    -rw-r--r--  1 blyth  staff  96 Jun 14 16:21 /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0/aiidentity.npy
    -rw-r--r--  1 blyth  staff  293600 Jun 14 16:21 /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0/bbox.npy
    ...
    In [2]: mm
    Out[2]: 
               aiidentity : (1, 1, 4) 
                     bbox : (12230, 6) 
               boundaries : (434816, 1) 
            center_extent : (12230, 4) 
                   colors : (225200, 3) 
                 identity : (12230, 4) 
                iidentity : (12230, 4) 
                  indices : (1304448, 1) 
              itransforms : (1, 4, 4) 
                   meshes : (12230, 1) 
                 nodeinfo : (12230, 4) 
                    nodes : (434816, 1) 
                  normals : (225200, 3) 
                  sensors : (434816, 1) 
               transforms : (12230, 16) 
                 vertices : (225200, 3) 


Looks like key method is::


     GMergedMesh::mergeSolidIdentity( GSolid* solid, bool selected ) 

     m_nodeinfo
     m_identity

     394 guint4* GMesh::getNodeInfo()
     395 {
     396     return m_nodeinfo ;
     397 }
     398 guint4 GMesh::getNodeInfo(unsigned int index)
     399 {
     400     return m_nodeinfo[index] ;
     401 }
     402 
     403 guint4* GMesh::getIdentity()
     404 {
     405     return m_identity ;
     406 }
     407 guint4 GMesh::getIdentity(unsigned int index)
     408 {
     409     return m_identity[index] ;
     410 }





Why is the tri nodelib numPV partial ?

* hmm tis reading the wrong one, need to use reldir prefix to distinguish.. and prepare path

::

    2017-06-21 12:33:52.824 INFO  [8897976] [GScene::compareTrees@84] GScene::compareTrees
     ana GNodeLib targetnode 3153 numPV 1660 numLV 1660 numSolids 1660 PV(0) /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xc2cf528 LV(0) /dd/Geometry/AD/lvADE0xc2a78c0
     tri GNodeLib targetnode 3153 numPV 1660 numLV 1660 numSolids 0 PV(0) /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xc2cf528 LV(0) /dd/Geometry/AD/lvADE0xc2a78c0
    Assertion failed: (0 && "GScene::init early exit for gltf==44"), function init, file /Users/blyth/opticks/ggeo/GScene.cc, line 72.
    Process 33509 stopped


Use reldir prefix to distinguish

::

    2017-06-21 15:30:52.867 INFO  [53670] [*GScene::createVolumeTree@207] GScene::createVolumeTree DONE num_nodes: 1660
    2017-06-21 15:30:52.870 INFO  [53670] [GNodeLib::save@64] GNodeLib::save idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae targetNodeOffset 3153
    2017-06-21 15:30:52.870 INFO  [53670] [GItemList::save@88] GItemList::save writing to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/analytic/GScene/GNodeLib/PVNames.txt
    2017-06-21 15:30:52.872 INFO  [53670] [GItemList::save@88] GItemList::save writing to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/analytic/GScene/GNodeLib/LVNames.txt
    2017-06-21 15:30:52.873 INFO  [53670] [GScene::compareTrees@87] GScene::compareTrees num_nd 1660 targetnode 3153
    2017-06-21 15:30:52.873 INFO  [53670] [GScene::compareTrees@92] nodelib (GSolid) volumes 
     ana GNodeLib targetnode 3153 reldir analytic/GScene/GNodeLib numPV 1660 numLV 1660 numSolids 1660 PV(0) /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xc2cf528 LV(0) /dd/Geometry/AD/lvADE0xc2a78c0
     tri GNodeLib targetnode 3153 reldir - numPV 12230 numLV 12230 numSolids 0 PV(0) top LV(0) World0xc15cfc0
    2017-06-21 15:30:52.873 INFO  [53670] [GScene::compareTrees@96] geolib (GMergedMesh)  
     tri GGeoLib numMergedMesh 2
         0  3153 ID(nd/ms/bd/sn)  (3153,192, 17,  0)  NI(nf/nv/ix/px)  ( 96, 50,3153,3152) 
         1  3154 ID(nd/ms/bd/sn)  (3154, 94, 18,  0)  NI(nf/nv/ix/px)  ( 96, 50,3154,3153) 
         2  3155 ID(nd/ms/bd/sn)  (3155, 90, 19,  0)  NI(nf/nv/ix/px)  ( 96, 50,3155,3154) 
         3  3156 ID(nd/ms/bd/sn)  (3156, 42, 20,  0)  NI(nf/nv/ix/px)  (288,146,3156,3155) 
         4  3157 ID(nd/ms/bd/sn)  (3157, 37, 21,  0)  NI(nf/nv/ix/px)  (332,168,3157,3156) 
         5  3158 ID(nd/ms/bd/sn)  (3158, 24, 22,  0)  NI(nf/nv/ix/px)  (288,146,3158,3157) 
         6  3159 ID(nd/ms/bd/sn)  (3159, 22, 23,  0)  NI(nf/nv/ix/px)  (288,146,3159,3158) 




GMergedMesh expecting relative ?
-----------------------------------


::

    2017-06-21 19:31:30.894 INFO  [149669] [GItemList::save@88] GItemList::save writing to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/analytic/GScene/GNodeLib/PVNames.txt
    2017-06-21 19:31:30.896 INFO  [149669] [GItemList::save@88] GItemList::save writing to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/analytic/GScene/GNodeLib/LVNames.txt
    2017-06-21 19:31:30.897 INFO  [149669] [GScene::compareTrees@132] nodelib (GSolid) volumes 
     ana GNodeLib targetnode 3153 reldir analytic/GScene/GNodeLib numPV 1660 numLV 1660 numSolids 1660 PV(0) /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xc2cf528 LV(0) /dd/Geometry/AD/lvADE0xc2a78c0
     tri GNodeLib targetnode 3153 reldir - numPV 12230 numLV 12230 numSolids 0 PV(0) top LV(0) World0xc15cfc0
    2017-06-21 19:31:30.897 INFO  [149669] [GScene::makeMergedMeshAndInstancedBuffers@497] GScene::makeMergedMeshAndInstancedBuffers num_repeats 21 START 
    2017-06-21 19:31:30.959 FATAL [149669] [GMergedMesh::mergeSolidIdentity@482] GMergedMesh::mergeSolid mismatch  nodeIndex 3153 m_cur_solid 0
    2017-06-21 19:31:30.960 FATAL [149669] [GMergedMesh::mergeSolidIdentity@482] GMergedMesh::mergeSolid mismatch  nodeIndex 3154 m_cur_solid 1
    2017-06-21 19:31:30.961 FATAL [149669] [GMergedMesh::mergeSolidIdentity@482] GMergedMesh::mergeSolid mismatch  nodeIndex 3155 m_cur_solid 2
    2017-06-21 19:31:30.962 FATAL [149669] [GMergedMesh::mergeSolidIdentity@482] GMergedMesh::mergeSolid mismatch  nodeIndex 3156 m_cur_solid 3
    2017-06-21 19:31:30.964 FATAL [149669] [GMergedMesh::mergeSolidIdentity@482] GMergedMesh::mergeSolid mismatch  nodeIndex 3157 m_cur_solid 4
    2017-06-21 19:31:30.965 FATAL [149669] [GMergedMesh::mergeSolidIdentity@482] GMergedMesh::mergeSolid mismatch  nodeIndex 3158 m_cur_solid 5

::

    479     if(isGlobal())
    480     {
    481          if(nodeIndex != m_cur_solid)
    482              LOG(fatal) << "GMergedMesh::mergeSolidIdentity mismatch "
    483                         <<  " nodeIndex " << nodeIndex
    484                         <<  " m_cur_solid " << m_cur_solid
    485                         ;
    486 
    487          //assert(nodeIndex == m_cur_solid);  // trips ggv-pmt still needed ?
    488     }






mesh index alignment looks like simple offset wont work
--------------------------------------------------------

Using the assimp aindex? Presumably this means absolute (full geometry) mesh indexing::

     879 GMesh* GGeo::getMesh(unsigned int aindex)
     880 {
     881     GMesh* mesh = NULL ;
     882     for(unsigned int i=0 ; i < m_meshes.size() ; i++ )
     883     {
     884         if(m_meshes[i]->getIndex() == aindex )
     885         {
     886             mesh = m_meshes[i] ;
     887             break ;
     888         }
     889     }
     890     return mesh ;
     891 }

     912 void GGeo::add(GMesh* mesh)
     913 {
     914     m_meshes.push_back(mesh);
     915 
     916     const char* name = mesh->getName();
     917     unsigned int index = mesh->getIndex();
     918 
     919     LOG(debug) << "GGeo::add (GMesh)"
     920               << " index " << std::setw(4) << index
     921               << " name " << name
     922               ;
     923 
     924     m_meshindex->add(name, index);
     925 }





GGeo holds a meshindex from geocache::

    simon:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ head -10 MeshIndexSource.json
    {
        "AcrylicCylinder0xc3d3830": "136",
        "AdPmtCollar0xc2c5260": "48",
        "AmCCo60AcrylicContainer0xc0b23b8": "131",
        "AmCCo60Cavity0xc0b3de0": "130",
        "AmCCo60SourceAcrylic0xc3ce678": "122",
        "AmCSS0xc3d0040": "120",
        "AmCSSCap0xc3cfc58": "115",
        "AmCSource0xc3d0708": "119",
        "AmCSourceAcrylicCup0xc3d1bc8": "118",


    simon:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ head -10 MeshIndexLocal.json
    {
        "AcrylicCylinder0xc3d3830": "137",
        "AdPmtCollar0xc2c5260": "49",
        "AmCCo60AcrylicContainer0xc0b23b8": "132",
        "AmCCo60Cavity0xc0b3de0": "131",
        "AmCCo60SourceAcrylic0xc3ce678": "123",
        "AmCSS0xc3d0040": "121",
        "AmCSSCap0xc3cfc58": "116",
        "AmCSource0xc3d0708": "120",
        "AmCSourceAcrylicCup0xc3d1bc8": "119",
    simon:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ 



::

    simon:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ grep \"0\" MeshIndexSource.json
        "near_top_cover_box0xc23f970": "0",
    simon:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ grep \"0\" MeshIndexLocal.json
    simon:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ 
    simon:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ grep \"1\" MeshIndexLocal.json
        "near_top_cover_box0xc23f970": "1",
    simon:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ 





GLTF meshes/extras has soName and lvIdx to create mesh indices mapping.

/tmp/blyth/opticks/tgltf/tgltf-gdml--.pretty.gltf::

     0009     "meshes": [
       10         {
       11             "extras": {
       12                 "lvIdx": 192,
       13                 "soName": "ade0xc2a7438",
       14                 "uri": "extras/192"
       15             },
       16             "name": "/dd/Geometry/AD/lvADE0xc2a78c0",
       17             "primitives": [
       18                 {
       19                     "attributes": []
       20                 }
       21             ]
       22         },



     1258         {
     1259             "extras": {
     1260                 "lvIdx": 131,
     1261                 "soName": "AmCCo60AcrylicContainer0xc0b23b8",
     1262                 "uri": "extras/131"
     1263             },
     1264             "name": "/dd/Geometry/CalibrationSources/lvAmCCo60AcrylicContainer0xc0b2d78",
     1265             "primitives": [
     1266                 {
     1267                     "attributes": []
     1268                 }
     1269             ]
     1270         },



DONE : sensor crossover
--------------------------

g4_00.idmap::

    3196 0 0 (-661623,449556,5116.69) (0.543174,0.83962,0)(-0.83962,0.543174,0)(0,0,1) /dd/Geometry/AdDetails/lvOcrGdsLsoInOav#pvOcrGdsTfbInOav
    3197 0 0 (-661623,449556,5116.69) (0.543174,0.83962,0)(-0.83962,0.543174,0)(0,0,1) /dd/Geometry/AdDetails/lvOcrGdsTfbInOav#pvOcrGdsInOav
    3198 0 0 (-661623,449556,5116.69) (0.543174,0.83962,0)(-0.83962,0.543174,0)(0,0,1) /dd/Geometry/AD/lvOAV#pvOcrCalLsoInOav
    3199 16843009 1010101 (8842.5,532069,599609) (3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17) /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt
    3200 16843009 1010101 (8842.5,532069,599609) (3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    3201 16843009 1010101 (8842.5,532069,599609) (3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
    3202 16843009 1010101 (8842.5,532069,599540) (3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
    3203 16843009 1010101 (8842.5,532069,599690) (3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode
    3204 0 0 (8842.5,532069,599553) (3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17) /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar
    3205 16843010 1010102 (8842.5,668528,441547) (5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17) /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt
    3206 16843010 1010102 (8842.5,668528,441547) (5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    3207 16843010 1010102 (8842.5,668528,441547) (5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
    3208 16843010 1010102 (8842.5,668528,441478) (5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
    3209 16843010 1010102 (8842.5,668528,441628) (5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode
    3210 0 0 (8842.5,668528,441491) (5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17) /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmtCollar
    3211 16843011 1010103 (8842.5,759428,253553) (5.76825e-17,0.335452,-0.942057)(-2.05398e-17,0.942057,0.335452)(1,6.16298e-33,6.12303e-17) /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUnit#pvAdPmt



::

    simon:opticksnpy blyth$ NSensorListTest --sensor 3198 3199 3200 3201 3202 3203 3204
    nodeIndex 0 sensor NULL 
    nodeIndex 3198 sensor NULL 
    nodeIndex 3199 sensor NSensor  index      0 idhex 1010101 iddec 16843009 node_index   3199 name /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt NOT-CATHODE 
    nodeIndex 3200 sensor NSensor  index      1 idhex 1010101 iddec 16843009 node_index   3200 name /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum NOT-CATHODE 
    nodeIndex 3201 sensor NSensor  index      2 idhex 1010101 iddec 16843009 node_index   3201 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode CATHODE 
    nodeIndex 3202 sensor NSensor  index      3 idhex 1010101 iddec 16843009 node_index   3202 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom NOT-CATHODE 
    nodeIndex 3203 sensor NSensor  index      4 idhex 1010101 iddec 16843009 node_index   3203 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode NOT-CATHODE 
    nodeIndex 3204 sensor NULL 
    simon:opticksnpy blyth$ 



::

    168 void GSolid::setSensor(NSensor* sensor)
    169 {
    170     m_sensor = sensor ;
    171     // every triangle needs a value... use 0 to mean unset, so sensor   
    172     setSensorIndices( NSensor::RefIndex(sensor) );
    173 }
    174 

     26 unsigned int NSensor::getIndex()
     27 {
     28     return m_index ;
     29 }  
     30 unsigned int NSensor::getIndex1()
     31 {
     32     return m_index + 1 ;
     33 }  
     34 
     35 unsigned int NSensor::RefIndex(NSensor* sensor)
     36 {
     37     return sensor ? sensor->getIndex1() : NSensor::UNSET_INDEX  ;
     38 }



All 5 nodes of the PMT have associated NSensor but only cathode has non-zero index::

    2017-06-21 17:57:26.955 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 21 tri_id.y 39 tri_meshIdx 39 mesh_idx 21
    2017-06-21 17:57:26.955 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 22 tri_id.y 38 tri_meshIdx 38 mesh_idx 22
    2017-06-21 17:57:26.955 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 23 tri_id.y 41 tri_meshIdx 41 mesh_idx 23
    got sensor  tri_nodeIdx 3199 tri_sensorSurfaceIdx 0
    2017-06-21 17:57:26.955 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 24 tri_id.y 47 tri_meshIdx 47 mesh_idx 24
    got sensor  tri_nodeIdx 3200 tri_sensorSurfaceIdx 0
    2017-06-21 17:57:26.955 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 25 tri_id.y 46 tri_meshIdx 46 mesh_idx 25
    got sensor  tri_nodeIdx 3201 tri_sensorSurfaceIdx 3
    2017-06-21 17:57:26.955 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 26 tri_id.y 43 tri_meshIdx 43 mesh_idx 26
    got sensor  tri_nodeIdx 3202 tri_sensorSurfaceIdx 0
    2017-06-21 17:57:26.956 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 27 tri_id.y 44 tri_meshIdx 44 mesh_idx 27
    got sensor  tri_nodeIdx 3203 tri_sensorSurfaceIdx 0
    2017-06-21 17:57:26.956 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 28 tri_id.y 45 tri_meshIdx 45 mesh_idx 28
    2017-06-21 17:57:26.956 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 29 tri_id.y 48 tri_meshIdx 48 mesh_idx 29
    got sensor  tri_nodeIdx 3205 tri_sensorSurfaceIdx 0
    2017-06-21 17:57:26.956 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 24 tri_id.y 47 tri_meshIdx 47 mesh_idx 24
    got sensor  tri_nodeIdx 3206 tri_sensorSurfaceIdx 0
    2017-06-21 17:57:26.956 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 25 tri_id.y 46 tri_meshIdx 46 mesh_idx 25
    got sensor  tri_nodeIdx 3207 tri_sensorSurfaceIdx 8
    2017-06-21 17:57:26.956 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 26 tri_id.y 43 tri_meshIdx 43 mesh_idx 26
    got sensor  tri_nodeIdx 3208 tri_sensorSurfaceIdx 0
    2017-06-21 17:57:26.956 INFO  [109843] [*GScene::createVolume@355]  match_mesh_index  check_id.y 27 tri_id.y 44 tri_meshIdx 44 mesh_idx 27








DONE : gltftarget(config) and targetnode(metadata)
-----------------------------------------------------

Partial targetnode now available via these routes.



DONE : analytic/triangulated tree alignment for full traversals
------------------------------------------------------------------

Verified same node counts, traversal order and pv/lv identifiers 
in ana/nodelib.py 

Implemented using 2 GGeoLib instances for the GMergedMesh
and 2 GNodeLib instances for the GSolid:

* triangulated directly in GGeo
* analytic within GGeo/GScene


GGeo/GGeoLib
---------------

Normally loaded from cache::


     610 void GGeo::loadFromCache()
     611 {  
     612     LOG(trace) << "GGeo::loadFromCache START" ;
     613 
     614     m_geolib = GGeoLib::load(m_ok);
     615    


GDML File LV volume elements have material refs
--------------------------------------------------

/tmp/g4_00.gdml::

     .121     <material name="/dd/Materials/GdDopedLS0xc2a8ed0" state="solid">
      122       <P unit="pascal" value="101324.946686941"/>
      123       <D unit="g/cm3" value="0.86019954739804"/>
     ....
     3809     <volume name="/dd/Geometry/AD/lvGDS0xbf6cbb8">
     3810       <materialref ref="/dd/Materials/GdDopedLS0xc2a8ed0"/>
     3811       <solidref ref="gds0xc28d3f0"/>
     3812     </volume>
     3813     <volume name="/dd/Geometry/AdDetails/lvOcrGdsInIav0xbf6dd58">
     3814       <materialref ref="/dd/Materials/GdDopedLS0xc2a8ed0"/>
     3815       <solidref ref="OcrGdsInIav0xc405b10"/>
     3816     </volume>
     3817     <volume name="/dd/Geometry/AD/lvIAV0xc404ee8">
     3818       <materialref ref="/dd/Materials/Acrylic0xc02ab98"/>
     3819       <solidref ref="iav0xc346f90"/>
     3820       <physvol name="/dd/Geometry/AD/lvIAV#pvGDS0xbf6ab00">
     3821         <volumeref ref="/dd/Geometry/AD/lvGDS0xbf6cbb8"/>
     3822         <position name="/dd/Geometry/AD/lvIAV#pvGDS0xbf6ab00_pos" unit="mm" x="0" y="0" z="7.5"/>
     3823       </physvol>
     3824       <physvol name="/dd/Geometry/AD/lvIAV#pvOcrGdsInIAV0xbf6b0e0">
     3825         <volumeref ref="/dd/Geometry/AdDetails/lvOcrGdsInIav0xbf6dd58"/>
     3826         <position name="/dd/Geometry/AD/lvIAV#pvOcrGdsInIAV0xbf6b0e0_pos" unit="mm" x="0" y="0" z="1587.21981588594"/>
     3827       </physvol>
     3828     </volume>


analytic/gdml.py::

     861 class Volume(G):
     862     """
     863     ::
     864 
     865         In [15]: for v in gdml.volumes.values():print v.material.shortname
     866         PPE
     867         MixGas
     868         Air
     869         Bakelite
     870         Air
     871         Bakelite
     872         Foam
     873         Aluminium
     874         Air
     875         ...
     876 
     877     """
     878     materialref = property(lambda self:self.elem.find("materialref").attrib["ref"])
     879     solidref = property(lambda self:self.elem.find("solidref").attrib["ref"])
     880     solid = property(lambda self:self.g.solids[self.solidref])
     881     material = property(lambda self:self.g.materials[self.materialref])
     882 


Whats needed for analytic material ?
---------------------------------------

* need boundary "omat/osur/isur/imat" spec strings for all volumes...


In tboolean testing these boundary spec are set manually on the 
csg object of the solids.

    343 container = CSG("box")
    344 container.boundary = args.container

ana/base.py::

    305     container = kwa.get("container","Rock//perfectAbsorbSurface/Vacuum")
    306     testobject = kwa.get("testobject","Vacuum///GlassSchottF2" )


npy/NCSG.cpp sets boundary strings on the NCSG tree instances::

     885 int NCSG::Deserialize(const char* basedir, std::vector<NCSG*>& trees, int verbosity )
     886 {
     ...
     898     NTxt bnd(txtpath.c_str());
     899     bnd.read();
     900     //bnd.dump("NCSG::Deserialize");    
     901 
     902     unsigned nbnd = bnd.getNumLines();
     903 
     904     LOG(info) << "NCSG::Deserialize"
     905               << " VERBOSITY " << verbosity
     906               << " basedir " << basedir
     907               << " txtpath " << txtpath
     908               << " nbnd " << nbnd
     909               ;
     ...
     917     for(unsigned j=0 ; j < nbnd ; j++)
     918     {
     919         unsigned i = nbnd - 1 - j ;
     920         std::string treedir = BFile::FormPath(basedir, BStr::itoa(i));
     921 
     922         NCSG* tree = new NCSG(treedir.c_str());
     923         tree->setIndex(i);
     924         tree->setVerbosity( verbosity );
     925         tree->setBoundary( bnd.getLine(i) );
     926 

Which are serialized from python source via a csg.txt bnd file::

    simon:tboolean-disc-- blyth$ pwd
    /tmp/blyth/opticks/tboolean-disc--
    simon:tboolean-disc-- blyth$ cat csg.txt 
    Rock//perfectAbsorbSurface/Vacuum
    Vacuum///GlassSchottF2


The above is the python CSG testing route, what about full analytic GDML/GLTF  route ? tgltf-gdml

* the boundary from the node/extras of the GLTF is applied to the structural nd in  NScene::import_r

::

    278 nd* NScene::import_r(int idx,  nd* parent, int depth)
    279 {
    280     ygltf::node_t* ynode = getNode(idx);
    281     auto extras = ynode->extras ;
    282     std::string boundary = extras["boundary"] ;
    283 
    284     nd* n = new nd ;   // NB these are structural nodes, not CSG tree nodes
    285 
    286     n->idx = idx ;
    287     n->repeatIdx = 0 ;
    288     n->mesh = ynode->mesh ;
    289     n->parent = parent ;
    290     n->depth = depth ;
    291     n->boundary = boundary ;
    292     n->transform = new nmat4triple( ynode->matrix.data() );
    293     n->gtransform = nd::make_global_transform(n) ;
    294 
    295     for(int child : ynode->children) n->children.push_back(import_r(child, n, depth+1));  // recursive call
    296 
    297     m_nd[idx] = n ;
    298 
    299     return n ;
    300 }




::

    113 tgltf-gdml(){  TGLTFPATH=$($FUNCNAME- 2>/dev/null) tgltf-- $* ; }

    115 tgltf-gdml--(){ cat << EOP
    116 
    117 import os, logging, sys, numpy as np
    118 
    119 log = logging.getLogger(__name__)
    120 
    121 from opticks.ana.base import opticks_main
    122 from opticks.analytic.treebase import Tree
    123 from opticks.analytic.gdml import GDML
    124 from opticks.analytic.sc import Sc
    125 
    126 args = opticks_main()
    127 
    128 oil = "/dd/Geometry/AD/lvOIL0xbf5e0b8"
    129 #sel = oil
    130 #sel = 3153
    131 sel = 1
    132 idx = 0 
    133 
    134 wgg = GDML.parse()
    135 tree = Tree(wgg.world)
    136 
    137 target = tree.findnode(sel=sel, idx=idx)
    138 
    139 sc = Sc(maxcsgheight=3)
    140 sc.extras["verbosity"] = 1
    141 
    142 tg = sc.add_tree_gdml( target, maxdepth=0)
    143 
    144 path = "$TMP/tgltf/$FUNCNAME.gltf"
    145 gltf = sc.save(path)
    146 
    147 print path      ## <-- WARNING COMMUNICATION PRINT
    148 
    149 EOP
    150 }


    039 tgltf--(){
     40 
     41     tgltf-
     42 
     43     local cmdline=$*
     44     local tgltfpath=${TGLTFPATH:-$TMP/nd/scene.gltf}
     45 
     46     local gltf=1
     47     #local gltf=4  # early exit from GGeo::loadFromGLTF
     48 
     49     op.sh  \
     50             $cmdline \
     51             --debugger \
     52             --gltf $gltf \
     53             --gltfbase $(dirname $tgltfpath) \
     54             --gltfname $(basename $tgltfpath) \
     55             --target 3 \
     56             --animtimemax 10 \
     57             --timemax 10 \
     58             --geocenter \
     59             --eye 1,0,0 \
     60             --dbganalytic \
     61             --tag $(tgltf-tag) --cat $(tgltf-det) \
     62             --save
     63 }



::


    simon:issues blyth$ tgltf-;tgltf-gdml-
    args: 
    [2017-06-20 14:02:53,885] p85498 {/Users/blyth/opticks/analytic/gdml.py:987} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 
    [2017-06-20 14:02:53,923] p85498 {/Users/blyth/opticks/analytic/gdml.py:1001} INFO - wrapping gdml element  
    [2017-06-20 14:02:54,765] p85498 {/Users/blyth/opticks/analytic/sc.py:279} INFO - add_tree_gdml START maxdepth:0 maxcsgheight:3 nodesCount:    0
    ...
    [2017-06-20 14:02:57,976] p85498 {/Users/blyth/opticks/analytic/sc.py:304} INFO - saving to /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf 
    [2017-06-20 14:02:58,221] p85498 {/Users/blyth/opticks/analytic/sc.py:300} INFO - save_extras /tmp/blyth/opticks/tgltf/extras  : saved 248 
    /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf


     cat /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf | python -m json.tool




/tmp/blyth/opticks/tgltf/tgltf-gdml--.pretty.gltf the boundary spec are in nodes extras::

    3234     "nodes": [
    3235         {
    3236             "children": [
    3237                 1,
    3238                 3146
    3239             ],
    3240             "extras": {
    3241                 "boundary": "Vacuum///Rock"
    3242             },

    3243             "matrix": [
    3244                 -0.5431744456291199,
    ....
    3259                 1.0
    3260             ],
    3261             "mesh": 0,
    3262             "name": "ndIdx:  0,soIdx:  0,lvName:/dd/Geometry/Sites/lvNearSiteRock0xc030350"
    3263         },



Currently no surface spec::

    simon:opticksnpy blyth$ grep boundary /tmp/blyth/opticks/tgltf/tgltf-gdml--.pretty.gltf | sort | uniq
                    "boundary": "Acrylic///Air"
                    "boundary": "Acrylic///Aluminium"
                    "boundary": "Acrylic///GdDopedLS"
                    "boundary": "Acrylic///LiquidScintillator"
                    "boundary": "Acrylic///Nylon"
                    "boundary": "Acrylic///StainlessSteel"
                    "boundary": "Acrylic///Vacuum"
                    "boundary": "Air///Acrylic"
                    "boundary": "Air///Air"
                    "boundary": "Air///Aluminium"
                    "boundary": "Air///ESR"
                    "boundary": "Air///Iron"
                    "boundary": "Air///MixGas"
                    "boundary": "Air///PPE"
                    "boundary": "Air///StainlessSteel"
                    "boundary": "Aluminium///Co_60"
                    "boundary": "Aluminium///Foam"
                    "boundary": "Aluminium///Ge_68"
                    "boundary": "Bakelite///Air"
                    "boundary": "DeadWater///ADTableStainlessSteel"
                    "boundary": "DeadWater///Tyvek"
                    "boundary": "Foam///Bakelite"
                    "boundary": "IwsWater///ADTableStainlessSteel"
                    "boundary": "IwsWater///IwsWater"
                    "boundary": "IwsWater///PVC"
                    "boundary": "IwsWater///Pyrex"
                    "boundary": "IwsWater///StainlessSteel"
                    "boundary": "IwsWater///UnstStainlessSteel"
                    "boundary": "IwsWater///Water"
                    "boundary": "LiquidScintillator///Acrylic"
                    "boundary": "LiquidScintillator///GdDopedLS"
                    "boundary": "LiquidScintillator///Teflon"
                    "boundary": "MineralOil///Acrylic"


analytic/sc.py::

    034 class Nd(object):
     35     def __init__(self, ndIdx, soIdx, transform, boundary, name, depth, scene):
     36         """
     37         :param ndIdx: local within subtree nd index, used for child/parent Nd referencing
     38         :param soIdx: local within substree so index, used for referencing to distinct solids/meshes
     39         """
     40         self.ndIdx = ndIdx
     41         self.soIdx = soIdx
     42         self.transform = transform
     43         self.extras = dict(boundary=boundary)

    090 class Sc(object):
     91     def __init__(self, maxcsgheight=4):
    ...
    144     def add_node(self, lvIdx, lvName, soName, transform, boundary, depth):
    145 
    146         mesh = self.add_mesh(lvIdx, lvName, soName)
    147         soIdx = mesh.soIdx
    148 
    149         ndIdx = len(self.nodes)
    150         name = "ndIdx:%3d,soIdx:%3d,lvName:%s" % (ndIdx, soIdx, lvName)
    151 
    152         #log.info("add_node %s " % name)
    153         assert transform is not None
    154 
    155         nd = Nd(ndIdx, soIdx, transform, boundary, name, depth, self )
    156         nd.mesh = mesh
    ...
    166     def add_node_gdml(self, node, depth, debug=False):
    167 
    168         lvIdx = node.lv.idx
    169         lvName = node.lv.name
    170         soName = node.lv.solid.name
    171         transform = node.pv.transform
    172         boundary = node.boundary
    173         nodeIdx = node.index
    174 
    175         msg = "sc.py:add_node_gdml nodeIdx:%4d lvIdx:%2d soName:%30s lvName:%s " % (nodeIdx, lvIdx, soName, lvName )
    176         #print msg
    177 
    178         if debug:
    179             solidIdx = node.lv.solid.idx
    180             self.ulv.add(lvIdx)
    181             self.uso.add(solidIdx)
    182             assert len(self.ulv) == len(self.uso)
    183             sys.stderr.write(msg+"\n" + repr(transform)+"\n")
    184         pass
    185 
    186         nd = self.add_node( lvIdx, lvName, soName, transform, boundary, depth )


analytic/treebase.py::

    040 class Node(object):
    ...
    168     def _get_boundary(self):
    169         """
    170         ::
    171 
    172             In [23]: target.lv.material.shortname
    173             Out[23]: 'StainlessSteel'
    174 
    175             In [24]: target.parent.lv.material.shortname
    176             Out[24]: 'IwsWater'
    177 
    178 
    179         What about root volume
    180 
    181         * for actual root, the issue is mute as world boundary is not a real one
    182         * but for sub-roots maybe need use input, actually its OK as always parse 
    183           the entire GDML file
    184 
    185         """
    186         omat = 'Vacuum' if self.parent is None else self.parent.lv.material.shortname
    187         osur = ""
    188         isur = ""
    189         imat = self.lv.material.shortname
    190         return "/".join([omat,osur,isur,imat])
    191     boundary = property(_get_boundary)


* surf not imp



Contrast with G4DAE/Assimp route 
----------------------------------------

* hmm are going to need to use the G4DAE optical props anyhow... so 
  no point at moment to implement python parsing of G4DAE.  Actually 
  no point in long run of doing this either, the correct solution is 
  to add the missing info to the GDML. 

* need to find an appropriate point to ensure the GLTF and G4DAE trees
  are aligned, and then bring over the information missing ? 

  * ggeo/GScene is the likely location, its here that the G4DAE info is currently cleared 
  * perhaps having two GGeo instances (for the different routes) is the way to proceed ?
    (not so keen, seems too fundamental a change on first thought : but actually 
    when one is subbordinate it wouldnt be too disruptive)

  * hmm GScene has for the analytic route usurped a lot of what GGeo does for the triangulated

  * so the task is GGeo merging ...


* Hmm is bringing over even needed ... will need to merge GLTF 
  and G4DAE/GGeo info in the conversion to GPU geometry  



Analogous paths in the two routes
-------------------------------------

ggeo/GScene.cc::

    167 GSolid* GScene::createVolume(nd* n)
    168 {
    ...
    197 
    198     GSolid* solid = new GSolid(node_idx, gtransform, mesh, UINT_MAX, NULL );
    199 
    200     solid->setLevelTransform(ltransform);
    201 
    202     // see AssimpGGeo::convertStructureVisit
    203 
    204     solid->setSensor( NULL );
    205 
    206     solid->setCSGFlag( csg->getRootType() );
    207 
    208     solid->setCSGSkip( csg->isSkip() );
    209 
    210 
    211     // analytic spec currently missing surface info...
    212     // here need 
    213  
    214     unsigned boundary = m_bndlib->addBoundary(spec);  // only adds if not existing
    215 
    216     solid->setBoundary(boundary);     // unlike ctor these create arrays


assimprap/AssimGGeo.cc::

    0836 GSolid* AssimpGGeo::convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* /*parent*/)
     837 {
     ...
     912     GSolid* solid = new GSolid(nodeIndex, gtransform, mesh, UINT_MAX, NULL ); // sensor starts NULL
     913     solid->setLevelTransform(ltransform);
     914 
     915     const char* lv   = node->getName(0);
     916     const char* pv   = node->getName(1);
     917     const char* pv_p   = pnode->getName(1);
     918 
     919     gg->countMeshUsage(msi, nodeIndex, lv, pv);
     920 
     921     GBorderSurface* obs = gg->findBorderSurface(pv_p, pv);  // outer surface (parent->self) 
     922     GBorderSurface* ibs = gg->findBorderSurface(pv, pv_p);  // inner surface (self->parent) 
     923     GSkinSurface*   sks = gg->findSkinSurface(lv);
     924 
    ....
     998     // boundary identification via 4-uint 
     999     unsigned int boundary = blib->addBoundary(
    1000                                                mt_p->getShortName(),
    1001                                                osurf ? osurf->getShortName() : NULL ,
    1002                                                isurf ? isurf->getShortName() : NULL ,
    1003                                                mt->getShortName()
    1004                                              );
    1005 
    1006     solid->setBoundary(boundary);
    1007     {
    1008        // sensor indices are set even for non sensitive volumes in PMT viscinity
    1009        // TODO: change that 
    1010        // this is a workaround that requires an associated sensitive surface
    1011        // in order for the index to be provided
    1012 
    1013         unsigned int surface = blib->getOuterSurface(boundary);
    1014         bool oss = slib->isSensorSurface(surface);
    1015         unsigned int ssi = oss ? NSensor::RefIndex(sensor) : 0 ;
    1016         solid->setSensorSurfaceIndex( ssi );
    1017     }

    0361 void AssimpGGeo::convertMaterials(const aiScene* scene, GGeo* gg, const char* query )
     362 {
     363     LOG(info)<<"AssimpGGeo::convertMaterials "
     364              << " query " << query
     365              << " mNumMaterials " << scene->mNumMaterials
     366              ;
     367 
     368     //GDomain<float>* standard_domain = gg->getBoundaryLib()->getStandardDomain(); 
     369     GDomain<float>* standard_domain = gg->getBndLib()->getStandardDomain();
     370 
     371 
     372     for(unsigned int i = 0; i < scene->mNumMaterials; i++)
     373     {
     374         unsigned int index = i ;  // hmm, make 1-based later 
     375 
     376         aiMaterial* mat = scene->mMaterials[i] ;
     377 
     378         aiString name_;
     379         mat->Get(AI_MATKEY_NAME, name_);
     380 
     381         const char* name = name_.C_Str();
     382 
     383         //if(strncmp(query, name, strlen(query))!=0) continue ;  
     384 
     385         LOG(debug) << "AssimpGGeo::convertMaterials " << i << " " << name ;
     386 
     387         const char* bspv1 = getStringProperty(mat, g4dae_bordersurface_physvolume1 );
     388         const char* bspv2 = getStringProperty(mat, g4dae_bordersurface_physvolume2 );
     389 
     390         const char* sslv  = getStringProperty(mat, g4dae_skinsurface_volume );
     391 
     392         const char* osnam = getStringProperty(mat, g4dae_opticalsurface_name );
     393         const char* ostyp = getStringProperty(mat, g4dae_opticalsurface_type );
     394         const char* osmod = getStringProperty(mat, g4dae_opticalsurface_model );
     395         const char* osfin = getStringProperty(mat, g4dae_opticalsurface_finish );
     396         const char* osval = getStringProperty(mat, g4dae_opticalsurface_value );






