GPtsTest : Comparing postcache GParts(GPts) with precache GParts(NCSG)
===============================================================================

Context
----------

* :doc:`x016`

* It turns out that can avoid the CMaker problem with the back translation
  of the sFastener (asserts on left transforms of balanced trees) without using GParts.

* Neverthless the postcache GParts(GPts) seems to avoid a still mystifying bug 
  with repeated ndIdx in the precache GParts(NCSG).

  * NOW FIXED 

* Also need to measure using the GParts(GPts) approach, 
  it enables moving most of GParts creation from hot node level code 
  into cool solid level code.

  * not much time (5s) saving, but 1.3G of memory saved from deferred GParts creation 



GGeoTest with proxying is broken by move todeferred GParts : FIXED
-------------------------------------------------------------------------------

* tempted to add non-proxy GMesh/NCSG onto end of standard GMeshLib meshes/solids,  
  so proxy references can "just work" 


* FIXED by moving GGeoTest closer to standard geometry setup using GMergedMesh::Create





BUG FIXED : WAS A MISSING idxbuf CLONE IN GParts::Make : causing the repated ndIdx in GPts
------------------------------------------------------------------------------------------------

::

     439 GParts* GParts::Make( const NCSG* tree, const char* spec, unsigned ndIdx )
     440 {
     441     assert(spec);
     442 
     443     bool usedglobally = tree->isUsedGlobally() ;   // see opticks/notes/issues/subtree_instances_missing_transform.rst
     444     assert( usedglobally == true );  // always true now ?   
     445 
     446     NPY<unsigned>* tree_idxbuf = tree->getIdxBuffer() ;   // (1,4) identity indices (index,soIdx,lvIdx,height)
     447     NPY<float>*   tree_tranbuf = tree->getGTransformBuffer() ;
     448     NPY<float>*   tree_planbuf = tree->getPlaneBuffer() ;
     449     assert( tree_tranbuf );
     450 
     451     NPY<unsigned>* idxbuf = tree_idxbuf->clone()  ;   // <-- lacking this clone was cause of the mystifying repeated indices see notes/issues/GPtsTest             
     452     NPY<float>* nodebuf = tree->getNodeBuffer();       // serialized binary tree
     453     NPY<float>* tranbuf = usedglobally                 ? tree_tranbuf->clone() : tree_tranbuf ;
     454     NPY<float>* planbuf = usedglobally && tree_planbuf ? tree_planbuf->clone() : tree_planbuf ;
     455 
     456     
     457     // overwrite the cloned idxbuf swapping the tree index for the ndIdx 
     458     // as being promoted to node level 
     459     {
     460         assert( idxbuf->getNumItems() == 1 );
     461         unsigned i=0u ;
     462         unsigned j=0u ;
     463         unsigned k=0u ;
     464         unsigned l=0u ;
     465         idxbuf->setUInt(i,j,k,l, ndIdx);
     466     }



Fix confirmed by GPtsTest::

    [blyth@localhost issues]$ GPtsTest 
    2019-06-16 21:17:18.310 INFO  [409948] [Opticks::init@312] INTEROP_MODE
    2019-06-16 21:17:18.311 FATAL [409948] [Opticks::configure@1732]  --interop mode with no cvd specified, adopting OPTICKS_DEFAULT_INTEROP_CVD hinted by envvar [1]
    2019-06-16 21:17:18.311 INFO  [409948] [Opticks::configure@1739]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
    2019-06-16 21:17:18.318 INFO  [409948] [BOpticksResource::setupViaKey@531] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
                     exename  : OKX4Test
             current_exename  : GPtsTest
                       class  : X4PhysicalVolume
                     volname  : lWorld0x4bc2710_PV
                      digest  : f6cc352e44243f8fa536ab483ad390ce
                      idname  : OKX4Test_lWorld0x4bc2710_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2019-06-16 21:17:18.318 ERROR [409948] [OpticksResource::initRunResultsDir@260] /home/blyth/local/opticks/results/GPtsTest/R0_cvd_1/20190616_211718
    2019-06-16 21:17:18.417 INFO  [409948] [GMeshLib::loadMeshes@434]  loaded  meshes 41 solids 41
    2019-06-16 21:17:18.418 INFO  [409948] [GMeshLib::loadAltReferences@163]  mesh.i 16 altindex 40
    2019-06-16 21:17:18.418 INFO  [409948] [GMeshLib::loadAltReferences@163]  mesh.i 40 altindex 16
    2019-06-16 21:17:18.545 INFO  [409948] [main@122]  geolib.nmm 6
    2019-06-16 21:17:18.554 INFO  [409948] [testGPts::compare@88]  mm.index 0 meshlib.solids 41 RC 0
    2019-06-16 21:17:18.554 INFO  [409948] [testGPts::save@78] /tmp/blyth/location/GGeo/GPtsTest/0
    2019-06-16 21:17:18.556 INFO  [409948] [testGPts::compare@88]  mm.index 1 meshlib.solids 41 RC 0
    2019-06-16 21:17:18.556 INFO  [409948] [testGPts::save@78] /tmp/blyth/location/GGeo/GPtsTest/1
    2019-06-16 21:17:18.558 INFO  [409948] [testGPts::compare@88]  mm.index 2 meshlib.solids 41 RC 0
    2019-06-16 21:17:18.558 INFO  [409948] [testGPts::save@78] /tmp/blyth/location/GGeo/GPtsTest/2
    2019-06-16 21:17:18.564 INFO  [409948] [testGPts::compare@88]  mm.index 3 meshlib.solids 41 RC 0
    2019-06-16 21:17:18.564 INFO  [409948] [testGPts::save@78] /tmp/blyth/location/GGeo/GPtsTest/3
    2019-06-16 21:17:18.565 INFO  [409948] [testGPts::compare@88]  mm.index 4 meshlib.solids 41 RC 0
    2019-06-16 21:17:18.565 INFO  [409948] [testGPts::save@78] /tmp/blyth/location/GGeo/GPtsTest/4
    2019-06-16 21:17:18.567 INFO  [409948] [testGPts::compare@88]  mm.index 5 meshlib.solids 41 RC 0
    2019-06-16 21:17:18.567 INFO  [409948] [testGPts::save@78] /tmp/blyth/location/GGeo/GPtsTest/5
    [blyth@localhost issues]$ 







mm 0 : idx again 
----------------------

::

    [blyth@localhost ggeo]$ GPtsTest 
    2019-06-16 15:45:50.628 INFO  [270405] [BOpticksResource::setupViaKey@531] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
                     exename  : OKX4Test
             current_exename  : GPtsTest
                       class  : X4PhysicalVolume
                     volname  : lWorld0x4bc2710_PV
                      digest  : f6cc352e44243f8fa536ab483ad390ce
                      idname  : OKX4Test_lWorld0x4bc2710_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2019-06-16 15:45:50.748 INFO  [270405] [GMeshLib::loadMeshes@434]  loaded  meshes 41 solids 41
    2019-06-16 15:45:50.748 INFO  [270405] [GMeshLib::loadAltReferences@163]  mesh.i 16 altindex 40
    2019-06-16 15:45:50.748 INFO  [270405] [GMeshLib::loadAltReferences@163]  mesh.i 40 altindex 16
    2019-06-16 15:45:50.880 INFO  [270405] [main@119]  geolib.nmm 6
    2019-06-16 15:45:50.886 INFO  [270405] [testGPts::init@60]  mm.index 0 meshlib.solids 41
    2019-06-16 15:45:50.887 INFO  [270405] [GParts::Compare@60] 
                                   qty                                 A                                 B
                           VolumeIndex                                 0                                 0
                                  Name                              NULL                              NULL
                                BndLib                         0x28a7b00                         0x28a7b00
                                Closed                                 0                                 1
                                Loaded                                 1                                 0
                        PrimFlagString                      flagnodetree                      flagnodetree
                              NumParts                               219                               219
                               NumPrim                               201                               201
                                  prim  5c3c1051b34dc5116a9208e3556263de  5c3c1051b34dc5116a9208e3556263de
                                   idx  e8dbf2bf4b8dba375b4d4f67d5cc4d27  b5a18d9a7bc6c00bc9ab1d27f6c20299  <<<<<<<<<<
                                  part  faf26502e310670d13dfe100725b43c6  faf26502e310670d13dfe100725b43c6
                                  tran  344a04b61961356c74f63f71d3f3ed4c  344a04b61961356c74f63f71d3f3ed4c
                                  plan  d41d8cd98f00b204e9800998ecf8427e  d41d8cd98f00b204e9800998ecf8427e
    2019-06-16 15:45:50.887 INFO  [270405] [testGPts::compare@86]  RC 1
    2019-06-16 15:45:50.887 INFO  [270405] [testGPts::save@78] $TMP/GGeo/GPtsTest
    [blyth@localhost ggeo]$ 


::

    [blyth@localhost GPtsTest]$ np.py parts
    /tmp/blyth/location/GGeo/GPtsTest/parts
    . :                                             parts/GParts.txt :                  219 : 5672a006d9e7cdd0a860260fe66811ce : 20190616-1545 
    . :                                          parts/idxBuffer.npy :             (201, 4) : 3e6f8c55de891e502afb5ac6c94ff0d0 : 20190616-1545 
    . :                                         parts/partBuffer.npy :          (219, 4, 4) : 40330b525562dbd866103ed81c9fe8bf : 20190616-1545 
    . :                                         parts/primBuffer.npy :             (201, 4) : cd3222aea6b1292bd3382340b61e1d62 : 20190616-1545 
    . :                                         parts/tranBuffer.npy :       (206, 3, 4, 4) : d06f209a19c0d85d84ccac15c501d676 : 20190616-1545 
    [blyth@localhost GPtsTest]$ np.py parts2
    /tmp/blyth/location/GGeo/GPtsTest/parts2
    . :                                            parts2/GParts.txt :                  219 : 5672a006d9e7cdd0a860260fe66811ce : 20190616-1545 
    . :                                         parts2/idxBuffer.npy :             (201, 4) : a11a7fb1c63f32b0442eae4c1c40ee8e : 20190616-1545 
    . :                                        parts2/partBuffer.npy :          (219, 4, 4) : 40330b525562dbd866103ed81c9fe8bf : 20190616-1545 
    . :                                        parts2/primBuffer.npy :             (201, 4) : cd3222aea6b1292bd3382340b61e1d62 : 20190616-1545 
    . :                                        parts2/tranBuffer.npy :       (206, 3, 4, 4) : d06f209a19c0d85d84ccac15c501d676 : 20190616-1545 
    [blyth@localhost GPtsTest]$ 



::

    In [1]: a = np.load("parts/idxBuffer.npy")

    In [2]: b = np.load("parts2/idxBuffer.npy")


    In [10]: np.all(a[:,1] == b[:,1]) 
    Out[10]: True

    In [11]: np.all(a[:,3] == b[:,3]) 
    Out[11]: True

    In [12]: np.all(a[:,2] == b[:,2]) 
    Out[12]: True


Looks like preorder vs postorder indexing difference ?::

    In [17]: a[-20:,0]
    Out[17]: 
    array([ 62067,  61545,  62067,  62067,  61545,  62067,  62067,  62588,
            62589,  62590,  62591,  62592,  62593,  62594, 352849, 352850,
           352851, 352852, 352853, 352854], dtype=uint32)

    In [18]: b[-20:,0]
    Out[18]: 
    array([ 59981,  60502,  60503,  61024,  61545,  61546,  62067,  62588,
            62589,  62590,  62591,  62592,  62593,  62594, 352849, 352850,
           352851, 352852, 352853, 352854], dtype=uint32)

    In [19]: a[:20,0]
    Out[19]: 
    array([    0,     1,     2,     3,     4,     5,     6,     7, 61545,
           62067, 62067, 61545, 62067, 62067, 61545, 62067, 62067, 61545,
           62067, 62067], dtype=uint32)

    In [20]: b[:20,0]
    Out[20]: 
    array([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,  530,
           1051, 1052, 1573, 2094, 2095, 2616, 3137, 3138, 3659], dtype=uint32)



mm3 : volIdx also
----------------------

* not expecting to see repetition ? must be a bug 

::

    In [20]: np.unique(a[:,0])
    Out[20]: array([62458, 62459, 62586, 62587], dtype=uint32)

    In [21]: np.unique(b[:,0])
    Out[21]: 
    array([62458, 62459, 62460, 62461, 62462, 62463, 62464, 62465, 62466,
           62467, 62468, 62469, 62470, 62471, 62472, 62473, 62474, 62475,
           62476, 62477, 62478, 62479, 62480, 62481, 62482, 62483, 62484,
           62485, 62486, 62487, 62488, 62489, 62490, 62491, 62492, 62493,
           62494, 62495, 62496, 62497, 62498, 62499, 62500, 62501, 62502,
           62503, 62504, 62505, 62506, 62507, 62508, 62509, 62510, 62511,
           62512, 62513, 62514, 62515, 62516, 62517, 62518, 62519, 62520,
           62521, 62522, 62523, 62524, 62525, 62526, 62527, 62528, 62529,
           62530, 62531, 62532, 62533, 62534, 62535, 62536, 62537, 62538,
           62539, 62540, 62541, 62542, 62543, 62544, 62545, 62546, 62547,
           62548, 62549, 62550, 62551, 62552, 62553, 62554, 62555, 62556,
           62557, 62558, 62559, 62560, 62561, 62562, 62563, 62564, 62565,
           62566, 62567, 62568, 62569, 62570, 62571, 62572, 62573, 62574,
           62575, 62576, 62577, 62578, 62579, 62580, 62581, 62582, 62583,
           62584, 62585, 62586, 62587], dtype=uint32)

    In [8]: len(np.unique(b[:,0]))
    Out[8]: 130


::

    ipython $(which GPtsTest.py) -i -- 3

    /tmp/blyth/location/GGeo/GPtsTest/3
    A:(130, 4) /tmp/blyth/location/GGeo/GPtsTest/3/parts/idxBuffer.npy au:4
    B:(130, 4) /tmp/blyth/location/GGeo/GPtsTest/3/parts2/idxBuffer.npy bu:130


    In [2]: np.unique(a[:,0])
    Out[2]: array([62458, 62459, 62586, 62587], dtype=uint32)

    In [3]: np.where( a[:,0] == 62458 )
    Out[3]: (array([0]),)

    In [5]: np.where( a[:,0] == 62459 )
    Out[5]: (array([1]),)

    In [6]: np.where( a[:,0] == 62586 )
    Out[6]: 
    (array([  2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,
             28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,
             54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,
             80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100, 102, 104,
            106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128]),)

    In [7]: np.where( a[:,0] == 62587 )
    Out[7]: 
    (array([  3,   5,   7,   9,  11,  13,  15,  17,  19,  21,  23,  25,  27,
             29,  31,  33,  35,  37,  39,  41,  43,  45,  47,  49,  51,  53,
             55,  57,  59,  61,  63,  65,  67,  69,  71,  73,  75,  77,  79,
             81,  83,  85,  87,  89,  91,  93,  95,  97,  99, 101, 103, 105,
            107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129]),)





sFastener mm5
------------------

* for sFastener good match apart from getVolumeIndex(0)
* FIXED by somewhat unsatisfactorily changing to use GInstancer::getLastRepeatExample

::

    2019-06-16 13:59:39.338 INFO  [58250] [GParts::Compare@57] 
                                   qty                                 A                                 B
                           VolumeIndex                             63554                             63075   <<<<<<<<<<<<<<<<<<<<
                                  Name                              NULL                              NULL
                                BndLib                         0x11acb00                         0x11acb00
                                Closed                                 1                                 1
                                Loaded                                 1                                 0
                        PrimFlagString                      flagnodetree                      flagnodetree
                              NumParts                                31                                31
                               NumPrim                                 1                                 1
                                  prim  e1a7612fac70b684990129fedf3b8ce7  e1a7612fac70b684990129fedf3b8ce7
                                   idx  614ed949cf70802dab93ed8fb14578f6  535302f0401f8d763415925ce5b3acc1
                                  part  c9150c5e22f758aded28d128c69912da  c9150c5e22f758aded28d128c69912da
                                  tran  e95415b9ce12a474595951b44d131b0d  e95415b9ce12a474595951b44d131b0d
                                  plan  d41d8cd98f00b204e9800998ecf8427e  d41d8cd98f00b204e9800998ecf8427e


Get discrepant idxBuffer::


    [blyth@localhost GPtsTest]$ np.py parts/idxBuffer.npy -iFv
    a :                                          parts/idxBuffer.npy :               (1, 4) : d988f81268f2555ea952d45d32060d08 : 20190615-2344 
    (1, 4)
    i32
    [[[63554    16    16     4]]]

    [blyth@localhost GPtsTest]$ np.py parts2/idxBuffer.npy -iFv
    a :                                         parts2/idxBuffer.npy :               (1, 4) : 000516c4738eac9ef5392eaa5fafe0f0 : 20190615-2344 
    (1, 4)
    i32
    [[[63075    16    16     4]]]


Dumping the indices those are the last and first node indices of lvIdx 16::

    2019-06-16 10:11:23.151 INFO  [109830] [X4PhysicalVolume::convertSolids@450] ]
    2019-06-16 10:11:23.151 INFO  [109830] [X4PhysicalVolume::convertStructure@722] [ creating large tree of GVolume instances
    2019-06-16 10:11:27.266 INFO  [109830] [X4PhysicalVolume::convertNode@980]  lvIdx 16 ndIdx 63075 csgIdx 16 boundaryName Water///Copper
    2019-06-16 10:11:27.266 INFO  [109830] [X4PhysicalVolume::convertNode@980]  lvIdx 16 ndIdx 63076 csgIdx 16 boundaryName Water///Copper
    ...
    2019-06-16 10:11:27.331 INFO  [109830] [X4PhysicalVolume::convertNode@980]  lvIdx 16 ndIdx 63553 csgIdx 16 boundaryName Water///Copper
    2019-06-16 10:11:27.331 INFO  [109830] [X4PhysicalVolume::convertNode@980]  lvIdx 16 ndIdx 63554 csgIdx 16 boundaryName Water///Copper
    2019-06-16 10:11:48.435 INFO  [109830] [X4PhysicalVolume::convertStructure@742] ] tree contains GGeo::getNumVolumes() 366697
    2019-06-16 10:11:48.435 INFO  [109830] [GGeo::prepare@672] [


* immediate GParts(NCSG) gets the last ndIdx : 63554
* deferred GParts(GPts) gets the first ndIdx : 63075 

* there are 480 GVolume with lvIdx 16, each with an associated GPt 
* GInstancer picks apparently the last 
* GPts from GGeoLib merged mesh used by GParts::Create is the first 

Have an inkling this is due to the GInstancer ridx node selection, which for repeated nodes just makes the mesh for the first.


Huh, but here it looks like the first::

:

    2019-06-16 14:16:24.997 INFO  [89548] [GInstancer::getRepeatExample@540]  ridx 5
     first.pt  lvIdx   16 ndIdx   63075 csgIdx      16 spec                 Water///Copper placement Id
     last.pt   lvIdx   16 ndIdx   63554 csgIdx      16 spec                 Water///Copper placement Id
    2019-06-16 14:16:24.997 INFO  [89548] [GMergedMesh::Create@239]  ridx 5 starting from lFasteners_phys0x4c01450
    2019-06-16 14:16:24.998 INFO  [89548] [GMergedMesh::mergeVolume@504]  m_cur_volume 1 parts.getVolumeIndex(0) 63554 selected YES pt  lvIdx   16 ndIdx   63075 csgIdx      16 spec                 Water///Copper placement Id
    2019-06-16 14:16:24.998 INFO  [89548] [GMergedMesh::mergeVolumeAnalytic@811]  lvIdx   16 ndIdx   63075 csgIdx      16 spec                 Water///Copper placement Id
    2019-06-16 14:16:24.998 ERROR [89548] [GGeoLib::makeMergedMesh@280] mm index   5 geocode   T                  numVolumes          1 numFaces        1856 numITransforms           0 numITransforms*numVolumes           0
    2019-06-16 14:16:25.076 INFO  [89548] [GInstancer::dump@676] GGeo::prepareVolumes


::

    2019-06-16 14:50:18.843 INFO  [161890] [GParts::Compare@57] 
                                   qty                                 A                                 B
                           VolumeIndex                             63554                             63554
                                  Name                              NULL                              NULL
                                BndLib                         0x10ccb00                         0x10ccb00
                                Closed                                 1                                 1
                                Loaded                                 1                                 0
                        PrimFlagString                      flagnodetree                      flagnodetree
                              NumParts                                31                                31
                               NumPrim                                 1                                 1
                                  prim  e1a7612fac70b684990129fedf3b8ce7  e1a7612fac70b684990129fedf3b8ce7
                                   idx  614ed949cf70802dab93ed8fb14578f6  614ed949cf70802dab93ed8fb14578f6
                                  part  c9150c5e22f758aded28d128c69912da  c9150c5e22f758aded28d128c69912da
                                  tran  e95415b9ce12a474595951b44d131b0d  e95415b9ce12a474595951b44d131b0d
                                  plan  d41d8cd98f00b204e9800998ecf8427e  d41d8cd98f00b204e9800998ecf8427e


Somewhat unsatisfactory solution is to add *GInstancer::getLastRepeatExample* and use that::


    531 GNode* GInstancer::getRepeatExample(unsigned ridx)
    532 {
    533     std::vector<GNode*> placements = getPlacements(ridx);
    534     std::string pdig = m_repeat_candidates[ridx-1];
    535     GNode* node = m_root->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    536     assert(placements[0] == node);
    537 
    538     GVolume* first = static_cast<GVolume*>(placements.front()) ;
    539     GVolume* last = static_cast<GVolume*>(placements.back()) ;
    540 
    541     LOG(info)
    542         << " ridx " << ridx
    543         << std::endl
    544         << " first.pt " << first->getPt()->desc()
    545         << std::endl
    546         << " last.pt  " << last->getPt()->desc()
    547         ;
    548 
    549     return node ; 
    550 }
    551 
    552 GNode* GInstancer::getLastRepeatExample(unsigned ridx)
    553 {    
    554     std::vector<GNode*> placements = getPlacements(ridx);
    555     std::string pdig = m_repeat_candidates[ridx-1];
    556     GNode* node = m_root->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    557     assert(placements[0] == node);
    558     return placements.back() ;
    559 }    
    560 



::

    551 void GInstancer::makeMergedMeshAndInstancedBuffers(unsigned verbosity)
    552 {
    553 
    554     GNode* root = m_nodelib->getNode(0);
    555     assert(root);
    556     GNode* base = NULL ;
    557 
    558 
    559     // passes thru to GMergedMesh::create with management of the mm in GGeoLib
    560     GMergedMesh* mm0 = m_geolib->makeMergedMesh(0, base, root, verbosity );
    561 
    562 
    563     std::vector<GNode*> placements = getPlacements(0);  // just m_root
    564     assert(placements.size() == 1 );
    565     mm0->addInstancedBuffers(placements);  // call for global for common structure 
    566 
    567 
    568     unsigned numRepeats = getNumRepeats();
    569     unsigned numRidx = numRepeats + 1 ;
    570 
    571     LOG(info)
    572         << " numRepeats " << numRepeats
    573         << " numRidx " << numRidx
    574         ;
    575 
    576     for(unsigned ridx=1 ; ridx < numRidx ; ridx++)  // 1-based index
    577     {
    578          GNode*   rbase  = getRepeatExample(ridx) ;    // <--- why not the parent ? off-by-one confusion here as to which transforms to include
    ///
    ///     result of GNode::findProgenyDigest    
    ///
    579 
    580          if(m_verbosity > 2)
    581          LOG(info)
    582              << " ridx " << ridx  
    583              << " rbase " << rbase 
    584              ;
    585 
    586          GMergedMesh* mm = m_geolib->makeMergedMesh(ridx, rbase, root, verbosity );
    587 
    588          std::vector<GNode*> placements_ = getPlacements(ridx);
    589 
    590          mm->addInstancedBuffers(placements_);
    591     
    592          //mm->reportMeshUsage( ggeo, "GInstancer::CreateInstancedMergedMeshes reportMeshUsage (instanced)");
    593     }
    594 }


    533 GNode* GInstancer::getRepeatExample(unsigned ridx)
    534 {
    535     std::vector<GNode*> placements = getPlacements(ridx);
    536     std::string pdig = m_repeat_candidates[ridx-1];
    537     GNode* node = m_root->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    538     assert(placements[0] == node);
    539     return node ;
    540 }
    541 


For instanced the traversal starts from base node, so for sFastener this is picking which of the 480::

    219 GMergedMesh* GMergedMesh::Create(unsigned ridx, GNode* base, GNode* root, unsigned verbosity ) // static
    220 {
    221     assert(root && "root node is required");
    222 
    223     LOG(LEVEL)
    224         << " ridx " << ridx
    225         << " base " << base
    226         << " root " << root
    227         << " verbosity " << verbosity
    228         ;
    229 
    230 
    231     OKI_PROFILE("_GMergedMesh::Create");
    232 
    233     GMergedMesh* mm = new GMergedMesh( ridx );
    234     mm->setCurrentBase(base);  // <-- when NULL it means will use global not base relative transforms
    235 
    236     GNode* start = base ? base : root ;
    237 
    238     //if(verbosity > 1)
    239     LOG(LEVEL)
    240         << " ridx " << ridx
    241         << " starting from " << start->getName() ;
    242         ;
    243 
    244     mm->traverse_r( start, 0, PASS_COUNT, verbosity  );  // 1st pass traversal : counts vertices and faces
    245 















