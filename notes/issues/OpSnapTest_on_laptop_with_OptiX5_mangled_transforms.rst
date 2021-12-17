OpSnapTest_on_laptop_with_OptiX5_mangled_transforms
======================================================


CAUSE FOUND
-----------------

* I was thinking that the "--gparts" option was some other control whereas in fact that is just a short form of "--gparts_transform_offset"

::

    epsilon:optickscore blyth$ OpticksTest --gp
    2021-12-17 13:31:56.667 INFO  [1167133] [main@273] OpticksTest
    option '--gp' is ambiguous and matches '--gparts_transform_offset', '--gpumon', and '--gpumonpath'

    2021-12-17 13:31:56.675 INFO  [1167133] [test_isGPartsTransformOffset@265]  is_gparts_transform_offset 0
    epsilon:optickscore blyth$ OpticksTest --gpa

    2021-12-17 13:32:01.444 INFO  [1167175] [main@273] OpticksTest
    2021-12-17 13:32:01.451 INFO  [1167175] [test_isGPartsTransformOffset@265]  is_gparts_transform_offset 1
    epsilon:optickscore blyth$ 


ADDED ABORT IN OGeo WHEN THIS BAD OPTION IS USED
--------------------------------------------------

* OGeo (as part of optixrap) is only used in the pre7 workflow 
* hence detecting that option enabled there is sufficient grounds for an abort 

::

    2021-12-17 13:45:25.963 INFO  [1286116] [OGeo::init@227]  is_gparts_transform_offset 1
    2021-12-17 13:45:25.963 FATAL [1286116] [OGeo::init@231]  using the old pre7 optixrap machinery with option --gparts_transform_offset enabled will result in mangled transforms 
    2021-12-17 13:45:25.963 FATAL [1286116] [OGeo::init@233]  the --gparts_transform_offset is only appropriate when using the new optix7 machinery, eg CSG/CSGOptiX/CSG_GGeo/.. 


Issue : the snap from OpSnapTest shows incorrect transforms being applied
---------------------------------------------------------------------------

::

    epsilon:opticks blyth$ find . -name OpSnapTest.cc
    ./okop/tests/OpSnapTest.cc



* big cylinder is lifted and rotated : presumably because "--gparts_transform_offset IS ENABLED" when it should not be 

::

    epsilon:opticks blyth$ open /tmp/blyth/opticks/snap/frame00000.jpg
    epsilon:opticks blyth$ 


::

    epsilon:opticks blyth$ OpSnapTest --gparts
    2021-12-17 12:02:17.352 INFO  [1001715] [OpticksHub::loadGeometry@283] [ /usr/local/opticks/geocache/OKX4Test_lWorld0x574e7f0_PV_g4live/g4ok_gltf/f65f5cd1a197e3a0c9fe55975ff2c7a7/1
    2021-12-17 12:02:20.130 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  1
    2021-12-17 12:02:20.130 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  2
    2021-12-17 12:02:20.131 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  3
    2021-12-17 12:02:20.131 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  4
    2021-12-17 12:02:20.131 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  5
    2021-12-17 12:02:20.131 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  6
    2021-12-17 12:02:20.131 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  7
    2021-12-17 12:02:20.131 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  8
    2021-12-17 12:02:20.132 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  9
    2021-12-17 12:02:20.303 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  1000
    2021-12-17 12:02:20.504 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  2000
    2021-12-17 12:02:20.652 INFO  [1001715] [GParts::add@1314]  --gparts_transform_offset IS ENABLED, COUNT  3000
    2021-12-17 12:02:21.045 INFO  [1001715] [OpticksHub::loadGeometry@315] ]
    2021-12-17 12:02:21.045 INFO  [1001715] [*Opticks::makeSimpleTorchStep@4346] [ts.setFrameTransform
    2021-12-17 12:02:22.336 INFO  [1001715] [OContext::CheckDevices@226] 
    Device 0                GeForce GT 750M ordinal 0 Compute Support: 3 0 Total Memory: 2147024896

    2021-12-17 12:02:22.380 INFO  [1001715] [CDevice::Dump@265] Visible devices[0:GeForce_GT_750M]
    2021-12-17 12:02:22.380 INFO  [1001715] [CDevice::Dump@269] idx/ord/mpc/cc:0/0/2/30   2.000 GB  GeForce GT 750M
    2021-12-17 12:02:22.380 INFO  [1001715] [CDevice::Dump@265] All devices[0:GeForce_GT_750M]
    2021-12-17 12:02:22.380 INFO  [1001715] [CDevice::Dump@269] idx/ord/mpc/cc:0/0/2/30   2.000 GB  GeForce GT 750M
    2021-12-17 12:02:22.566 INFO  [1001715] [*NPY<double>::MakeFloat@2032]  nv 1024
    2021-12-17 12:02:22.566 INFO  [1001715] [*NPY<double>::MakeFloat@2032]  nv 12288
    2021-12-17 12:02:22.568 INFO  [1001715] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
    2021-12-17 12:02:22.570 INFO  [1001715] [OGeo::convert@298] 
    OGeo::convert GGeoLib numMergedMesh 10 ptr 0x7fea07e22e90
    mm index   0 geocode   A                  numVolumes       3084 numFaces      183096 numITransforms           1 numITransforms*numVolumes        3084 GParts Y GPts Y
    mm index   1 geocode   A                  numVolumes          5 numFaces        1584 numITransforms       25600 numITransforms*numVolumes      128000 GParts Y GPts Y
    mm index   2 geocode   A                  numVolumes          7 numFaces        5232 numITransforms       12612 numITransforms*numVolumes       88284 GParts Y GPts Y
    mm index   3 geocode   A                  numVolumes          7 numFaces        5202 numITransforms        5000 numITransforms*numVolumes       35000 GParts Y GPts Y
    mm index   4 geocode   A                  numVolumes          6 numFaces        3284 numITransforms        2400 numITransforms*numVolumes       14400 GParts Y GPts Y
    mm index   5 geocode   A                  numVolumes          1 numFaces         528 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   6 geocode   A                  numVolumes          1 numFaces         960 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   7 geocode   A                  numVolumes          1 numFaces         384 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   8 geocode   A                  numVolumes          1 numFaces        1272 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   9 geocode   A                  numVolumes        130 numFaces        1560 numITransforms         504 numITransforms*numVolumes       65520 GParts Y GPts Y
     num_remainder_volumes 3084 num_instanced_volumes 333564 num_remainder_volumes + num_instanced_volumes 336648 num_total_faces 203102 num_total_faces_woi 143252280 (woi:without instancing) 
       0 pts Y  GPts.NumPt  3084 lvIdx ( 132 12 11 3 0 1 2 10 9 8 ... 88 88 88 88 88 120 117 118 119) 0:1 1:1 2:1 3:1 8:126 9:63 10:1 11:1 12:1 13:10 14:30 15:30 16:30 17:30 18:30 19:30 20:30 21:30 22:30 23:30 24:30 25:30 26:30 27:30 28:30 29:30 30:30 31:30 32:30 33:30 34:10 35:30 36:30 37:30 38:30 39:30 40:30 41:30 42:30 43:30 44:30 45:30 46:30 47:30 48:30 49:30 50:30 51:30 52:30 53:30 54:30 55:30 56:30 57:30 58:30 59:30 60:30 61:30 62:30 63:30 64:30 65:30 66:30 67:30 68:30 69:30 70:30 71:30 72:30 73:30 74:30 75:30 76:30 77:30 78:30 79:30 80:30 81:30 82:30 83:30 84:30 85:2 86:36 87:8 88:64 89:1 90:1 91:370 92:220 97:56 117:1 118:1 119:1 120:1 121:1 122:1 129:1 130:1 131:1 132:1
       1 pts Y  GPts.NumPt     5 lvIdx ( 116 114 112 113 115) 112 113 114 115 116 all_same_count 1
       2 pts Y  GPts.NumPt     7 lvIdx ( 104 98 99 103 102 100 101) 98 99 100 101 102 103 104 all_same_count 1
       3 pts Y  GPts.NumPt     7 lvIdx ( 111 105 106 110 109 107 108) 105 106 107 108 109 110 111 all_same_count 1
       4 pts Y  GPts.NumPt     6 lvIdx ( 128 123 127 126 124 125) 123 124 125 126 127 128 all_same_count 1
       5 pts Y  GPts.NumPt     1 lvIdx ( 93) 93 all_same_count 1
       6 pts Y  GPts.NumPt     1 lvIdx ( 94) 94 all_same_count 1
       7 pts Y  GPts.NumPt     1 lvIdx ( 95) 95 all_same_count 1
       8 pts Y  GPts.NumPt     1 lvIdx ( 96) 96 all_same_count 1
       9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4) 4:64 5:64 6:1 7:1

    2021-12-17 12:02:22.571 INFO  [1001715] [OGeo::convert@302] [ nmm 10
    2021-12-17 12:02:24.128 INFO  [1001715] [OGeo::convert@321] ] nmm 10
    2021-12-17 12:02:24.154 INFO  [1001715] [*NPY<double>::MakeFloat@2032]  nv 998432
    2021-12-17 12:02:28.975 INFO  [1001715] [Snap::render@32] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 2 ekf 6 eks 2
    2021-12-17 12:02:28.975 INFO  [1001715] [BFile::preparePath@836] created directory /tmp/blyth/opticks/snap
    2021-12-17 12:02:28.976 ERROR [1001715] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn -1 cmdline_target 0 gdmlaux_target -1 active_target 0
    2021-12-17 12:02:28.976 ERROR [1001715] [Camera::getType@319]  type 0
    2021-12-17 12:02:31.957 INFO  [1001715] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-17316.9) front 0.7071,0.7071,0.0000
    2021-12-17 12:02:39.882 INFO  [1001715] [Snap::render_one@67]  eye: -1.0000,-1.0000,0.0000,1.0000 |     0/1 dt     3.1541 |  -e ~0 | /tmp/blyth/opticks/snap/frame00000.jpg | 24
    NP::Write dtype <f8 ni        1 nj  -1 nk  -1 nl  -1 nm  -1 path $TMP/snap/frame.npy
    2021-12-17 12:02:39.972 INFO  [1001715] [Snap::render@53] ] 0
    2021-12-17 12:02:39.972 INFO  [1001715] [OTracer::report@192] OpTracer::render_snap
    2021-12-17 12:02:39.972 INFO  [1001715] [OTracer::report@195] 
     trace_count              1 trace_prep         2.98113 avg    2.98113
     trace_time         7.92496 avg    7.92496

    2021-12-17 12:02:39.972 INFO  [1001715] [OTracer::report@203] OTracer::report
                  validate000                  0.03056
                   compile000              3.99999e-06
                 prelaunch000                  4.48877
                    launch000                  3.15409
                    launchAVG                  3.15409

    2021-12-17 12:02:39.972 INFO  [1001715] [OTracer::report@208] save to /tmp/blyth/opticks/results/OpSnapTest/R0_cvd_/20211217_120217




Looking for why --gparts_transform_offset is enabled
------------------------------------------------------

::

     50 int main(int argc, char** argv)
     51 {
     52     OPTICKS_LOG(argc, argv);
     53     Opticks ok(argc, argv, "--tracer");   // tempted to put --embedded here 
     54     OpMgr op(&ok);
     55     int rc = op.render_snap();
     56     if(rc) LOG(fatal) << " rc " << rc ;
     57     return 0 ;
     58 }


okc/OpticksCfg.cc::

     235 
     236    m_desc.add_options()
     237        ("gparts_transform_offset",  "see GParts::add") ;
     238 
     239 


::

    1263 /**
    1264 GParts::add
    1265 -------------
    1266 
    1267 Basis for combination of analytic geometry.
    1268 
    1269 Notice the --gparts_transform_offset option which 
    1270 is necessary for CSG_GGeo creation of CSGFoundry as in that case the 
    1271 entire geometry is treated together. 
    1272 Without it get JUNO Chimney in middle of CD !
    1273 
    1274 Whereas for pre-7 running each GMergedMesh transforms 
    1275 are handled separately, hence --gparts_transform_offset
    1276 should not be used. 
    1277 
    1278 **/
    1279 
    1280 void GParts::add(GParts* other)
    1281 {
    1282     COUNT += 1 ;
    1283 
    1284     m_subs.push_back(other);
    1285 
    1286     if(getBndLib() == NULL)
    1287     {
    1288         setBndLib(other->getBndLib());
    1289     }
    1290     else
    1291     {
    1292         assert(getBndLib() == other->getBndLib());
    1293     }
    1294 
    1295     unsigned int n0 = getNumParts(); // before adding
    1296 
    1297     m_bndspec->add(other->getBndSpec());
    1298 
    1299 
    1300     // count the tran and plan collected so far into this GParts
    1301     unsigned tranOffset = m_tran_buffer->getNumItems();
    1302     //unsigned planOffset = m_plan_buffer->getNumItems(); 
    1303 
    1304     NPY<unsigned>* other_idx_buffer = other->getIdxBuffer() ;
    1305     NPY<float>* other_part_buffer = other->getPartBuffer()->clone() ;
    1306     NPY<float>* other_tran_buffer = other->getTranBuffer() ;
    1307     NPY<float>* other_plan_buffer = other->getPlanBuffer() ;
    1308 
    1309 
    1310     bool dump = COUNT < 10 || COUNT % 1000 == 0 ;
    1311 
    1312     if(m_ok && m_ok->isGPartsTransformOffset())  // --gparts_transform_offset
    1313     {
    1314         if(dump) LOG(info) << " --gparts_transform_offset IS ENABLED, COUNT  " << COUNT  ;
    1315 


