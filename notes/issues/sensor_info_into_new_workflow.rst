sensor_info_into_new_workflow
===============================

* from :doc:`update_juno_opticks_integration_for_new_workflow`
* see also :doc:`instanceIdentity-into-new-workflow`



Comparing cf.inst with stree f.inst_f4 using ntds3/G4CXOpticks saved geometry
--------------------------------------------------------------------------------

::

    cd ~/opticks/sysrap/tests
    ./stree_test.sh 

    010 export STBASE=/tmp/$USER/opticks/ntds3/G4CXOpticks
     14 export FOLD=$STBASE/stree
     15 export CFBASE=$STBASE


::

    ./stree_test.sh ana

    In [6]: cf.inst.shape
    Out[6]: (48477, 4, 4)

    In [8]: f.inst_f4.shape
    Out[8]: (48477, 4, 4)

    In [7]: cf.inst.view(np.int32)[:,:,3]
    Out[7]: 
    array([[     0,      0,     -1,     -1],
           [     1,      1, 300000,  17613],
           [     2,      1, 300001,  17614],
           [     3,      1, 300002,  17615],
           [     4,      1, 300003,  17616],
           ...,
           [ 48472,      9,     -1,     -1],
           [ 48473,      9,     -1,     -1],
           [ 48474,      9,     -1,     -1],
           [ 48475,      9,     -1,     -1],
           [ 48476,      9,     -1,     -1]], dtype=int32)


    In [9]: f.inst_f4.view(np.int32)[:,:,3]
    Out[9]: 
    array([[     1,      1,     -1,     -1],
           [     2,      2, 300000,  17612],
           [     3,      2, 300001,  17613],
           [     4,      2, 300002,  17614],
           [     5,      2, 300003,  17615],
           ...,
           [ 48473,     10,     -1,     -1],
           [ 48474,     10,     -1,     -1],
           [ 48475,     10,     -1,     -1],
           [ 48476,     10,     -1,     -1],
           [ 48477,     10,     -1,     -1]], dtype=int32)


Column 3, Row 0,1, (inst_idx,gas_idx) are +1 in f.inst_f4/strid.h::

    In [10]: f.inst_f4.view(np.int32)[:,0,3]
    Out[10]: array([    1,     2,     3,     4,     5, ..., 48473, 48474, 48475, 48476, 48477], dtype=int32)

    In [11]: cf.inst.view(np.int32)[:,0,3]
    Out[11]: array([    0,     1,     2,     3,     4, ..., 48472, 48473, 48474, 48475, 48476], dtype=int32)

    In [12]: np.all( cf.inst.view(np.int32)[:,0,3]  + 1 == f.inst_f4.view(np.int32)[:,0,3]  )
    Out[12]: True

    In [13]: f.inst_f4.view(np.int32)[:,1,3]
    Out[13]: array([ 1,  2,  2,  2,  2, ..., 10, 10, 10, 10, 10], dtype=int32)

    In [14]: cf.inst.view(np.int32)[:,1,3]
    Out[14]: array([0, 1, 1, 1, 1, ..., 9, 9, 9, 9, 9], dtype=int32)

    In [15]: np.all( cf.inst.view(np.int32)[:,1,3] + 1 == f.inst_f4.view(np.int32)[:,1,3] )
    Out[15]: True


HMM strid.h does not do the +1 its done in strid::add_inst::

    1315 inline void stree::add_inst( glm::tmat4x4<double>& tr_m2w,  glm::tmat4x4<double>& tr_w2m, unsigned gas_idx, int nidx )
    1316 {
    1317     assert( nidx > -1 && nidx < int(nds.size()) );
    1318     const snode& nd = nds[nidx];
    1319 
    1320 
    1321     unsigned ins_idx = inst.size();     // follow sqat4.h::setIdentity
    1322     //unsigned ias_idx = 0 ; 
    1323 
    1324     glm::tvec4<uint64_t> col3 ;
    1325     col3.x = ins_idx + 1 ;
    1326     col3.y = gas_idx + 1 ;
    1327     //col3.z = ias_idx + 1 ; 
    1328     col3.z = nd.sensor_id ;
    1329     col3.w = nd.sensor_index ;
    1330 
    1331     strid::Encode(tr_m2w, col3 );
    1332     strid::Encode(tr_w2m, col3 );
    1333 
    1334     inst.push_back(tr_m2w);
    1335     iinst.push_back(tr_w2m);
    1336 
    1337 }


    1307 /**
    1308 stree::add_inst
    1309 ----------------
    1310 
    1311 Canonically invoked from U4Tree::Create 
    1312 
    1313 **/
    1314 
    1315 inline void stree::add_inst( glm::tmat4x4<double>& tr_m2w,  glm::tmat4x4<double>& tr_w2m, int gas_idx, int nidx )
    1316 {
    1317     assert( nidx > -1 && nidx < int(nds.size()) );
    1318     const snode& nd = nds[nidx];
    1319 
    1320     int ins_idx = int(inst.size());     // follow sqat4.h::setIdentity
    1321 
    1322     glm::tvec4<int64_t> col3 ;   // formerly uint64_t 
    1323     col3.x = ins_idx ;            // formerly  +1 
    1324     col3.y = gas_idx ;            // formerly  +1 
    1325     col3.z = nd.sensor_id ;       // formerly ias_idx + 1 (which was always 1)
    1326     col3.w = nd.sensor_index ;
    1327 
    1328     strid::Encode(tr_m2w, col3 );
    1329     strid::Encode(tr_w2m, col3 );
    1330 
    1331     inst.push_back(tr_m2w);
    1332     iinst.push_back(tr_w2m);
    1333  
    1334 }





Column 3, Row 2 (sensor_identifier) matches::

    In [16]: f.inst_f4.view(np.int32)[:,2,3]
    Out[16]: array([    -1, 300000, 300001, 300002, 300003, ...,     -1,     -1,     -1,     -1,     -1], dtype=int32)

    In [17]: cf.inst.view(np.int32)[:,2,3]
    Out[17]: array([    -1, 300000, 300001, 300002, 300003, ...,     -1,     -1,     -1,     -1,     -1], dtype=int32)

    In [18]: np.all( f.inst_f4.view(np.int32)[:,2,3]  == cf.inst.view(np.int32)[:,2,3] )
    Out[18]: True


Column 3, Row 3 (sensor_index) is curiously mixed up.

The not-a-sensor -1 are matched::

    In [19]: f.inst_f4.view(np.int32)[:,3,3]
    Out[19]: array([   -1, 17612, 17613, 17614, 17615, ...,    -1,    -1,    -1,    -1,    -1], dtype=int32)

    In [20]: cf.inst.view(np.int32)[:,3,3]
    Out[20]: array([   -1, 17613, 17614, 17615, 17616, ...,    -1,    -1,    -1,    -1,    -1], dtype=int32)

    In [21]: np.where( f.inst_f4.view(np.int32)[:,3,3] == -1 )
    Out[21]: (array([    0, 45613, 45614, 45615, 45616, ..., 48472, 48473, 48474, 48475, 48476]),)

    In [22]: np.where( cf.inst.view(np.int32)[:,3,3] == -1 )
    Out[22]: (array([    0, 45613, 45614, 45615, 45616, ..., 48472, 48473, 48474, 48475, 48476]),)

    In [23]: np.all( np.where( f.inst_f4.view(np.int32)[:,3,3] == -1 )[0] == np.where( cf.inst.view(np.int32)[:,3,3] == -1 )[0] )
    Out[23]: True

The sensor_index are off-by-1, but this time its cf.inst that is +1 unlike the above case::

    In [24]: w = np.where( f.inst_f4.view(np.int32)[:,3,3] > -1 )[0]

    In [25]: f.inst_f4.view(np.int32)[w,3,3]
    Out[25]: array([17612, 17613, 17614, 17615, 17616, ..., 45607, 45608, 45609, 45610, 45611], dtype=int32)

    In [26]: cf.inst.view(np.int32)[w,3,3]
    Out[26]: array([17613, 17614, 17615, 17616, 17617, ..., 45608, 45609, 45610, 45611, 45612], dtype=int32)

    In [27]: np.all( f.inst_f4.view(np.int32)[w,3,3] + 1 == cf.inst.view(np.int32)[w,3,3]  )
    Out[27]: True







Get the expected id ranges when realize that the sensor_index is 1-based
----------------------------------------------------------------------------

The below is teleporting in the sensor_id::

     71     NP* sensor_id = NP::Load("/tmp/blyth/opticks/ntds3/G4CXOpticks/stree_reorderSensors/sensor_id.npy") ;
     72     const int* sid = sensor_id->cvalues<int>();
     73     unsigned num_sid = sensor_id->shape[0] ;

DONE: now grab that from the stree. 


::

     cd ~/opticks/GGeo/tests
     ./GGeoLoadFromDirTest.sh 

     ridx   0 mm 0x7f9f8e408d00 num_inst       1 iid        1,3089,4 sensor_index       1 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
     ridx   1 mm 0x7f9f8e486360 num_inst   25600 iid       25600,5,4 sensor_index   25600 idx_mn   17613 idx_mx   43212 id_mn  300000 id_mx  325599
     ridx   2 mm 0x7f9f8e489d70 num_inst   12615 iid       12615,7,4 sensor_index   12615 idx_mn       3 idx_mx   17591 id_mn       2 id_mx   17590
     ridx   3 mm 0x7f9f8e735360 num_inst    4997 iid        4997,7,4 sensor_index    4997 idx_mn       1 idx_mx   17612 id_mn       0 id_mx   17611
     ridx   4 mm 0x7f9f8e739340 num_inst    2400 iid        2400,6,4 sensor_index    2400 idx_mn   43213 idx_mx   45612 id_mn   30000 id_mx   32399
     ridx   5 mm 0x7f9f8e73cff0 num_inst     590 iid         590,1,4 sensor_index     590 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
     ridx   6 mm 0x7f9f8e73fd30 num_inst     590 iid         590,1,4 sensor_index     590 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
     ridx   7 mm 0x7f9f8e742af0 num_inst     590 iid         590,1,4 sensor_index     590 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
     ridx   8 mm 0x7f9f8e502de0 num_inst     590 iid         590,1,4 sensor_index     590 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
     ridx   9 mm 0x7f9f8e5067f0 num_inst     504 iid       504,130,4 sensor_index     504 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
    epsilon:tests blyth$ 



Not the expected id ranges
----------------------------

::

    2022-08-13 18:03:06.643 INFO  [27510975] [main@68]  ggeo 0x7fb3f55222b0 nmm 10 ridx -1
     ridx   0 mm 0x7fb3f54147f0 num_inst       1 iid        1,3089,4 sensor_index       1 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
     ridx   1 mm 0x7fb3f81263d0 num_inst   25600 iid       25600,5,4 sensor_index   25600 idx_mn   17613 idx_mx   43212 id_mn   30000 id_mx  325599
     ridx   2 mm 0x7fb3f8129e00 num_inst   12615 iid       12615,7,4 sensor_index   12615 idx_mn       3 idx_mx   17591 id_mn       3 id_mx   17591
     ridx   3 mm 0x7fb3f812db90 num_inst    4997 iid        4997,7,4 sensor_index    4997 idx_mn       1 idx_mx   17612 id_mn       1 id_mx  300000
     ridx   4 mm 0x7fb3f5418420 num_inst    2400 iid        2400,6,4 sensor_index    2400 idx_mn   43213 idx_mx   45612 id_mn       0 id_mx   32399
     ridx   5 mm 0x7fb3f541c0a0 num_inst     590 iid         590,1,4 sensor_index     590 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
     ridx   6 mm 0x7fb3f541edb0 num_inst     590 iid         590,1,4 sensor_index     590 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
     ridx   7 mm 0x7fb3f5421ba0 num_inst     590 iid         590,1,4 sensor_index     590 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
     ridx   8 mm 0x7fb3f54249b0 num_inst     590 iid         590,1,4 sensor_index     590 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
     ridx   9 mm 0x7fb3f54283c0 num_inst     504 iid       504,130,4 sensor_index     504 idx_mn      -1 idx_mx      -1 id_mn      -1 id_mx      -1
    epsilon:tests blyth$ vi GGeoLoadFromDirTest.cc

::

    In [1]: a = np.load("/tmp/blyth/opticks/ntds3/G4CXOpticks/stree_reorderSensors/sensor_id.npy")

    In [2]: a 
    Out[2]: array([    0,     1,     2, ..., 32397, 32398, 32399], dtype=int32)

    In [3]: np.where( np.diff(a) != 1 )
    Out[3]: (array([17611, 43211]),)

    In [4]: a[0:17612]
    Out[4]: array([    0,     1,     2, ..., 17609, 17610, 17611], dtype=int32)

    In [6]: np.all( a[0:17612] == np.arange(17612) )
    Out[6]: True

    In [11]: a[17612:43212+1]
    Out[11]: array([300000, 300001, 300002, ..., 325598, 325599,  30000], dtype=int32)

    In [15]: a[43212:43212+2400]
    Out[15]: array([30000, 30001, 30002, ..., 32397, 32398, 32399], dtype=int32)

    In [16]: a[43212:43212+2400+1]
    Out[16]: array([30000, 30001, 30002, ..., 32397, 32398, 32399], dtype=int32)


* expected three species of identifiers





Transitional issue wrt sensor_id 
-----------------------------------

The old G4Opticks workflow relied on additional calls 
to set the sensor_id given the sensor placement vector. 

New workflow does away with the need for this API for sensor_id 
using U4SensorIdentifier and U4SensorIdentifierDefault 

BUT: that poses a transitional problem as in the current WITH_G4CXOPTICKS
the sensor placement stuff is not being done.

SO: I need to provide something similar from the stree ? In order to 
get the sensor_id into the 4th column of instances. 

AHH: bit not as simple as providing API as need to add to 4th column 
of the inst.


::

    192 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    193 {
    194     LOG(LEVEL) << " G4VPhysicalVolume world " << world ;
    195     assert(world);
    196     wd = world ;
    197     tr = U4Tree::Create(world, SensorIdentifier ) ;
    198 
    199 #ifdef __APPLE__
    200     return ;
    201 #endif
    202 
    203     // GGeo creation done when starting from a gdml or live G4,  still needs Opticks instance
    204     Opticks::Configure("--gparts_transform_offset --allownokey" );
    205 
    206     GGeo* gg_ = X4Geo::Translate(wd) ;
    207     setGeometry(gg_);
    208 }




After stree::reorderSensors
------------------------------

* after reordering the sensor_index they match (modulo +1)


::

    cd ~/opticks/sysrap/tests
    ./stree_test.sh build_run

    [ stree::reorderSensors
    ] stree::reorderSensors sensor_count 45612
    stree::add_inst i   0 gas_idx   1 nodes.size   25600
    stree::add_inst i   1 gas_idx   2 nodes.size   12615
    stree::add_inst i   2 gas_idx   3 nodes.size    4997
    stree::add_inst i   3 gas_idx   4 nodes.size    2400
    stree::add_inst i   4 gas_idx   5 nodes.size     590
    stree::add_inst i   5 gas_idx   6 nodes.size     590
    stree::add_inst i   6 gas_idx   7 nodes.size     590
    stree::add_inst i   7 gas_idx   8 nodes.size     590
    stree::add_inst i   8 gas_idx   9 nodes.size     504
    stree::save_ /tmp/blyth/opticks/ntds3/G4CXOpticks/stree_reorderSensors
    epsilon:tests blyth$ 

::

    In [4]: t.inst.view(np.int64)[:,:,3]
    Out[4]: 
    array([[     1,      1,     -1,     -1],
           [     2,      2, 300000,  17612],
           [     3,      2, 300001,  17613],
           [     4,      2, 300002,  17614],
           [     5,      2, 300003,  17615],
           ...,
           [ 48473,     10,     -1,     -1],
           [ 48474,     10,     -1,     -1],
           [ 48475,     10,     -1,     -1],
           [ 48476,     10,     -1,     -1],
           [ 48477,     10,     -1,     -1]])


    In [11]: w = np.where( t.inst.view(np.int64)[:,3,3]  > -1 )[0]

    In [12]: w
    Out[12]: array([    1,     2,     3,     4,     5, ..., 45608, 45609, 45610, 45611, 45612])

    In [13]: w.shape
    Out[13]: (45612,)

    In [14]: t.inst.view(np.int64)[w,:,3]
    Out[14]: 
    array([[     2,      2, 300000,  17612],
           [     3,      2, 300001,  17613],
           [     4,      2, 300002,  17614],
           [     5,      2, 300003,  17615],
           [     6,      2, 300004,  17616],
           ...,
           [ 45609,      5,  32395,  45607],
           [ 45610,      5,  32396,  45608],
           [ 45611,      5,  32397,  45609],
           [ 45612,      5,  32398,  45610],
           [ 45613,      5,  32399,  45611]])

    In [15]: t.inst.view(np.int64)[w,:,3].shape
    Out[15]: (45612, 4)

    In [16]: t.inst.view(np.int64)[w,3,3]
    Out[16]: array([17612, 17613, 17614, 17615, 17616, ..., 45607, 45608, 45609, 45610, 45611])

    In [17]: sidx   ## created by concatenating the values extract from iid 
    Out[17]: array([17613, 17614, 17615, 17616, 17617, ..., 45608, 45609, 45610, 45611, 45612], dtype=uint32)

    In [18]: np.all( t.inst.view(np.int64)[w,3,3] + 1 == sidx  )
    Out[18]: True


    i = t.inst.view(np.int64) 


    In [51]: i[:,1,3]
    Out[51]: array([ 1,  2,  2,  2,  2, ..., 10, 10, 10, 10, 10])

    In [52]: np.unique( i[:,1,3], return_counts=True )
    Out[52]: 
    (array([    1,     2,     3,     4,     5,     6,     7,     8,     9,    10]),
     array([    1, 25600, 12615,  4997,  2400,   590,   590,   590,   590,   504]))


    w2 = np.where( i[:,1,3] == 2 )[0]  
    w3 = np.where( i[:,1,3] == 3 )[0]  
    w4 = np.where( i[:,1,3] == 4 )[0]  
    w5 = np.where( i[:,1,3] == 5 )[0]  
    w6 = np.where( i[:,1,3] == 6 )[0]  


    In [2]: i[w2,:,3]
    Out[2]: 
    array([[     2,      2, 300000,  17612],
           [     3,      2, 300001,  17613],
           [     4,      2, 300002,  17614],
           [     5,      2, 300003,  17615],
           [     6,      2, 300004,  17616],
           ...,
           [ 25597,      2, 325595,  43207],
           [ 25598,      2, 325596,  43208],
           [ 25599,      2, 325597,  43209],
           [ 25600,      2, 325598,  43210],
           [ 25601,      2, 325599,  43211]])

    In [3]: i[w3,:,3]
    Out[3]: 
    array([[25602,     3,     2,     2],
           [25603,     3,     4,     4],
           [25604,     3,     6,     6],
           [25605,     3,    21,    21],
           [25606,     3,    22,    22],
           ...,
           [38212,     3, 17586, 17586],
           [38213,     3, 17587, 17587],
           [38214,     3, 17588, 17588],
           [38215,     3, 17589, 17589],
           [38216,     3, 17590, 17590]])

    In [4]: i[w4,:,3]
    Out[4]: 
    array([[38217,     4,     0,     0],
           [38218,     4,     1,     1],
           [38219,     4,     3,     3],
           [38220,     4,     5,     5],
           [38221,     4,     7,     7],
           ...,
           [43209,     4, 17607, 17607],
           [43210,     4, 17608, 17608],
           [43211,     4, 17609, 17609],
           [43212,     4, 17610, 17610],
           [43213,     4, 17611, 17611]])

    In [5]: i[w5,:,3]
    Out[5]: 
    array([[43214,     5, 30000, 43212],
           [43215,     5, 30001, 43213],
           [43216,     5, 30002, 43214],
           [43217,     5, 30003, 43215],
           [43218,     5, 30004, 43216],
           ...,
           [45609,     5, 32395, 45607],
           [45610,     5, 32396, 45608],
           [45611,     5, 32397, 45609],
           [45612,     5, 32398, 45610],
           [45613,     5, 32399, 45611]])

    In [6]: i[w6,:,3]
    Out[6]: 
    array([[45614,     6,    -1,    -1],
           [45615,     6,    -1,    -1],
           [45616,     6,    -1,    -1],
           [45617,     6,    -1,    -1],
           [45618,     6,    -1,    -1],
           ...,
           [46199,     6,    -1,    -1],
           [46200,     6,    -1,    -1],
           [46201,     6,    -1,    -1],
           [46202,     6,    -1,    -1],
           [46203,     6,    -1,    -1]])



Compare GGeo/iid with the stree/inst
---------------------------------------

* GGeo/iid orders sensors in preorder of the placements
* added stree::reorderSensors to duplicate this 


::

    In [29]: inst.shape
    Out[29]: (48477, 4, 4)

    In [34]: np.where( inst.view(np.int64)[:,3,3] == -1 )[0].shape   ## non-sensor instances
    Out[34]: (2865,)

    In [35]: 48477 - 2865
    Out[35]: 45612


    In [22]: inst.view(np.int64)[:100,:,3]
    Out[22]: 
    array([[                  1,                   1,                  -1,                  -1],
           [                  2,                   2,              300000, 4607182418800017408],   ## issue with 0 : was strid.h kludge skipped
           [                  3,                   2,              300001,                   1],
           [                  4,                   2,              300002,                   2],
           [                  5,                   2,              300003,                   3],


    In [3]: t.inst_f4.view(np.int32)[:,:,3]
    Out[3]: 
    array([[         1,          1,         -1,         -1],
           [         2,          2,     300000, 1065353216],
           [         3,          2,     300001,          1],
           [         4,          2,     300002,          2],
           [         5,          2,     300003,          3],
           ...,
           [     48473,         10,         -1,         -1],
           [     48474,         10,         -1,         -1],
           [     48475,         10,         -1,         -1],
           [     48476,         10,         -1,         -1],
           [     48477,         10,         -1,         -1]], dtype=int32)



    In [18]: t.inst.view(np.int64)[25590:25610,2,3]
    Out[18]: array([325589, 325590, 325591, 325592, 325593, 325594, 325595, 325596, 325597, 325598, 325599,      2,      4,      6,     21,     22,     23,     24,     25,     26])

    In [28]: inst.view(np.int64)[25590:25700,:,3]
    Out[28]: 
    array([[ 25591,      2, 325589,  25589],
           [ 25592,      2, 325590,  25590],
           [ 25593,      2, 325591,  25591],
           [ 25594,      2, 325592,  25592],
           [ 25595,      2, 325593,  25593],
           [ 25596,      2, 325594,  25594],
           [ 25597,      2, 325595,  25595],
           [ 25598,      2, 325596,  25596],
           [ 25599,      2, 325597,  25597],
           [ 25600,      2, 325598,  25598],
           [ 25601,      2, 325599,  25599],
           [ 25602,      3,      2,  25600],
           [ 25603,      3,      4,  25601],
           [ 25604,      3,      6,  25602],
           [ 25605,      3,     21,  25603],



G4Opticks::getHit HMM it was a mistake to treat identifier like efficiencies, as somehow more fundamental::

    1357     // via m_sensorlib 
    1358     hit->sensor_identifier = getSensorIdentifier(pflag.sensorIndex);
    1359 

    0868 int G4Opticks::getSensorIdentifier(unsigned sensorIndex) const
     869 {
     870     assert( m_sensorlib );
     871     return m_sensorlib->getSensorIdentifier(sensorIndex);
     872 }

     856 void G4Opticks::setSensorData(unsigned sensorIndex, float efficiency_1, float efficiency_2, int category, int identifier)
     857 {
     858     assert( m_sensorlib );
     859     m_sensorlib->setSensorData(sensorIndex, efficiency_1, efficiency_2, category, identifier);
     860 }
     861 
     862 void G4Opticks::getSensorData(unsigned sensorIndex, float& efficiency_1, float& efficiency_2, int& category, int& identifier) const
     863 {
     864     assert( m_sensorlib );
     865     m_sensorlib->getSensorData(sensorIndex, efficiency_1, efficiency_2, category, identifier);
     866 }
     867 
     868 int G4Opticks::getSensorIdentifier(unsigned sensorIndex) const
     869 {
     870     assert( m_sensorlib );
     871     return m_sensorlib->getSensorIdentifier(sensorIndex);
     872 }

    epsilon:opticks blyth$ find . -name SensorLib.hh
    ./optickscore/SensorLib.hh
    epsilon:opticks blyth$ 

    197 int SensorLib::getSensorIdentifier(unsigned sensorIndex) const
    198 {
    199     unsigned i = sensorIndex - 1 ;   // 1-based
    200     assert( i < m_sensor_num );
    201     assert( m_sensor_data );
    202     return m_sensor_data->getInt( i, 3, 0, 0);
    203 }

Ordering was based on sensor_placements, jcv LSExpDetectorConstruction_Opticks::

    123     const std::vector<G4PVPlacement*>& sensor_placements = g4ok->getSensorPlacements() ;
    124     unsigned num_sensor = sensor_placements.size();
    125 
    126     // 2. use the placements to pass sensor data : efficiencies, categories, identifiers  
    127 
    128     const junoSD_PMT_v2* sd = dynamic_cast<const junoSD_PMT_v2*>(sd_) ;
    129     assert(sd) ;
    130 
    131     LOG(info) << "[ setSensorData num_sensor " << num_sensor ;
    132     for(unsigned i=0 ; i < num_sensor ; i++)
    133     {   
    134         const G4PVPlacement* pv = sensor_placements[i] ; // i is 0-based unlike sensor_index
    135         unsigned sensor_index = 1 + i ; // 1-based 
    136         assert(pv);  
    137         G4int copyNo = pv->GetCopyNo();
    138         int pmtid = copyNo ; 
    139         int pmtcat = 0 ; // sd->getPMTCategory(pmtid); 
    140         float efficiency_1 = sd->getQuantumEfficiency(pmtid);
    141         float efficiency_2 = sd->getEfficiencyScale() ;
    142         
    143         g4ok->setSensorData( sensor_index, efficiency_1, efficiency_2, pmtcat, pmtid );
    144     }
    145     LOG(info) << "] setSensorData num_sensor " << num_sensor ;
    146 

::

     763 /**
     764 G4Opticks::getSensorPlacements (pre-cache live running only)
     765 ---------------------------------------------------------------
     766 
     767 Sensor placements are the outer volumes of instance assemblies that 
     768 contain sensor volumes.  The order of the returned vector of G4PVPlacement
     769 is that of the Opticks sensorIndex. 
     770 This vector allows the connection between the Opticks sensorIndex 
     771 and detector specific handling of sensor quantities to be established.
     772 
     773 NB this assumes only one volume with a sensitive surface within each 
     774 repeated geometry instance
     775 
     776 For example JUNO uses G4PVPlacement::GetCopyNo() as a non-contiguous PMT 
     777 identifier, which allows lookup of efficiencies and PMT categories.
     778 
     779 Sensor data is assigned via calls to setSensorData with 
     780 the 0-based contiguous Opticks sensorIndex as the first argument.   
     781 
     782 **/
     783 
     784 const std::vector<G4PVPlacement*>& G4Opticks::getSensorPlacements() const
     785 {
     786     return m_sensor_placements ;
     787 }

     648 void G4Opticks::setGeometry(const GGeo* ggeo)
     649 {
     650     bool loaded = ggeo->isLoadedFromCache() ;
     651     unsigned num_sensor = ggeo->getNumSensorVolumes();
     652 
     653 
     654     if( loaded == false )
     655     {
     656         if(m_placement_outer_volume) LOG(error) << "CAUTION : m_placement_outer_volume TRUE " ;
     657         X4PhysicalVolume::GetSensorPlacements(ggeo, m_sensor_placements, m_placement_outer_volume);
     658         assert( num_sensor == m_sensor_placements.size() ) ;
     659     }
     660 

::

    1995 /**
    1996 X4PhysicalVolume::GetSensorPlacements
    1997 ---------------------------------------
    1998 
    1999 Populates placements with the void* origins obtained from ggeo, casting them back to G4PVPlacement.
    2000 
    2001 
    2002 Invoked from G4Opticks::translateGeometry, kinda feels misplaced being in X4PhysicalVolume
    2003 as depends only on GGeo+G4, perhaps should live in G4Opticks ?
    2004 Possibly the positioning is side effect from the difficulties of testing G4Opticks 
    2005 due to it not being able to boot from cache.
    2006 
    2007 **/
    2008 
    2009 void X4PhysicalVolume::GetSensorPlacements(const GGeo* gg, std::vector<G4PVPlacement*>& placements, bool outer_volume ) // static
    2010 {
    2011     placements.clear();
    2012 
    2013     std::vector<void*> placements_ ;
    2014     gg->getSensorPlacements(placements_, outer_volume);
    2015 
    2016     for(unsigned i=0 ; i < placements_.size() ; i++)
    2017     {
    2018          G4PVPlacement* p = static_cast<G4PVPlacement*>(placements_[i]);
    2019          placements.push_back(p);
    2020     }
    2021 }

    1235 void GGeo::getSensorPlacements(std::vector<void*>& placements, bool outer_volume) const
    1236 {
    1237     m_nodelib->getSensorPlacements(placements, outer_volume);
    1238 }

    0681 void GNodeLib::getSensorPlacements(std::vector<void*>& placements, bool outer_volume) const
     682 {
     683     unsigned numSensorVolumes = getNumSensorVolumes();
     684     LOG(LEVEL) << "numSensorVolumes " << numSensorVolumes ;
     685     for(unsigned i=0 ; i < numSensorVolumes ; i++)
     686     {
     687         unsigned sensorIndex = 1 + i ; // 1-based
     688         const GVolume* sensor = getSensorVolume(sensorIndex) ;
     689         assert(sensor);
     690 
     691         void* origin = NULL ;
     692 
     693         if(outer_volume)
     694         {
     695             const GVolume* outer = sensor->getOuterVolume() ;
     696             assert(outer);
     697             origin = outer->getOriginNode() ;
     698             assert(origin);
     699         }
     700         else
     701         {
     702             origin = sensor->getOriginNode() ;
     703             assert(origin);
     704         }
     705 
     706         placements.push_back(origin);
     707     }
     708 }

     570 /**
     571 GNodeLib::getSensorVolume (precache only)
     572 -------------------------------------------
     573 
     574 **/
     575 
     576 const GVolume* GNodeLib::getSensorVolume(unsigned sensorIndex) const
     577 {
     578     return m_loaded ? NULL : m_sensor_volumes[sensorIndex-1];  // 1-based sensorIndex
     579 }


     449 void GNodeLib::addVolume(const GVolume* volume)
     450 {
     ...
     486     bool is_sensor = volume->hasSensorIndex(); // volume with 1-based sensorIndex assigned
     487     if(is_sensor)
     488     {
     489         m_sensor_volumes.push_back(volume);
     490         m_sensor_identity.push_back(id);
     491         m_num_sensors += 1 ;
     492     }

Volumes added to nodelib in preorder, so sensor ordering is preorder:: 

     840 void GInstancer::collectNodes()
     841 {
     842     assert(m_root);
     843     collectNodes_r(m_root, 0);
     844 }
     845 void GInstancer::collectNodes_r(const GNode* node, unsigned depth )
     846 {
     847     const GVolume* volume = dynamic_cast<const GVolume*>(node);
     848     m_nodelib->addVolume(volume);
     849     for(unsigned i = 0; i < node->getNumChildren(); i++) collectNodes_r(node->getChild(i), depth + 1 );
     850 }




::

    329 bool GVolume::hasSensorIndex() const
    330 {
    331     return m_sensorIndex != SENSOR_UNSET ;
    332 }

    308 /**
    309 GVolume::setSensorIndex
    310 -------------------------
    311 
    312 sensorIndex is expected to be a 1-based contiguous index, with the 
    313 default value of SENSOR_UNSET (0)  meaning no sensor.
    314 
    315 This is canonically invoked from X4PhysicalVolume::convertNode during GVolume creation.
    316 
    317 * GNode::setSensorIndices duplicates the index to all faces of m_mesh triangulated geometry
    318 
    319 **/
    320 void GVolume::setSensorIndex(unsigned sensorIndex)
    321 {
    322     m_sensorIndex = sensorIndex ;
    323     setSensorIndices( m_sensorIndex );
    324 }


    1679 GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
    1680 {
    ....
    1857     ///////// sensor decision for the volume happens here  ////////////////////////
    1858     //////// TODO: encapsulate into a GBndLib::formSensorIndex ? 
    1859 
    1860     bool is_sensor = m_blib->isSensorBoundary(boundary) ; // this means that isurf/osurf has non-zero EFFICIENCY property 
    1861     unsigned sensorIndex = GVolume::SENSOR_UNSET ;
    1862     if(is_sensor)
    1863     {
    1864         sensorIndex = 1 + m_blib->getSensorCount() ;  // 1-based index
    1865         m_blib->countSensorBoundary(boundary);
    1866     }
    1867     volume->setSensorIndex(sensorIndex);   // must set to GVolume::SENSOR_UNSET for non-sensors, for sensor_indices array  
    1868 



Arghh need parallel development on the intermediate workflow
----------------------------------------------------------------

The U4Tree/stree/inst creation and persisting of sensor info seems to be working OK, insofar as can test. 
BUT: cannot proceed and fully test this as are still using the GGeo CSG_GGeo converted CSGFoundry geometry. 

So need to add analogous sensor info via the GGeo CSG_GGeo route into CSGFoundry. 
in order to mimic what are doing in U4Tree/stree : in the same locations in inst fourth column. 

This is an interim solution until make the leap to the new geometry workflow. 

* straightforward to add sensor handling to CSGFoundry::addInstance and qat4 
* BUT: where to get sensor_id and sensor_idx in this workflow ?

  * GGeo/GVolume/GNode is the old heavyweight equivalent of stree 


HMM: probably sensor info needs to come via InstancedIdentityBuffer ?::

     200 void CSG_GGeo_Convert::addInstances(unsigned repeatIdx )
     201 {   
     202     unsigned nmm = ggeo->getNumMergedMesh();
     203     assert( repeatIdx < nmm ); 
     204     const GMergedMesh* mm = ggeo->getMergedMesh(repeatIdx);
     205     unsigned num_inst = mm->getNumITransforms() ;
     206     NPY<unsigned>* iid = mm->getInstancedIdentityBuffer();
     207     
     208     LOG(LEVEL) 
     209         << " repeatIdx " << repeatIdx
     210         << " num_inst (GMergedMesh::getNumITransforms) " << num_inst
     211         << " iid " << ( iid ? iid->getShapeString() : "-"  )
     212         ;
     213     
     214     //LOG(LEVEL) << " nmm " << nmm << " repeatIdx " << repeatIdx << " num_inst " << num_inst ; 
     215     
     216     for(unsigned i=0 ; i < num_inst ; i++)
     217     {   
     218         glm::mat4 it = mm->getITransform_(i);
     219         
     220         const float* tr16 = glm::value_ptr(it) ;
     221         unsigned gas_idx = repeatIdx ;
     222         unsigned ias_idx = 0 ;
     223         
     224         foundry->addInstance(tr16, gas_idx, ias_idx);
     225     }
     226 }



* HMM: threading it the sensor_id all the way thru GGeo seems like a lot of effort 
  for just a simple mapping from sensor_index to sensor_id (especially as this 
  code does not have long to live)

* so instead can just have the sensor_id/sensor_index mapping array 
  as an input to the CG conversion 

Prep for bringing sensor_index and sensor_id to instance fourth column 
with GMergedMesh::getInstancedIdentityBuffer_SensorIndex for use 
from the CSG_GGeo_Convert::addInstances::

     203 void CSG_GGeo_Convert::addInstances(unsigned repeatIdx )
     204 {
     205     unsigned nmm = ggeo->getNumMergedMesh();
     206     assert( repeatIdx < nmm );
     207     const GMergedMesh* mm = ggeo->getMergedMesh(repeatIdx);
     208     unsigned num_inst = mm->getNumITransforms() ;
     209     NPY<unsigned>* iid = mm->getInstancedIdentityBuffer();
     210 
     211     std::vector<int> sensor_index ;
     212     mm->getInstancedIdentityBuffer_SensorIndex(sensor_index);
     213     
     214     unsigned ni = iid->getShape(0); 
     215     unsigned nj = iid->getShape(1);
     216     unsigned nk = iid->getShape(2);
     217     assert( ni == sensor_index.size() );
     218     assert( nk == 4 );
     219     
     220     LOG(LEVEL)
     221         << " repeatIdx " << repeatIdx
     222         << " num_inst (GMergedMesh::getNumITransforms) " << num_inst
     223         << " iid " << ( iid ? iid->getShapeString() : "-"  )
     224         << " ni " << ni 
     225         << " nj " << nj     
     226         << " nk " << nk    
     227         ;
     228         

     



::

     609 /**
     610 GMesh::getInstancedIdentity
     611 -----------------------------
     612 
     613 All nodes of the geometry tree have a quad of identity uint.
     614 InstancedIdentity exists to rearrange that identity information 
     615 into a buffer that can be used for creation of the GPU instanced geometry,
     616 which requires to access the identity with an instance index, rather 
     617 than the node index.
     618 
     619 See notes/issues/identity_review.rst
     620 
     621 **/
     622 
     623 guint4 GMesh::getInstancedIdentity(unsigned int index) const
     624 {
     625     return m_iidentity[index] ;
     626 }


::

    226 /**
    227 GVolume::getIdentity
    228 ----------------------
    229 
    230 The volume identity quad is available GPU side for all intersects
    231 with geometry.
    232 
    233 1. node_index (3 bytes at least as JUNO needs more than 2-bytes : so little to gain from packing) 
    234 2. triplet_identity (4 bytes, pre-packed)
    235 3. SPack::Encode22(mesh_index, boundary_index)
    236 
    237    * mesh_index: 2 bytes easily enough, 0xffff = 65535
    238    * boundary_index: 2 bytes easily enough  
    239 
    240 4. sensorIndex (2 bytes easily enough) 
    241 
    242 The sensor_identifier is detector specific so would have to allow 4-bytes 
    243 hence exclude it from this identity, instead can use sensorIndex to 
    244 look up sensor_identifier within G4Opticks::getHit 
    245 
    246 Formerly::
    247 
    248    guint4 id(getIndex(), getMeshIndex(),  getBoundary(), getSensorIndex()) ;
    249 
    250 **/
    251 
    252 glm::uvec4 GVolume::getIdentity() const
    253 {
    254     glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ;
    255     return id ;
    256 }
    257 


* HMM this identity goes into GMergedMesh::m_identity

::

    1245 /**
    1246 GMergedMesh::addInstancedBuffers
    1247 -----------------------------------
    1248 
    1249 Canonically invoked only by GInstancer::makeMergedMeshAndInstancedBuffers
    1250 
    1251 
    1252 itransforms InstanceTransformsBuffer
    1253     (num_instances, 4, 4)
    1254 
    1255     collect GNode placement transforms into buffer
    1256 
    1257 iidentity InstanceIdentityBuffer
    1258     From Aug 2020: (num_instances, num_volumes_per_instance, 4 )
    1259     Before:        (num_instances*num_volumes_per_instance, 4 )
    1260 
    1261     collects the results of GVolume::getIdentity for all volumes within all instances. 
    1262 
    1263 **/
    1264 
    1265 void GMergedMesh::addInstancedBuffers(const std::vector<const GNode*>& placements)
    1266 {
    1267     LOG(LEVEL) << " placements.size() " << placements.size() ;
    1268 
    1269     NPY<float>* itransforms = GTree::makeInstanceTransformsBuffer(placements);
    1270     setITransformsBuffer(itransforms);
    1271 
    1272     NPY<unsigned int>* iidentity  = GTree::makeInstanceIdentityBuffer(placements);
    1273     setInstancedIdentityBuffer(iidentity);
    1274 }
    1275 

The iid contains numPlacements*numVolumes(in the instance subtree) with getVolume being called for all vol.
So thats a little awkward unless the sensor info was repeated across the instance progeny::

    126 NPY<unsigned int>* GTree::makeInstanceIdentityBuffer(const std::vector<const GNode*>& placements)  // static
    127 {
    ...
    164     NPY<unsigned>* buf = NPY<unsigned>::make(0, 4);
    165     NPY<unsigned>* buf2 = NPY<unsigned>::make(numPlacements, numVolumes, 4);
    166     buf2->zero();
    ...
    206         unsigned s_count = 0 ;
    207         for(unsigned s=0 ; s < numVolumesAll ; s++ )
    208         {
    209             const GNode* node = s == 0 ? base : progeny[s-1] ;
    210             const GVolume* volume = dynamic_cast<const GVolume*>(node) ;
    211             bool skip = node->isCSGSkip() ;
    212             if(!skip)
    213             {
    214                 glm::uvec4 id = volume->getIdentity();
    215                 buf->add(id.x, id.y, id.z, id.w );
    216                 buf2->setQuad( id, i, s_count, 0) ;
    217                 s_count += 1 ;
    218             }
    219         }      // over volumes 
    220     }          // over placements 



Looking at the arrays the sensor_index is not repeated across the subtree::

    epsilon:tests blyth$ cd /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/1/
    epsilon:1 blyth$ i

    In [1]: iid = np.load("placement_iidentity.npy")

    In [3]: iid.shape
    Out[3]: (25600, 5, 4)


    In [2]: iid
    Out[2]: 
    array([[[  194249, 16777216,  7995420,        0],
            [  194250, 16777217,  7864351,        0],
            [  194251, 16777218,  7733286,    17613],
            [  194252, 16777219,  7798823,        0],
            [  194253, 16777220,  7929882,        0]],

           [[  194254, 16777472,  7995420,        0],
            [  194255, 16777473,  7864351,        0],
            [  194256, 16777474,  7733286,    17614],
            [  194257, 16777475,  7798823,        0],
            [  194258, 16777476,  7929882,        0]],

    In [4]: iid[:,2,3]
    Out[4]: array([17613, 17614, 17615, ..., 43210, 43211, 43212], dtype=uint32)

    In [5]: iid[:,2,3].min()
    Out[5]: 17613

    In [6]: iid[:,2,3].max()
    Out[6]: 43212


::

    epsilon:tests blyth$ ./iid.sh 
    symbol a a         (1, 3089, 4) path /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/0/placement_iidentity.npy 
    symbol b a        (25600, 5, 4) path /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/1/placement_iidentity.npy 
    symbol c a        (12615, 7, 4) path /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/2/placement_iidentity.npy 
    symbol d a         (4997, 7, 4) path /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/3/placement_iidentity.npy 
    symbol e a         (2400, 6, 4) path /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/4/placement_iidentity.npy 
    symbol f a          (590, 1, 4) path /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/5/placement_iidentity.npy 
    symbol g a          (590, 1, 4) path /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/6/placement_iidentity.npy 
    symbol h a          (590, 1, 4) path /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/7/placement_iidentity.npy 
    symbol i a          (590, 1, 4) path /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/8/placement_iidentity.npy 
    symbol j a        (504, 130, 4) path /tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/GMergedMesh/9/placement_iidentity.npy 


    In [1]: b[0]
    Out[1]: 
    array([[  194249, 16777216,  7995420,        0],
           [  194250, 16777217,  7864351,        0],
           [  194251, 16777218,  7733286,    17613],
           [  194252, 16777219,  7798823,        0],
           [  194253, 16777220,  7929882,        0]], dtype=uint32)

    In [2]: (b[:,2,3].min(),b[:,2,3].max())
    Out[2]: (17613, 43212)

    In [3]: c[0]
    Out[3]: 
    array([[   70979, 33554432,  7667740,        0],
           [   70980, 33554433,  7274525,        0],
           [   70981, 33554434,  7340067,        0],
           [   70982, 33554435,  7602207,        0],
           [   70983, 33554436,  7536672,        0],
           [   70984, 33554437,  7405604,        3],
           [   70985, 33554438,  7471141,        0]], dtype=uint32)

    In [4]: (c[:,5,3].min(),c[:,5,3].max())
    Out[4]: (3, 17591)

    In [5]: d[0]
    Out[5]: 
    array([[   70965, 50331648,  7208988,        0],
           [   70966, 50331649,  6815773,        0],
           [   70967, 50331650,  6881310,        0],
           [   70968, 50331651,  7143455,        0],
           [   70969, 50331652,  7077920,        0],
           [   70970, 50331653,  6946849,        1],
           [   70971, 50331654,  7012386,        0]], dtype=uint32)

    In [6]: (d[:,5,3].min(), d[:,5,3].max())
    Out[6]: (1, 17612)

    In [7]: e[0]
    Out[7]: 
    array([[  322253, 67108864,  8781866,        0],
           [  322254, 67108865,  8454163,        0],
           [  322255, 67108866,  8716319,        0],
           [  322256, 67108867,  8650784,        0],
           [  322257, 67108868,  8519723,    43213],
           [  322258, 67108869,  8585260,        0]], dtype=uint32)

    In [8]: (e[:,4,3].min(), e[:,4,3].max()) 
    Out[8]: (43213, 45612)


Look to be 1-based and use different orderng convention to stree. 





::

    1536 /**
    1537 CSGFoundry::addInstance
    1538 ------------------------
    1539 
    1540 Used for example from 
    1541 
    1542 1. CSG_GGeo_Convert::addInstances when creating CSGFoundry from GGeo
    1543 2. CSGCopy::copy/CSGCopy::copySolidInstances when copy a loaded CSGFoundry to apply a selection
    1544 
    1545 **/
    1546 
    1547 void CSGFoundry::addInstance(const float* tr16, unsigned gas_idx, unsigned ias_idx )
    1548 {
    1549     qat4 instance(tr16) ;  // identity matrix if tr16 is nullptr 
    1550     unsigned ins_idx = inst.size() ;
    1551 
    1552     instance.setIdentity( ins_idx, gas_idx, ias_idx );
    1553 
    1554     LOG(debug)
    1555         << " ins_idx " << ins_idx
    1556         << " gas_idx " << gas_idx
    1557         << " ias_idx " << ias_idx
    1558         ;
    1559 
    1560     inst.push_back( instance );
    1561 }





Not so keen on passing efficiencies one-by-one this way
--------------------------------------------------------

* identifiers and indices seems ok, as only one of those but 
  the other info will tend to need to be expanded

* better to establish the placement order and accept all values for
  all sensors in single API 


::

     30 struct ExampleSensor : public U4Sensor
     31 {
     32     // In reality would need ctor argument eg junoSD_PMT_v2 to lookup real values 
     33     unsigned getId(           const G4PVPlacement* pv) const { return pv->GetCopyNo() ; }
     34     float getEfficiency(      const G4PVPlacement* pv) const { return 1. ; }
     35     float getEfficiencyScale( const G4PVPlacement* pv) const { return 1. ; }
     36 }; 


Opted for::

     22 struct U4SensorIdentifier
     23 {
     24     virtual int getIdentity(const G4VPhysicalVolume* instance_outer_pv ) const = 0 ;
     25 };

     09 struct U4SensorIdentifierDefault
     10 {
     11     int getIdentity(const G4VPhysicalVolume* instance_outer_pv ) const ;
     12     static void FindSD_r( std::vector<const G4VPhysicalVolume*>& sdpv , const G4VPhysicalVolume* pv, int depth );
     13 };
     14 
     15 
     16 inline int U4SensorIdentifierDefault::getIdentity( const G4VPhysicalVolume* instance_outer_pv ) const
     17 {
     18     const G4PVPlacement* pvp = dynamic_cast<const G4PVPlacement*>(instance_outer_pv) ;
     19     int copyno = pvp ? pvp->GetCopyNo() : -1 ;
     20 
     21     std::vector<const G4VPhysicalVolume*> sdpv ;
     22     FindSD_r(sdpv, instance_outer_pv, 0 );
     23 
     24     unsigned num_sd = sdpv.size() ;
     25     int sensor_id = num_sd == 0 ? -1 : copyno ;
     26 
     27     std::cout
     28         << "U4SensorIdentifierDefault::getIdentity"
     29         << " copyno " << copyno
     30         << " num_sd " << num_sd
     31         << " sensor_id " << sensor_id
     32         ;
     33 
     34     return sensor_id ;
     35 }
     36 
     37 inline void U4SensorIdentifierDefault::FindSD_r( std::vector<const G4VPhysicalVolume*>& sdpv , const G4VPhysicalVolume* pv, int depth )
     38 {
     39     const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
     40     G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ;
     41     if(sd) sdpv.push_back(pv);
     42     for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) FindSD_r( lv->GetDaughter(i), depth+1, );
     43 }




Compare with Framework ProcessHits
-------------------------------------

::

     316 G4bool junoSD_PMT_v2::ProcessHits(G4Step * step,G4TouchableHistory*)
     317 {
     ...
     391     // == get the copy number -> pmt id
     392     int pmtID = get_pmtid(track);
     ...
     444     if (m_pmthitmerger and m_pmthitmerger->getMergeFlag()) {
     445         // == if merged, just return true. That means just update the hit
     446         // NOTE: only the time and count will be update here, the others 
     447         //       will not filled.
     448         bool ok = m_pmthitmerger->doMerge(pmtID, hittime);
     449         if (ok) {
     450             m_merge_count += 1 ;
     451             return true;
     452         }





What is the Opticks equivalent of junoSD_PMT_v2::get_pmtid ?
-------------------------------------------------------------

Opticks shifts focus to geometry preparation stage, so it doesnt have to 
be repeated for every photon.  That means:

1. duplicating sensor_id and sensor_index labels to all ~5-6 nodes of the subtree of 
   each instance within stree (formerly GGeo/GNodeLib/GNode)

2. planting sensor_id and sensor_index within the CSGFoundry inst in 
   fourth column of the transform. 

But how to get sensor_id and sensor_index in first place ?

sensor_index 
   0-based index that orders the sensors as they are 
   encountered in the standard postorder traversal of the volumes

   * this means that given a way to get sensor_id of a volume 
     can derive the sensor index within Opticks   

sensor_id
   this comes from the copyNo but that is JUNO specific so 
   cannot assume that is the 


How to label the subtrees ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

U4Tree::convertNodes_r 
     too early as the instances not yet defined 
    
stree::add_inst
     is the right place to label the tree and populate the inst 4th column, 
     but need to operate without Geant4 types within stree : so need to 
     collect sensor_id integer into the stree/snode during U4Tree::convertNodes_r 
     using the U4Sensor object passed from the framework (or copyno) 



junoSD_PMT_v2::get_pmtid
---------------------------

::

    junoSD_PMT_v2::ProcessHits dumpcount 0
    U4Touchable::Desc depth 8
     i  0 cp      0 so HamamatsuR12860_PMT_20inch_body_solid_1_4 pv                         HamamatsuR12860_PMT_20inch_body_phys
     i  1 cp      0 so HamamatsuR12860_PMT_20inch_pmt_solid_1_4 pv                          HamamatsuR12860_PMT_20inch_log_phys
     i  2 cp   9744 so             HamamatsuR12860sMask_virtual pv                                       pLPMT_Hamamatsu_R12860
     i  3 cp      0 so                              sInnerWater pv                                                  pInnerWater
     i  4 cp      0 so                           sReflectorInCD pv                                             pCentralDetector
     i  5 cp      0 so                          sOuterWaterPool pv                                              pOuterWaterPool
     i  6 cp      0 so                              sPoolLining pv                                                  pPoolLining
     i  7 cp      0 so                              sBottomRock pv                                                     pBtmRock

    junoSD_PMT_v2::ProcessHits dumpcount 1
    U4Touchable::Desc depth 8
     i  0 cp      0 so    NNVTMCPPMT_PMT_20inch_body_solid_head pv                              NNVTMCPPMT_PMT_20inch_body_phys
     i  1 cp      0 so     NNVTMCPPMT_PMT_20inch_pmt_solid_head pv                               NNVTMCPPMT_PMT_20inch_log_phys
     i  2 cp   3505 so                  NNVTMCPPMTsMask_virtual pv                                            pLPMT_NNVT_MCPPMT
     i  3 cp      0 so                              sInnerWater pv                                                  pInnerWater
     i  4 cp      0 so                           sReflectorInCD pv                                             pCentralDetector
     i  5 cp      0 so                          sOuterWaterPool pv                                              pOuterWaterPool
     i  6 cp      0 so                              sPoolLining pv                                                  pPoolLining
     i  7 cp      0 so                              sBottomRock pv                                                     pBtmRock





::

     477 int junoSD_PMT_v2::get_pmtid(G4Track* track) {
     478     int ipmt= -1;
     479     // find which pmt we are in
     480     // The following doesn't work anymore (due to new geometry optimization?)
     481     //  ipmt=fastTrack.GetEnvelopePhysicalVolume()->GetMother()->GetCopyNo();
     482     // so we do this:
     483     {
     484         const G4VTouchable* touch= track->GetTouchable();
     485         int nd= touch->GetHistoryDepth();
     486         int id=0;
     487         for (id=0; id<nd; id++) {   
     488             if (touch->GetVolume(id)==track->GetVolume()) {
     ///
     ///         iterate up stack of volumes : until find the one of this track : 
     ///         would expect that to be the first 
     ///
     489                 int idid=1;
     490                 G4VPhysicalVolume* tmp_pv=NULL;
     491                 for (idid=1; idid < (nd-id); ++idid) {
     ///
     ///            code edited to make less obtuse. 
     ///            looks like proceeds up the stack until finds a volume with siblings
     ///            in order to get the CopyNo  
     ///
     ...
     494                     G4LogicalVolume* mother_vol = touch->GetVolume(id+idid)->GetLogicalVolume();
     495                     G4LogicalVolume* daughter_vol = touch->GetVolume(id+idid-1)->GetLogicalVolume();

     497                     int no_daugh = mother_vol -> GetNoDaughters();
     498                     if (no_daugh > 1) {
     499                         int count = 0;
     500                         for (int i=0; (count<2) &&(i < no_daugh); ++i) {
     501                             if (daughter_vol->GetName()==mother_vol->GetDaughter(i)->GetLogicalVolume()->GetName()) {
     503                                 ++count;
     504                             }
     505                         }
     506                         if (count > 1) {
     507                             break;
     508                         }
     509                     }
     510                     // continue to find
     511                 }
     512                 ipmt= touch->GetReplicaNumber(id+idid-1);
     513                 break;
     514             }
     515         }
     516         if (ipmt < 0) {
     517             G4Exception("junoPMTOpticalModel: could not find envelope -- where am I !?!", // issue
     518                     "", //Error Code
     519                     FatalException, // severity
     520                     "");
     521         }
     522     }
     523 
     524     return ipmt;
     525 }


g4-cls G4VTouchable::

     34 inline
     35 G4int G4VTouchable::GetCopyNumber(G4int depth) const
     36 { 
     37   return GetReplicaNumber(depth);
     38 }


     59 inline
     60 G4VPhysicalVolume* G4TouchableHistory::GetVolume( G4int depth ) const
     61 {   
     62   return fhistory.GetVolume(CalculateHistoryIndex(depth));
     63 }
     64    
     65 inline
     66 G4VSolid* G4TouchableHistory::GetSolid( G4int depth ) const
     67 {
     68   return fhistory.GetVolume(CalculateHistoryIndex(depth))
     69                             ->GetLogicalVolume()->GetSolid();
     70 }
     71   
     72 inline
     73 G4int G4TouchableHistory::GetReplicaNumber( G4int depth ) const
     74 {
     75   return fhistory.GetReplicaNo(CalculateHistoryIndex(depth));
     76 }
     77 

     53 inline
     54 G4int G4TouchableHistory::CalculateHistoryIndex( G4int stackDepth ) const
     55 { 
     56   return (fhistory.GetDepth()-stackDepth); // was -1
     57 }

::

    098   G4ThreeVector ftlate;
     99   G4NavigationHistory fhistory;
    100 };




U4Sensor
----------

::

    epsilon:u4 blyth$ opticks-f U4Sensor
    ./u4/CMakeLists.txt:    U4Sensor.h
    ./u4/U4Sensor.h:U4Sensor.h
    ./u4/U4Sensor.h:struct U4Sensor
    ./g4cx/G4CXOpticks.hh:struct U4Sensor ; 
    ./g4cx/G4CXOpticks.hh:    const U4Sensor* sd ; 
    ./g4cx/G4CXOpticks.hh:    void setSensor(const U4Sensor* sd );
    ./g4cx/G4CXOpticks.hh:    // HMM: maybe add U4Sensor arg here, 
    ./g4cx/tests/G4CXSimulateTest.cc:#include "U4Sensor.h"
    ./g4cx/tests/G4CXSimulateTest.cc:struct ExampleSensor : public U4Sensor
    ./g4cx/G4CXOpticks.cc:void G4CXOpticks::setSensor(const U4Sensor* sd_ )
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 

::

    120 void G4CXOpticks::setSensor(const U4Sensor* sd_ )
    121 {
    122     sd = sd_ ;
    123 }

    030 struct ExampleSensor : public U4Sensor
     31 {
     32     // In reality would need ctor argument eg junoSD_PMT_v2 to lookup real values 
     33     unsigned getId(           const G4PVPlacement* pv) const { return pv->GetCopyNo() ; }
     34     float getEfficiency(      const G4PVPlacement* pv) const { return 1. ; }
     35     float getEfficiencyScale( const G4PVPlacement* pv) const { return 1. ; }
     36 }; 




What is the effect of having non-sensitive SD volumes ?
----------------------------------------------------------

Probably no effect, as need "theStatus == Detection" anyhow
and to get "Detection" need an efficiency property with value 
greater than zero and a suitable random throw. 

BUT : it adds a complication for communicating efficiencies 

::

    411 inline
    412 void InstrumentedG4OpBoundaryProcess::DoAbsorption()
    413 {
    414               theStatus = Absorption;
    415 
    416               if ( G4BooleanRand_theEfficiency(theEfficiency) ) {
    417 
    418                  // EnergyDeposited =/= 0 means: photon has been detected
    419                  theStatus = Detection;
    420                  aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
    421               }
    422               else {
    423                  aParticleChange.ProposeLocalEnergyDeposit(0.0);
    424               }
    425 
    426               NewMomentum = OldMomentum;
    427               NewPolarization = OldPolarization;
    428 
    429 //              aParticleChange.ProposeEnergy(0.0);
    430               aParticleChange.ProposeTrackStatus(fStopAndKill);
    431 }


::

    1617 G4bool InstrumentedG4OpBoundaryProcess::InvokeSD(const G4Step* pStep)
    1618 {
    1619   G4Step aStep = *pStep;
    1620 
    1621   aStep.AddTotalEnergyDeposit(thePhotonMomentum);
    1622 
    1623   G4VSensitiveDetector* sd = aStep.GetPostStepPoint()->GetSensitiveDetector();
    1624   if (sd) return sd->Hit(&aStep);
    1625   else return false;
    1626 }


    0222 G4VParticleChange*
     223 InstrumentedG4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     224 {

     663         if ( theStatus == Detection && fInvokeSD ) InvokeSD(pStep);
     664 
     665         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     666 }



Check Sensors : systematically 2x the number of SD than would expect ?
------------------------------------------------------------------------

::

    epsilon:sysrap blyth$ jgr SetSensitive 
    ./Simulation/DetSimV2/PMTSim/src/Hello3inchPMTManager.cc:    body_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/Hello3inchPMTManager.cc:    inner1_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/dyw_PMT_LogicalVolume.cc:  body_log->SetSensitiveDetector(detector);
    ./Simulation/DetSimV2/PMTSim/src/dyw_PMT_LogicalVolume.cc:  inner1_log->SetSensitiveDetector(detector);
    ...


    457 void NNVTMCPPMTManager::helper_make_logical_volume()
    458 {
    459     body_log= new G4LogicalVolume
    460         ( body_solid,
    461           GlassMat,
    462           GetName()+"_body_log" );
    463 
    464     m_logical_pmt = new G4LogicalVolume
    465         ( pmt_solid,
    466           GlassMat,
    467           GetName()+"_log" );
    468 
    469     body_log->SetSensitiveDetector(m_detector);
    470 
    471     inner1_log= new G4LogicalVolume
    472         ( inner1_solid,
    473           PMT_Vacuum,
    474           GetName()+"_inner1_log" );
    475     inner1_log->SetSensitiveDetector(m_detector);
    476 

::

    desc_sensor
        nds :  lv :                                             soname : 0th 
       4997 : 106 :          HamamatsuR12860_PMT_20inch_inner1_solid_I : 70970 
       4997 : 108 :          HamamatsuR12860_PMT_20inch_body_solid_1_4 : 70969 
      12615 : 113 :            NNVTMCPPMT_PMT_20inch_inner1_solid_head : 70984 
      12615 : 115 :              NNVTMCPPMT_PMT_20inch_body_solid_head : 70983 
      25600 : 118 :                  PMT_3inch_inner1_solid_ell_helper : 194251 
      25600 : 120 :                PMT_3inch_body_solid_ell_ell_helper : 194250 
       2400 : 130 :                       PMT_20inch_veto_inner1_solid : 322257 
       2400 : 132 :                     PMT_20inch_veto_body_solid_1_2 : 322256 
      91224 :     :                                                    :  
    zth:70970
             +      snode ix:  70970 dh: 9 nc:    0 lv:106 se:      1. sf 125 :   -4997 : 8a3d4fe0109975976aef9a87c7842a63. HamamatsuR12860_PMT_20inch_inner1_solid_I
    zth:70969
            +       snode ix:  70969 dh: 8 nc:    2 lv:108 se:      0. sf 124 :   -4997 : f343253c582a107559795892ee52220f. HamamatsuR12860_PMT_20inch_body_solid_1_4
             +      snode ix:  70970 dh: 9 nc:    0 lv:106 se:      1. sf 125 :   -4997 : 8a3d4fe0109975976aef9a87c7842a63. HamamatsuR12860_PMT_20inch_inner1_solid_I
             +      snode ix:  70971 dh: 9 nc:    0 lv:107 se:     -1. sf 126 :   -4997 : fd63d016360b18a01ab74dcd01b5e32c. HamamatsuR12860_PMT_20inch_inner2_solid_1_4
    zth:70984
             +      snode ix:  70984 dh: 9 nc:    0 lv:113 se:      5. sf 131 :  -12615 : 341ae4bffe82aa82798d3886484179a6. NNVTMCPPMT_PMT_20inch_inner1_solid_head
    zth:70983
            +       snode ix:  70983 dh: 8 nc:    2 lv:115 se:      4. sf 130 :  -12615 : 067136473b80d872bffc4de42fbf2337. NNVTMCPPMT_PMT_20inch_body_solid_head
             +      snode ix:  70984 dh: 9 nc:    0 lv:113 se:      5. sf 131 :  -12615 : 341ae4bffe82aa82798d3886484179a6. NNVTMCPPMT_PMT_20inch_inner1_solid_head
             +      snode ix:  70985 dh: 9 nc:    0 lv:114 se:     -1. sf 132 :  -12615 : 946e0765de8ecaf64388ebe09c86680e. NNVTMCPPMT_PMT_20inch_inner2_solid_head
    zth:194251
            +       snode ix: 194251 dh: 8 nc:    0 lv:118 se:  35225. sf 133 :  -25600 : c301322ae66e730aac2a27836ead8b89. PMT_3inch_inner1_solid_ell_helper
    zth:194250
           +        snode ix: 194250 dh: 7 nc:    2 lv:120 se:  35224. sf 135 :  -25600 : 2485b31b2df8ec818453e3a773f02436. PMT_3inch_body_solid_ell_ell_helper
            +       snode ix: 194251 dh: 8 nc:    0 lv:118 se:  35225. sf 133 :  -25600 : c301322ae66e730aac2a27836ead8b89. PMT_3inch_inner1_solid_ell_helper
            +       snode ix: 194252 dh: 8 nc:    0 lv:119 se:     -1. sf 136 :  -25600 : 511486df0c29cd5e2e9a38b4a6d2e108. PMT_3inch_inner2_solid_ell_helper
    zth:322257
           +        snode ix: 322257 dh: 7 nc:    0 lv:130 se:  86425. sf 116 :   -2400 : 4c4aff2e5de757833006d7f55c3f2127. PMT_20inch_veto_inner1_solid
    zth:322256
          +         snode ix: 322256 dh: 6 nc:    2 lv:132 se:  86424. sf 118 :   -2400 : 38ba238fc5def688b7fe3639cc3f6c6f. PMT_20inch_veto_body_solid_1_2
           +        snode ix: 322257 dh: 7 nc:    0 lv:130 se:  86425. sf 116 :   -2400 : 4c4aff2e5de757833006d7f55c3f2127. PMT_20inch_veto_inner1_solid
           +        snode ix: 322258 dh: 7 nc:    0 lv:131 se:     -1. sf 117 :   -2400 : d2f14afe26c74ad9d618c6d18a2e25a1. PMT_20inch_veto_inner2_solid



::

     20 def desc_sensor(st):
     21     """
     22     desc_sensor
     23         nds :  lv : soname
     24        4997 : 106 : HamamatsuR12860_PMT_20inch_inner1_solid_I 
     25        4997 : 108 : HamamatsuR12860_PMT_20inch_body_solid_1_4 
     26       12615 : 113 : NNVTMCPPMT_PMT_20inch_inner1_solid_head 
     27       12615 : 115 : NNVTMCPPMT_PMT_20inch_body_solid_head 
     28       25600 : 118 : PMT_3inch_inner1_solid_ell_helper 
     29       25600 : 120 : PMT_3inch_body_solid_ell_ell_helper 
     30        2400 : 130 : PMT_20inch_veto_inner1_solid 
     31        2400 : 132 : PMT_20inch_veto_body_solid_1_2 
     32 
     33     """
     34     ws = np.where(st.nds.sensor > -1 )[0]
     35     se = st.nds.sensor[ws]
     36     xse = np.arange(len(se), dtype=np.int32)
     37     assert np.all( xse == se )  
     38     ulv, nlv = np.unique(st.nds.lvid[ws], return_counts=True)
     39     
     40     hfmt = "%7s : %3s : %s"
     41     fmt = "%7d : %3d : %s "
     42     hdr = hfmt % ("nds", "lv", "soname" )
     43     
     44     head = ["desc_sensor",hdr]
     45     body = [fmt % ( nlv[i], ulv[i], st.soname_[ulv[i]].decode() ) for i in range(len(ulv))]
     46     tail = [hfmt % ( nlv.sum(), "", "" ),]
     47     return "\n".join(head+body+tail)
     48     
     49     


::

    epsilon:offline blyth$ jgr _1_4
    ./Simulation/DetSimV2/PMTSim/src/Hamamatsu_R12860_PMTSolid.cc:				 solidname+"_1_4",
    ./Simulation/DetSimV2/PMTSim/src/Hamamatsu_R12860_PMTSolid.cc:    double neck_offset_z = -210. + m4_h/2 ;  // see _1_4 below
    ./Simulation/DetSimV2/PMTSim/src/Hamamatsu_R12860_PMTSolid.cc:    double c_cy = neck_offset_z -m4_h/2 ;    // -210. torus_z  (see _1_4 below)
    epsilon:offline blyth$ 




Should sensor_id be placed into OptixInstance .instanceId ?
------------------------------------------------------------------

::

    the returned unsigned value is used by IAS_Builder to set the OptixInstance .instanceId 
    Within CSGOptiX/CSGOptiX7.cu:: __closesthit__ch *optixGetInstanceId()* is used to 
    passes the instanceId value into "quad2* prd" (per-ray-data) which is available 
    within qudarap/qsim.h methods. 
    
    The 32 bit unsigned returned by *getInstanceIdentity* may not use the top 8 bits 
    because of an OptiX 7 limit of 24 bits, from Properties::dump::

        limitMaxInstanceId :   16777215    ffffff

    (that limit might well be raised in versions after 700)





HMM: how to split those 24 bits ? 

1. sensor id
2. sensor category (4 cat:2 bits, 8 cat: 3 bits)

::

    In [14]: for i in range(32): print(" (0x1 << %2d) - 1   %16x   %16d  %16.2f  " % (i, (0x1 << i)-1, (0x1 << i)-1, float((0x1 << i)-1)/1e6 )) 

     (0x1 <<  0) - 1                  0                  0              0.00  
     (0x1 <<  1) - 1                  1                  1              0.00  
     (0x1 <<  2) - 1                  3                  3              0.00  
     (0x1 <<  3) - 1                  7                  7              0.00  
     (0x1 <<  4) - 1                  f                 15              0.00  
     (0x1 <<  5) - 1                 1f                 31              0.00  
     (0x1 <<  6) - 1                 3f                 63              0.00  
     (0x1 <<  7) - 1                 7f                127              0.00  
     (0x1 <<  8) - 1                 ff                255              0.00  
     (0x1 <<  9) - 1                1ff                511              0.00  
     (0x1 << 10) - 1                3ff               1023              0.00  
     (0x1 << 11) - 1                7ff               2047              0.00  
     (0x1 << 12) - 1                fff               4095              0.00  
     (0x1 << 13) - 1               1fff               8191              0.01  
     (0x1 << 14) - 1               3fff              16383              0.02  
     (0x1 << 15) - 1               7fff              32767              0.03  
     (0x1 << 16) - 1               ffff              65535              0.07  
     (0x1 << 17) - 1              1ffff             131071              0.13  
     (0x1 << 18) - 1              3ffff             262143              0.26  
     (0x1 << 19) - 1              7ffff             524287              0.52  
     (0x1 << 20) - 1              fffff            1048575              1.05  
     (0x1 << 21) - 1             1fffff            2097151              2.10  
     (0x1 << 22) - 1             3fffff            4194303              4.19  
     (0x1 << 23) - 1             7fffff            8388607              8.39  
     (0x1 << 24) - 1             ffffff           16777215             16.78  
     (0x1 << 25) - 1            1ffffff           33554431             33.55  
     (0x1 << 26) - 1            3ffffff           67108863             67.11  
     (0x1 << 27) - 1            7ffffff          134217727            134.22  
     (0x1 << 28) - 1            fffffff          268435455            268.44  
     (0x1 << 29) - 1           1fffffff          536870911            536.87  
     (0x1 << 30) - 1           3fffffff         1073741823           1073.74  
     (0x1 << 31) - 1           7fffffff         2147483647           2147.48  







