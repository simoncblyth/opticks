review_sensor_id
====================

Context
----------

* from :doc:`QSimTest_shakedown_following_QPMT_extension`


WIP : trace where qat4 inst identity info comes from
-----------------------------------------------------------

::

    372     QAT4_METHOD void setIdentity(int ins_idx, int gas_idx, int sensor_identifier, int sensor_index )
    373     {
    374         q0.i.w = ins_idx ;             // formerly unsigned and "+ 1"
    375         q1.i.w = gas_idx ;
    376         q2.i.w = sensor_identifier ;
    377         q3.i.w = sensor_index ;
    378     }


    epsilon:sysrap blyth$ opticks-f setIdentity | grep -v tests
    ./CSGOptiX/SBT.cc:        q.setIdentity(ins_idx, gas_idx, sensor_identifier, sensor_index );
          non mainline SBT::createSolidSelectionIAS
    ./CSGOptiX/Six.cc:        q.setIdentity(ins_idx, gas_idx, sensor_identifier, sensor_index  );
          non mainline Six::createSolidSelectionIAS
    ./CSG/CSGMaker.cc:        instance.setIdentity( ins_idx, gas_idx, sensor_identifier, sensor_index );  
          non mainline CSGMaker::makeDemoGrid

    ./CSG/CSGFoundry.cc:    instance.setIdentity( ins_idx, gas_idx, sensor_identifier, sensor_index );

          THIS IS THE PRIMARY  


    ./sysrap/stree.h:TODO: modify sqat4.h::setIdentity to store this nidx and populate it 
    ./sysrap/stree.h:    int ins_idx = int(inst.size());     // follow sqat4.h::setIdentity
    ./sysrap/sqat4.h:    QAT4_METHOD void setIdentity(int ins_idx, int gas_idx, int sensor_identifier, int sensor_index )
    ./sysrap/stran.h:    v->setIdentity(ins_idx, gas_idx, sensor_identifier, sensor_index ) ;
    ./sysrap/sframe.py:        ins_idx_1 = i[1,0,3] - 1   # q0.u.w-1  see sqat4.h qat4::getIdentity qat4::setIdentity
    ./sysrap/sframe.py:        ins_idx_2 = i[2,0,3] - 1   # q0.u.w-1  see sqat4.h qat4::getIdentity qat4::setIdentity
    epsilon:opticks blyth$ 



::

    1691 /**
    1692 CSGFoundry::addInstance
    1693 ------------------------
    1694 
    1695 Used for example from 
    1696 
    1697 1. CSG_GGeo_Convert::addInstances when creating CSGFoundry from GGeo
    1698 2. CSGCopy::copy/CSGCopy::copySolidInstances when copy a loaded CSGFoundry to apply a selection
    1699 
    1700 **/
    1701 
    1702 void CSGFoundry::addInstance(const float* tr16, int gas_idx, int sensor_identifier, int sensor_index )
    1703 {
    1704     qat4 instance(tr16) ;  // identity matrix if tr16 is nullptr 
    1705     int ins_idx = int(inst.size()) ;
    1706 
    1707     instance.setIdentity( ins_idx, gas_idx, sensor_identifier, sensor_index );
    1708 



YUK, old/new mismash is handling the sensor_id::

     220 void CSG_GGeo_Convert::addInstances(unsigned repeatIdx )
     221 {
     222     unsigned nmm = ggeo->getNumMergedMesh();
     223     assert( repeatIdx < nmm );
     224     const GMergedMesh* mm = ggeo->getMergedMesh(repeatIdx);
     225     unsigned num_inst = mm->getNumITransforms() ;
     226     LOG(LEVEL) << " repeatIdx " << repeatIdx << " num_inst " << num_inst << " nmm " << nmm  ;
     227 
     228     NPY<unsigned>* iid = mm->getInstancedIdentityBuffer();
     229     LOG(LEVEL) << " iid " << ( iid ? iid->getShapeString() : "-"  ) ;
     230 
     231     assert(tree);
     232 
     233     bool one_based_index = true ;   // CAUTION : OLD WORLD 1-based sensor_index 
     234     std::vector<int> sensor_index ;
     235     mm->getInstancedIdentityBuffer_SensorIndex(sensor_index, one_based_index );
     236     LOG(LEVEL) << " sensor_index.size " << sensor_index.size() ;
     237 
     238 
     239     bool lookup_verbose = LEVEL == info ;
     240     std::vector<int> sensor_id ;
     241     tree->lookup_sensor_identifier(sensor_id, sensor_index, one_based_index, lookup_verbose );
     242 
     243     LOG(LEVEL) << " sensor_id.size " << sensor_id.size() ;
     244     LOG(LEVEL) << stree::DescSensor( sensor_id, sensor_index ) ;
     245 
     246     unsigned ni = iid->getShape(0);
     247     unsigned nj = iid->getShape(1);
     248     unsigned nk = iid->getShape(2);
     249     assert( ni == sensor_index.size() );


HMM this is relying on the single mm sensor index from old workflow
having the same meaning as the sensor index used in the new workflow. 

Suspect the the additional TT SD are messing up the indexing.::

    epsilon:stree blyth$ GEOM st
    cd /Users/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/stree
    epsilon:stree blyth$ cat sensor_name_names.txt
    PMT_3inch_log_phys
    pLPMT_NNVT_MCPPMT
    pLPMT_Hamamatsu_R12860
    mask_PMT_20inch_vetolMaskVirtual_phys
    pPanel_0_f_
    pPanel_1_f_
    pPanel_2_f_
    pPanel_3_f_
    epsilon:stree blyth$ 


Need to restrict what is treated as sensor, to avoid the unexpected pPanel 
messing up the indexing. 
Added "PMT" in name restriction to U4SensorIdentifierDefault.h  


Before the change clearly messed up s_identifier repeating (0,1,2,3,0,1,2,3,...) 
presumably from the 4 pPanel::

    2023-07-13 17:28:51.652 INFO  [264380] [CSG_GGeo_Convert::addInstances@226]  repeatIdx 1 num_inst 25600 nmm 10
    2023-07-13 17:28:51.652 INFO  [264380] [CSG_GGeo_Convert::addInstances@229]  iid 25600,5,4
    2023-07-13 17:28:51.659 INFO  [264380] [CSG_GGeo_Convert::addInstances@236]  sensor_index.size 25600
    stree::lookup_sensor_identifier.0 arg_sensor_identifier.size 0 arg_sensor_index.size 25600 sensor_id.size 46116 edge 10
    stree::lookup_sensor_identifier.1 i   0 s_index       0 s_index_inrange 1 s_identifier       0 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i   1 s_index       1 s_index_inrange 1 s_identifier       1 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i   2 s_index       2 s_index_inrange 1 s_identifier       2 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i   3 s_index       3 s_index_inrange 1 s_identifier       3 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i   4 s_index       4 s_index_inrange 1 s_identifier       0 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i   5 s_index       5 s_index_inrange 1 s_identifier       1 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i   6 s_index       6 s_index_inrange 1 s_identifier       2 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i   7 s_index       7 s_index_inrange 1 s_identifier       3 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i   8 s_index       8 s_index_inrange 1 s_identifier       0 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i   9 s_index       9 s_index_inrange 1 s_identifier       1 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i  10 ... 
    stree::lookup_sensor_identifier.1 i 25591 s_index   25591 s_index_inrange 1 s_identifier  307475 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i 25592 s_index   25592 s_index_inrange 1 s_identifier  307476 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i 25593 s_index   25593 s_index_inrange 1 s_identifier  307477 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i 25594 s_index   25594 s_index_inrange 1 s_identifier  307478 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i 25595 s_index   25595 s_index_inrange 1 s_identifier  307479 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i 25596 s_index   25596 s_index_inrange 1 s_identifier  307480 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i 25597 s_index   25597 s_index_inrange 1 s_identifier  307481 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i 25598 s_index   25598 s_index_inrange 1 s_identifier  307482 sensor_id.size   46116
    stree::lookup_sensor_identifier.1 i 25599 s_index   25599 s_index_inrange 1 s_identifier  307483 sensor_id.size   46116
    2023-07-13 17:28:51.660 INFO  [264380] [CSG_GGeo_Convert::addInstances@243]  sensor_id.size 25600
    2023-07-13 17:28:51.660 INFO  [264380] [CSG_GGeo_Convert::addInstances@244] stree::DescSensor num_sensor 25600
     i       0 s_index       1 s_identifier       0
     i       1 s_index       2 s_identifier       1
     i       2 s_index       3 s_identifier       2
     i       3 s_index       4 s_identifier       3
     i       4 s_index       5 s_identifier       0
     i       5 s_index       6 s_identifier       1
     i       6 s_index       7 s_identifier       2
     i       7 s_index       8 s_identifier       3
     i       8 s_index       9 s_identifier       0
     i       9 s_index      10 s_identifier       1
     i      10 s_index      11 s_identifier       2
     i      11 s_index      12 s_identifier       3
     i      12 s_index      13 s_identifier       0
     i      13 s_index      14 s_identifier       1
     i      14 s_index      15 s_identifier       2
     i      15 s_index      16 s_identifier       3
     i      16 s_index      17 s_identifier       0
     i      17 s_index      18 s_identifier       1
     i      18 s_index      19 s_identifier       2
     i      19 s_index      20 s_identifier       3
     i      20 s_index      21 s_identifier       0






WIP : need lpmtid GPU side for QPMT
---------------------------------------

::

    ct ; ./CSGFoundry_py_test.sh

    cf.inst[:,:,3].view(np.int32)
    [[    0     0    -1    -1]
     [    1     1     0     0]
     [    2     1     1     1]
     [    3     1     2     2]
     [    4     1     3     3]
     ...
     [48472     9    -1    -1]
     [48473     9    -1    -1]
     [48474     9    -1    -1]
     [48475     9    -1    -1]
     [48476     9    -1    -1]]

    In [1]: cf.inst.shape
    Out[1]: (48477, 4, 4)

    In [2]: sensor_identifier = cf.inst[:,2,3].view(np.int32) ; sensor_identifier
    Out[2]: array([-1,  0,  1,  2,  3, ..., -1, -1, -1, -1, -1], dtype=int32)


    In [1]: np.where( sensor_identifier == -1 )
    Out[1]: (array([    0, 25601, 25602, 25603, 25604, ..., 48472, 48473, 48474, 48475, 48476]),)

    In [2]: np.where( sensor_identifier == -1 )[0] 
    Out[2]: array([    0, 25601, 25602, 25603, 25604, ..., 48472, 48473, 48474, 48475, 48476])

    In [3]: np.where( sensor_identifier == -1 )[0].size
    Out[3]: 20477

    In [4]: np.where( sensor_index == -1 )[0].size
    Out[4]: 20477

    In [5]: sensor_identifier.size
    Out[5]: 48477

    In [6]: np.where( np.logical_and( sensor_identifier == sensor_index, sensor_index > 0 ) )
    Out[6]: (array([2, 3, 4]),)






WIP : Not getting expected sensor_id
---------------------------------------

::

    cf.inst[:,:,3].view(np.int32)
    [[    0     0    -1    -1]
     [    1     1     0     0]
     [    2     1     1     1]
     [    3     1     2     2]
     [    4     1     3     3]
     ...
     [48472     9    -1    -1]
     [48473     9    -1    -1]
     [48474     9    -1    -1]
     [48475     9    -1    -1]
     [48476     9    -1    -1]]
    (sid.min(), sid.max())
    (-1, 309883)
    (six.min(), six.max())
    (-1, 27999)
    np.c_[ugas,ngas,cf.mmlabel] 
    [[0 1 '2977:sWorld']
     [1 25600 '5:PMT_3inch_pmt_solid']
     [2 12615 '9:NNVTMCPPMTsMask_virtual']
     [3 4997 '12:HamamatsuR12860sMask_virtual']
     [4 2400 '6:mask_PMT_20inch_vetosMask_virtual']
     [5 590 '1:sStrutBallhead']
     [6 590 '1:uni1']
     [7 590 '1:base_steel']
     [8 590 '1:uni_acrylic1']
     [9 504 '130:sPanel']]
    np.c_[np.unique(sid[gas==0],return_counts=True)]     
    [[-1  1]]
    np.c_[np.unique(sid[gas==1],return_counts=True)]     
    [[     0    127]
     [     1    127]
     [     2    127]
     [     3    127]
     [     4      1]
     ...
     [307479      1]
     [307480      1]
     [307481      1]
     [307482      1]
     [307483      1]]
    np.c_[np.unique(sid[gas==2],return_counts=True)]     
    [[   -1 12615]]
    np.c_[np.unique(sid[gas==3],return_counts=True)]     
    [[  -1 4997]]
    np.c_[np.unique(sid[gas==4],return_counts=True)]     
    [[307484      1]
     [307485      1]
     [307486      1]
     [307487      1]
     [307488      1]
     ...
     [309879      1]
     [309880      1]
     [309881      1]
     [309882      1]
     [309883      1]]
    np.c_[np.unique(sid[gas==5],return_counts=True)]     
    [[ -1 590]]
    np.c_[np.unique(sid[gas==6],return_counts=True)]     
    [[ -1 590]]
    np.c_[np.unique(sid[gas==7],return_counts=True)]     
    [[ -1 590]]
    np.c_[np.unique(sid[gas==8],return_counts=True)]     
    [[ -1 590]]
    np.c_[np.unique(sid[gas==9],return_counts=True)]     
    [[ -1 504]]

    In [1]:                    


::

     40 const U4SensorIdentifier* G4CXOpticks::SensorIdentifier = nullptr ;
     41 void G4CXOpticks::SetSensorIdentifier( const U4SensorIdentifier* sid ){ SensorIdentifier = sid ; }  // static 


::

    240 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    241 {
    242     LOG(LEVEL) << " G4VPhysicalVolume world " << world ;
    243     assert(world);
    244     wd = world ;
    245 
    246     assert(sim && "sim instance should have been created in ctor" );
    247 
    248     stree* st = sim->get_tree();
    249     // TODO: sim argument, not st : or do SSim::Create inside U4Tree::Create 
    250     tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    251 
    252 
    253     // GGeo creation done when starting from a gdml or live G4,  still needs Opticks instance
    254     Opticks::Configure("--gparts_transform_offset --allownokey" );
    255 
    256     GGeo* gg_ = X4Geo::Translate(wd) ;
    257 
    258 
    259     setGeometry(gg_);
    260 }

::

    104     static U4Tree* Create( stree* st, const G4VPhysicalVolume* const top, const U4SensorIdentifier* sid=nullptr );
    105     U4Tree(stree* st, const G4VPhysicalVolume* const top=nullptr, const U4SensorIdentifier* sid=nullptr );
    106     void init();


    174 inline U4Tree::U4Tree(stree* st_, const G4VPhysicalVolume* const top_,  const U4SensorIdentifier* sid_ )
    175     :
    176     st(st_),
    177     top(top_),
    178     sid(sid_ ? sid_ : new U4SensorIdentifierDefault),
    179     level(st->level),
    180     num_surfaces(-1),
    181     rayleigh_table(CreateRayleighTable()),
    182     scint(nullptr)
    183 {
    184     init();
    185 }


Add sensor name dumping
--------------------------

Original sensor_id look OK, so maybe issue with reordering ::

    U4SensorIdentifierDefault::getIdentity copyno 325590 num_sd 2 sensor_id 325590 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325591 num_sd 2 sensor_id 325591 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325592 num_sd 2 sensor_id 325592 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325593 num_sd 2 sensor_id 325593 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325594 num_sd 2 sensor_id 325594 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325595 num_sd 2 sensor_id 325595 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325596 num_sd 2 sensor_id 325596 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325597 num_sd 2 sensor_id 325597 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325598 num_sd 2 sensor_id 325598 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325599 num_sd 2 sensor_id 325599 pvn PMT_3inch_log_phys

    U4SensorIdentifierDefault::getIdentity copyno 2 num_sd 2 sensor_id 2 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 4 num_sd 2 sensor_id 4 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 6 num_sd 2 sensor_id 6 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 21 num_sd 2 sensor_id 21 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 22 num_sd 2 sensor_id 22 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 23 num_sd 2 sensor_id 23 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 24 num_sd 2 sensor_id 24 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 25 num_sd 2 sensor_id 25 pvn pLPMT_NNVT_MCPPMT
    ...
    U4SensorIdentifierDefault::getIdentity copyno 17586 num_sd 2 sensor_id 17586 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 17587 num_sd 2 sensor_id 17587 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 17588 num_sd 2 sensor_id 17588 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 17589 num_sd 2 sensor_id 17589 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 17590 num_sd 2 sensor_id 17590 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 2 sensor_id 0 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 1 num_sd 2 sensor_id 1 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 2 sensor_id 3 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 5 num_sd 2 sensor_id 5 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 7 num_sd 2 sensor_id 7 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 8 num_sd 2 sensor_id 8 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 9 num_sd 2 sensor_id 9 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 10 num_sd 2 sensor_id 10 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 11 num_sd 2 sensor_id 11 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 12 num_sd 2 sensor_id 12 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 13 num_sd 2 sensor_id 13 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 14 num_sd 2 sensor_id 14 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 15 num_sd 2 sensor_id 15 pvn pLPMT_Hamamatsu_R12860
    ...
    U4SensorIdentifierDefault::getIdentity copyno 17606 num_sd 2 sensor_id 17606 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 17607 num_sd 2 sensor_id 17607 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 17608 num_sd 2 sensor_id 17608 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 17609 num_sd 2 sensor_id 17609 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 17610 num_sd 2 sensor_id 17610 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 17611 num_sd 2 sensor_id 17611 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 30000 num_sd 2 sensor_id 30000 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30001 num_sd 2 sensor_id 30001 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30002 num_sd 2 sensor_id 30002 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30003 num_sd 2 sensor_id 30003 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30004 num_sd 2 sensor_id 30004 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30005 num_sd 2 sensor_id 30005 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30006 num_sd 2 sensor_id 30006 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30007 num_sd 2 sensor_id 30007 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    ...
    U4SensorIdentifierDefault::getIdentity copyno 32389 num_sd 2 sensor_id 32389 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32390 num_sd 2 sensor_id 32390 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32391 num_sd 2 sensor_id 32391 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32392 num_sd 2 sensor_id 32392 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32393 num_sd 2 sensor_id 32393 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32394 num_sd 2 sensor_id 32394 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32395 num_sd 2 sensor_id 32395 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32396 num_sd 2 sensor_id 32396 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32397 num_sd 2 sensor_id 32397 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32398 num_sd 2 sensor_id 32398 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32399 num_sd 2 sensor_id 32399 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 64 sensor_id 0 pvn pPanel_0_f_
    U4SensorIdentifierDefault::getIdentity copyno 1 num_sd 64 sensor_id 1 pvn pPanel_1_f_
    U4SensorIdentifierDefault::getIdentity copyno 2 num_sd 64 sensor_id 2 pvn pPanel_2_f_
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 64 sensor_id 3 pvn pPanel_3_f_
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 64 sensor_id 0 pvn pPanel_0_f_
    ...
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 64 sensor_id 3 pvn pPanel_3_f_
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 64 sensor_id 0 pvn pPanel_0_f_
    U4SensorIdentifierDefault::getIdentity copyno 1 num_sd 64 sensor_id 1 pvn pPanel_1_f_
    U4SensorIdentifierDefault::getIdentity copyno 2 num_sd 64 sensor_id 2 pvn pPanel_2_f_
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 64 sensor_id 3 pvn pPanel_3_f_
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 64 sensor_id 0 pvn pPanel_0_f_
    U4SensorIdentifierDefault::getIdentity copyno 1 num_sd 64 sensor_id 1 pvn pPanel_1_f_
    U4SensorIdentifierDefault::getIdentity copyno 2 num_sd 64 sensor_id 2 pvn pPanel_2_f_
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 64 sensor_id 3 pvn pPanel_3_f_
    stree::add_inst i   0 gas_idx   1 nodes.size   25600
    stree::add_inst i   1 gas_idx   2 nodes.size   12615


::

    In [1]: sid.shape
    Out[1]: (48477,)

    In [2]: sid2.shape
    Out[2]: (46116,)

    In [3]: 48477 - 46116
    Out[3]: 2361


    In [26]: sid2[504:504+17612]
    Out[26]: array([    0,     1,     2,     3,     4, ..., 17607, 17608, 17609, 17610, 17611], dtype=int32)

    In [27]: np.all( np.arange(17612) == sid2[504:504+17612] )
    Out[27]: True

    In [34]: sid2[504+17612:504+17612+25600+1]
    Out[34]: array([300000, 300001, 300002, 300003, 300004, ..., 325596, 325597, 325598, 325599,  30000], dtype=int32)

    In [38]: sid2[504+17612+25600:504+17612+25600+2400]
    Out[38]: array([30000, 30001, 30002, 30003, 30004, ..., 32395, 32396, 32397, 32398, 32399], dtype=int32)


    In [39]: 17612+25600+2400
    Out[39]: 45612

    In [40]: sid2.shape
    Out[40]: (46116,)

    In [41]: 17612+25600+2400+504
    Out[41]: 46116







::

    2023-07-13 18:05:41.046 INFO  [278292] [CSG_GGeo_Convert::addInstances@229]  iid 2400,6,4
    2023-07-13 18:05:41.047 INFO  [278292] [CSG_GGeo_Convert::addInstances@236]  sensor_index.size 2400
    stree::lookup_sensor_identifier.0 arg_sensor_identifier.size 0 arg_sensor_index.size 2400 sensor_id.size 45612 edge 10
    stree::lookup_sensor_identifier.1 i   0 s_index   25600 s_index_inrange 1 s_identifier  307988 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i   1 s_index   25601 s_index_inrange 1 s_identifier  307989 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i   2 s_index   25602 s_index_inrange 1 s_identifier  307990 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i   3 s_index   25603 s_index_inrange 1 s_identifier  307991 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i   4 s_index   25604 s_index_inrange 1 s_identifier  307992 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i   5 s_index   25605 s_index_inrange 1 s_identifier  307993 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i   6 s_index   25606 s_index_inrange 1 s_identifier  307994 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i   7 s_index   25607 s_index_inrange 1 s_identifier  307995 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i   8 s_index   25608 s_index_inrange 1 s_identifier  307996 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i   9 s_index   25609 s_index_inrange 1 s_identifier  307997 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i  10 ... 
    stree::lookup_sensor_identifier.1 i 2391 s_index   27991 s_index_inrange 1 s_identifier  310379 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i 2392 s_index   27992 s_index_inrange 1 s_identifier  310380 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i 2393 s_index   27993 s_index_inrange 1 s_identifier  310381 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i 2394 s_index   27994 s_index_inrange 1 s_identifier  310382 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i 2395 s_index   27995 s_index_inrange 1 s_identifier  310383 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i 2396 s_index   27996 s_index_inrange 1 s_identifier  310384 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i 2397 s_index   27997 s_index_inrange 1 s_identifier  310385 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i 2398 s_index   27998 s_index_inrange 1 s_identifier  310386 sensor_id.size   45612
    stree::lookup_sensor_identifier.1 i 2399 s_index   27999 s_index_inrange 1 s_identifier  310387 sensor_id.size   45612
    2023-07-13 18:05:41.048 INFO  [278292] [CSG_GGeo_Convert::addInstances@243]  sensor_id.size 2400
    2023-07-13 18:05:41.048 INFO  [278292] [CSG_GGeo_Convert::addInstances@244] stree::DescSensor num_sensor 2400
     i       0 s_index   25601 s_identifier  307988
     i       1 s_index   25602 s_identifier  307989


