inst_transforms_XYZ_flips_discrepancy
=======================================

Previous :doc:`sensor_info_into_new_workflow` showed getting inst identity info to match.
BUT: now getting mismatch between the transforms. 

* HMM: I thought the transforms were matching previously, so recent changes may have broken them 

  * see :doc:`joined_up_thinking_geometry_translation`

* AHAH : could this be --gparts_transform_offset yet again ? 

  * dont think so 


Generate the geometry and grab using ntds3::

    cd ~/opticks/sysrap/tests
    ./stree_test.sh ana

    010 export STBASE=/tmp/$USER/opticks/ntds3/G4CXOpticks
     14 export FOLD=$STBASE/stree
     15 export CFBASE=$STBASE


    In [1]: cf.inst.view(np.int32)[:,:,3]
    Out[1]: 
    array([[     0,      0,     -1,     -1],
           [     1,      1, 300000,  17612],
           [     2,      1, 300001,  17613],
           [     3,      1, 300002,  17614],
           [     4,      1, 300003,  17615],
           ...,
           [ 48472,      9,     -1,     -1],
           [ 48473,      9,     -1,     -1],
           [ 48474,      9,     -1,     -1],
           [ 48475,      9,     -1,     -1],
           [ 48476,      9,     -1,     -1]], dtype=int32)

    In [2]: f.inst_f4.view(np.int32)[:,:,3]
    Out[2]: 
    array([[     0,      0,     -1,     -1],
           [     1,      1, 300000,  17612],
           [     2,      1, 300001,  17613],
           [     3,      1, 300002,  17614],
           [     4,      1, 300003,  17615],
           ...,
           [ 48472,      9,     -1,     -1],
           [ 48473,      9,     -1,     -1],
           [ 48474,      9,     -1,     -1],
           [ 48475,      9,     -1,     -1],
           [ 48476,      9,     -1,     -1]], dtype=int32)

    In [3]: np.all( cf.inst.view(np.int32)[:,:,3] == f.inst_f4.view(np.int32)[:,:,3] )
    Out[3]: True





Random sampling ~10 transforms, shows they all differ in X,Y or Z flips. 


Transforms not matching::

    In [37]: f.inst_f4[-1]
    Out[37]: 
    array([[     0.  ,      1.  ,      0.  ,      0.  ],
           [    -1.  ,      0.  ,      0.  ,      0.  ],
           [     0.  ,      0.  ,      1.  ,       nan],
           [-22672.5 ,   6711.2 ,  26504.15,       nan]], dtype=float32)

    In [38]: cf.inst[-1]
    Out[38]: 
    array([[    0.  ,     1.  ,     0.  ,     0.  ],
           [    1.  ,     0.  ,     0.  ,     0.  ],
           [    0.  ,     0.  ,     1.  ,      nan],
           [22672.5 ,  6711.2 , 26504.15,      nan]], dtype=float32)


Clear the identity info, and apply the transform. Shows have X or Y sign flip diffs::

    In [52]: a_inst = cf.inst.copy() 
    In [53]: b_inst = f.inst_f4.copy()        

    In [54]: a_inst[:,:,3] = [0,0,0,1]
    In [55]: b_inst[:,:,3] = [0,0,0,1]

    In [56]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), a_inst[10] )
    Out[56]: array([ 2694.681,  2773.886, 18994.307,     1.   ], dtype=float32)

    In [57]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), b_inst[10] )
    Out[57]: array([ 2694.681, -2773.886, 18994.307,     1.   ], dtype=float32)

    In [62]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), b_inst[-1] )
    Out[62]: array([-22672.5 ,   6711.2 ,  26504.15,      1.  ], dtype=float32)

    In [63]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), a_inst[-1] )
    Out[63]: array([22672.5 ,  6711.2 , 26504.15,     1.  ], dtype=float32)


    In [64]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), a_inst[1000] )
    Out[64]: array([ 8272.514, 16920.074,  4584.33 ,     1.   ], dtype=float32)

    In [65]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), b_inst[1000] )
    Out[65]: array([ -8272.514, -16920.074,  -4584.33 ,      1.   ], dtype=float32)


Hmm : to debug this need to see the transform stack being used in both cases.::

    In [70]: np.all( cf.inst.view(np.int32)[:,:,3]  == f.inst_f4.view(np.int32)[:,:,3] )
    Out[70]: True

    In [71]: iid = cf.inst.view(np.int32)[:,:,3]

    In [75]: iid
    Out[75]: 
    array([[     0,      0,     -1,     -1],
           [     1,      1, 300000,  17612],
           [     2,      1, 300001,  17613],
           [     3,      1, 300002,  17614],
           [     4,      1, 300003,  17615],
           ...,
           [ 48472,      9,     -1,     -1],
           [ 48473,      9,     -1,     -1],
           [ 48474,      9,     -1,     -1],
           [ 48475,      9,     -1,     -1],
           [ 48476,      9,     -1,     -1]], dtype=int32)

    In [78]: np.all( iid[:,0] == np.arange(len(iid)) )   ## 1st column is ins_idx
    Out[78]: True

    In [77]: iid[np.where( iid[:,1] == 2 )]
    Out[77]: 
    array([[25601,     2,     2,     2],
           [25602,     2,     4,     4],
           [25603,     2,     6,     6],
           [25604,     2,    21,    21],
           [25605,     2,    22,    22],
           ...,
           [38211,     2, 17586, 17586],
           [38212,     2, 17587, 17587],
           [38213,     2, 17588, 17588],
           [38214,     2, 17589, 17589],
           [38215,     2, 17590, 17590]], dtype=int32)

    In [81]: iid[np.where( iid[:,1] == 3 )]
    Out[81]: 
    array([[38216,     3,     0,     0],
           [38217,     3,     1,     1],
           [38218,     3,     3,     3],
           [38219,     3,     5,     5],
           [38220,     3,     7,     7],
           ...,
           [43208,     3, 17607, 17607],
           [43209,     3, 17608, 17608],
           [43210,     3, 17609, 17609],
           [43211,     3, 17610, 17610],
           [43212,     3, 17611, 17611]], dtype=int32)

    In [82]: a_inst[38216]
    Out[82]: 
    array([[    1.   ,     0.   ,     0.   ,     0.   ],
           [    0.   ,     1.   ,     0.   ,     0.   ],
           [    0.   ,     0.   ,     1.   ,     0.   ],
           [  930.298,   111.872, 19365.   ,     1.   ]], dtype=float32)

    In [83]: b_inst[38216]
    Out[83]: 
    array([[   -1.   ,     0.   ,    -0.   ,     0.   ],
           [    0.   ,     1.   ,     0.   ,     0.   ],
           [    0.   ,     0.   ,    -1.   ,     0.   ],
           [ -930.298,  -111.872, 19365.   ,     1.   ]], dtype=float32)


::


    In [84]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), a_inst[38216] )
    Out[84]: array([  930.298,   111.872, 19365.   ,     1.   ], dtype=float32)

    In [85]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), b_inst[38216] )
    Out[85]: array([ -930.298,  -111.872, 19365.   ,     1.   ], dtype=float32)


    In [89]: origin = np.array([0,0,0,1], dtype=np.float32 )

    In [92]: ii = 38216
    In [93]: ii, np.dot( origin, a_inst[ii] ), np.dot( origin, b_inst[ii] ) 
    Out[93]: 
    (38216,
     array([  930.298,   111.872, 19365.   ,     1.   ], dtype=float32),
     array([ -930.298,  -111.872, 19365.   ,     1.   ], dtype=float32))

    In [96]: ii, np.dot( origin, a_inst[ii] ), np.dot( origin, b_inst[ii] )
    Out[96]: 
    (48472,
     array([20133.6  ,  9250.101, 26489.85 ,     1.   ], dtype=float32),
     array([-20133.6  ,   9250.101,  26489.85 ,      1.   ], dtype=float32))



::

    In [97]: a_inst[40000]
    Out[97]: 
    array([[    0.138,     0.254,     0.957,     0.   ],
           [    0.879,     0.477,     0.   ,     0.   ],
           [    0.457,     0.841,     0.29 ,     0.   ],
           [ 8881.754, 16344.179,  5626.955,     1.   ]], dtype=float32)

    In [98]: b_inst[40000]
    Out[98]: 
    array([[   -0.138,    -0.254,     0.957,     0.   ],
           [   -0.879,     0.477,     0.   ,     0.   ],
           [   -0.457,    -0.841,    -0.29 ,     0.   ],
           [ 8881.754, 16344.179,  5626.955,     1.   ]], dtype=float32)

    In [100]: ii=40000 ; ii, np.dot( origin, a_inst[ii] ), np.dot( origin, b_inst[ii] )
    Out[100]: 
    (40000,
     array([ 8881.754, 16344.179,  5626.955,     1.   ], dtype=float32),
     array([ 8881.754, 16344.179,  5626.955,     1.   ], dtype=float32))

    In [101]: plus_z = np.array( [0,0,100,1], dtype=np.float32 )

    In [102]: ii=40000 ; ii, np.dot( plus_z, a_inst[ii] ), np.dot( plus_z, b_inst[ii] )
    Out[102]: 
    (40000,
     array([ 8927.456, 16428.28 ,  5655.909,     1.   ], dtype=float32),
     array([ 8836.052, 16260.078,  5598.001,     1.   ], dtype=float32))



How to debug ?
-----------------

The stree m2w w2m nds means that have all the transforms and ancestry info.
So should be able to reproduce the stree transforms from the m2w. 

Hmm but need the nidx of each instance ? Added that to stree::

    In [1]: f.inst_nidx
    Out[1]: array([     0, 194249, 194254, 194259, 194264, ...,  65071,  65202,  65332,  65462,  65592], dtype=int32)

    In [2]: f.inst_nidx.shape
    Out[2]: (48477,)



::

    f.base:/tmp/blyth/opticks/ntds3/G4CXOpticks/stree

      : f.sensor_id                                        :             (45612,) : 0:59:16.217105 

      : f.subs                                             :               336653 : 0:59:16.186542 
      : f.nds                                              :         (336653, 11) : 0:59:16.218986 
      : f.digs                                             :               336653 : 0:59:17.510443 
      : f.m2w                                              :       (336653, 4, 4) : 0:59:16.441178 
      : f.w2m                                              :       (336653, 4, 4) : 0:59:15.095604 

      : f.inst                                             :        (48477, 4, 4) : 0:59:17.038016 
      : f.inst_f4                                          :        (48477, 4, 4) : 0:59:17.015918 
      : f.iinst_f4                                         :        (48477, 4, 4) : 0:59:17.054821 
      : f.iinst                                            :        (48477, 4, 4) : 0:59:17.491596 

      : f.soname                                           :                  139 : 0:59:16.216731 
      : f.mtname                                           :                   20 : 0:59:16.436847 
      : f.factor                                           :              (9, 11) : 0:59:17.509037 





U4Tree/stree side rather simple, difficult to see anything wrong with it
--------------------------------------------------------------------------

::

    1338 inline void stree::add_inst()
    1339 {
    1340     glm::tmat4x4<double> tr_m2w(1.) ;
    1341     glm::tmat4x4<double> tr_w2m(1.) ;
    1342     add_inst(tr_m2w, tr_w2m, 0, 0 );   // global instance with identity transforms 
    1343 
    1344     unsigned num_factor = get_num_factor();
    1345     for(unsigned i=0 ; i < num_factor ; i++)
    1346     {
    1347         std::vector<int> nodes ;
    1348         get_factor_nodes(nodes, i);
    1349 
    1350         unsigned gas_idx = i + 1 ; // 0 is the global instance, so need this + 1  
    1351         std::cout
    1352             << "stree::add_inst"
    1353             << " i " << std::setw(3) << i
    1354             << " gas_idx " << std::setw(3) << gas_idx
    1355             << " nodes.size " << std::setw(7) << nodes.size()
    1356             << std::endl
    1357             ;
    1358 
    1359         for(unsigned j=0 ; j < nodes.size() ; j++)
    1360         {
    1361             int nidx = nodes[j];
    1362             get_m2w_product(tr_m2w, nidx, false);
    1363             get_w2m_product(tr_w2m, nidx, true );
    1364 
    1365             add_inst(tr_m2w, tr_w2m, gas_idx, nidx );
    1366         }
    1367     }
    1368 
    1369     strid::Narrow( inst_f4,   inst );
    1370     strid::Narrow( iinst_f4, iinst );
    1371 }

    0779 inline void stree::get_m2w_product( glm::tmat4x4<double>& transform, int nidx, bool reverse ) const
     780 {
     781     std::vector<int> nodes ;
     782     get_ancestors(nodes, nidx);
     783     nodes.push_back(nidx); 
     784     
     785     unsigned num_nodes = nodes.size();
     786     glm::tmat4x4<double> xform(1.);
     787     
     788     for(unsigned i=0 ; i < num_nodes ; i++ )
     789     {   
     790         int idx = nodes[reverse ? num_nodes - 1 - i : i] ;
     791         const glm::tmat4x4<double>& t = get_m2w(idx) ;
     792         xform *= t ;
     793     }
     794     assert( sizeof(glm::tmat4x4<double>) == sizeof(double)*16 ); 
     795     memcpy( glm::value_ptr(transform), glm::value_ptr(xform), sizeof(glm::tmat4x4<double>) );
     796 }

    0754 inline const glm::tmat4x4<double>& stree::get_m2w(int nidx) const
     755 {
     756     assert( nidx > -1 && nidx < m2w.size());
     757     return m2w[nidx] ;
     758 }


    193 inline int U4Tree::convertNodes_r( const G4VPhysicalVolume* const pv, int depth, int sibdex, snode* parent )
    194 {
    195     const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    196 
    197     int num_child = int(lv->GetNoDaughters()) ;
    198     int lvid = lvidx[lv] ;
    199 
    200     const G4PVPlacement* pvp = dynamic_cast<const G4PVPlacement*>(pv) ;
    201     int copyno = pvp ? pvp->GetCopyNo() : -1 ;
    202 
    203     glm::tmat4x4<double> tr_m2w(1.) ;
    204     U4Transform::GetObjectTransform(tr_m2w, pv);
    205 
    206     glm::tmat4x4<double> tr_w2m(1.) ;
    207     U4Transform::GetFrameTransform(tr_w2m, pv);
    208 
    209 
    210     st->m2w.push_back(tr_m2w);
    211     st->w2m.push_back(tr_w2m);
    212     pvs.push_back(pv);
    213 
    214     int nidx = st->nds.size() ;
    215 
    216     snode nd ;
    217 
    218     nd.index = nidx ;
    219     nd.depth = depth ;
    220     nd.sibdex = sibdex ;
    221     nd.parent = parent ? parent->index : -1 ;



GMesh/CSG_GGeo/CSGFoundry
-----------------------------

::

    1545 /**
    1546 CSGFoundry::addInstance
    1547 ------------------------
    1548    
    1549 Used for example from 
    1550 
    1551 1. CSG_GGeo_Convert::addInstances when creating CSGFoundry from GGeo
    1552 2. CSGCopy::copy/CSGCopy::copySolidInstances when copy a loaded CSGFoundry to apply a selection
    1553 
    1554 **/
    1555    
    1556 void CSGFoundry::addInstance(const float* tr16, int gas_idx, int sensor_identifier, int sensor_index )
    1557 {
    1558     qat4 instance(tr16) ;  // identity matrix if tr16 is nullptr 
    1559     int ins_idx = int(inst.size()) ;
    1560 
    1561     instance.setIdentity( ins_idx, gas_idx, sensor_identifier, sensor_index );
    1562    
    1563     LOG(debug)
    1564         << " ins_idx " << ins_idx 
    1565         << " gas_idx " << gas_idx 
    1566         << " sensor_identifier " << sensor_identifier
    1567         << " sensor_index " << sensor_index
    1568         ;
    1569    
    1570     inst.push_back( instance );
    1571 }


    0205 void CSG_GGeo_Convert::addInstances(unsigned repeatIdx )
     206 {
     ...
     243     for(unsigned i=0 ; i < num_inst ; i++)
     244     {
     245         int s_identifier = sensor_id[i] ;
     246         int s_index_1 = sensor_index[i] ;    // 1-based sensor index, 0 meaning not-a-sensor 
     247         int s_index_0 = s_index_1 - 1 ;      // 0-based sensor index, -1 meaning not-a-sensor
     248         // this simple correction relies on consistent invalid index, see GMergedMesh::Get3DFouthColumnNonZero
     249 
     250         glm::mat4 it = mm->getITransform_(i);
     251    
     252         const float* tr16 = glm::value_ptr(it) ;
     253         unsigned gas_idx = repeatIdx ;
     254         foundry->addInstance(tr16, gas_idx, s_identifier, s_index_0 );
     255     }
     256 }


::

    1146 float* GMesh::getTransform(unsigned index) const
    1147 {   
    1148     if(index >= m_num_volumes)
    1149     {   
    1150         LOG(fatal) << "GMesh::getTransform out of bounds "
    1151                      << " m_num_volumes " << m_num_volumes
    1152                      << " index " << index
    1153                      ;
    1154         assert(0);
    1155     }
    1156     return index < m_num_volumes ? m_transforms + index*16 : NULL  ;
    1157 }
    1158 
    1159 glm::mat4 GMesh::getTransform_(unsigned index) const
    1160 {
    1161     float* transform = getTransform(index) ;
    1162     glm::mat4 tr = glm::make_mat4(transform) ;
    1163     return tr ;
    1164 }
    1165 
    1166 float* GMesh::getITransform(unsigned index) const
    1167 {
    1168     unsigned int num_itransforms = getNumITransforms();
    1169     return index < num_itransforms ? m_itransforms + index*16 : NULL  ;
    1170 }
    1171 
    1172 glm::mat4 GMesh::getITransform_(unsigned index) const
    1173 {
    1174     float* transform = getITransform(index) ;
    1175     glm::mat4 tr = glm::make_mat4(transform) ;
    1176     return tr ;
    1177 }
    1178 


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


    032 /**
     33 GTree::makeInstanceTransformsBuffer
     34 -------------------------------------
     35 
     36 Returns transforms array of shape (num_placements, 4, 4)
     37 
     38 Collects transforms from GNode placement instances into a buffer.
     39 getPlacement for ridx=0 just returns m_root (which always has identity transform)
     40 for ridx > 0 returns all GNode subtree bases of the ridx repeats.
     41 
     42 Just getting transforms from one place to another, 
     43 not multiplying them so float probably OK. 
     44 
     45 TODO: faster to allocate in one go and set, instead of using NPY::add
     46 
     47 **/
     48 
     49 NPY<float>* GTree::makeInstanceTransformsBuffer(const std::vector<const GNode*>& placements) // static
     50 {
     51     LOG(LEVEL) << "[" ;
     52     unsigned numPlacements = placements.size();
     53     NPY<float>* buf = NPY<float>::make(0, 4, 4);
     54     for(unsigned i=0 ; i < numPlacements ; i++)
     55     {
     56         const GNode* place = placements[i] ;
     57         GMatrix<float>* t = place->getTransform();
     58         buf->add(t->getPointer(), 4*4*sizeof(float) );
     59     }
     60     assert(buf->getNumItems() == numPlacements);
     61     LOG(LEVEL) << "]" ;
     62     return buf ;
     63 }


::

    141 GMatrixF* GNode::getTransform() const
    142 {
    143    return m_transform ;
    144 }

    045 GNode::GNode(unsigned int index, GMatrixF* transform, const GMesh* mesh)
     46     :
     47     m_selfdigest(true),
     48     m_csgskip(false),
     49     m_selected(true),
     50     m_index(index),
     51     m_parent(NULL),
     52     m_description(NULL),
     53     m_transform(transform),
     54     m_ltransform(NULL),
     55     m_gtriple(NULL),
     56     m_ltriple(NULL),


::

    1679 GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
    1680 {
    1685     // record copynumber in GVolume, as thats one way to handle pmtid
    1686     const G4PVPlacement* placement = dynamic_cast<const G4PVPlacement*>(pv);
    1687     assert(placement);
    1688     G4int copyNumber = placement->GetCopyNo() ;
    1689 
    1690     X4Nd* parent_nd = parent ? static_cast<X4Nd*>(parent->getParallelNode()) : NULL ;
    1691 
    1692     unsigned boundary = addBoundary( pv, pv_p );
    1693     std::string boundaryName = m_blib->shortname(boundary);
    1694     int materialIdx = m_blib->getInnerMaterial(boundary);
    1695 
    1696 
    1697     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
    1698     const std::string& lvName = lv->GetName() ;
    1699     const std::string& pvName = pv->GetName() ;
    1700     unsigned ndIdx = m_node_count ;       // incremented below after GVolume instanciation
    1701 
    1702     int lvIdx = m_lvidx[lv] ;   // from postorder traverse in convertSolids to match GDML lvIdx : mesh identity uses lvIdx
    ....
    1747     glm::mat4 xf_local_t = X4Transform3D::GetObjectTransform(pv);
    ....
    1784     const nmat4triple* ltriple = m_xform->make_triple( glm::value_ptr(xf_local_t) ) ;   // YIKES does polardecomposition + inversion and checks them 
    1790 
    1791     GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(xf_local_t));
    1792 
    1797     X4Nd* nd = new X4Nd { parent_nd, ltriple } ;         // X4Nd just struct { parent, transform }
    1798 
    1799     const nmat4triple* gtriple = nxform<X4Nd>::make_global_transform(nd) ;  // product of transforms up the tree
    ....
    1805 
    1806     glm::mat4 xf_global = gtriple->t ;
    1807 
    1808     GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));
    ....
    1834     G4PVPlacement* _placement = const_cast<G4PVPlacement*>(placement) ;
    1835     void* origin_node = static_cast<void*>(_placement) ;
    1836     int origin_copyNumber = copyNumber ;
    1837 
    1838 
    1839     GVolume* volume = new GVolume(ndIdx, gtransform, mesh, origin_node, origin_copyNumber );
    1840     volume->setBoundary( boundary );   // must setBoundary before adding sensor volume 



stree::desc_m2w_product
---------------------------

Hmm its easy to access the full transform stack with stree. 
Must less so with the old way. 


::

    284 void test_desc_m2w_product(const stree& st)
    285 {
    286     int ins_idx = ssys::getenvint("INS_IDX", 1 );
    287     int num_inst = int(st.inst_nidx.size()) ; 
    288     if(ins_idx < 0 ) ins_idx += num_inst ;
    289     assert( ins_idx < num_inst );
    290     
    291     int nidx = st.inst_nidx[ins_idx] ;
    292     std::cout
    293          << "st.inst_nidx.size " << num_inst
    294          << " ins_idx INS_IDX " << ins_idx
    295          << " nidx " << nidx
    296          << std::endl
    297          ;
    298 
    299     bool reverse = false ;
    300     std::cout << st.desc_m2w_product(nidx, reverse) << std::endl ;
    301 }


    epsilon:tests blyth$ INS_IDX=-1 ./stree_test.sh build_run
    stree::load_ /tmp/blyth/opticks/ntds3/G4CXOpticks/stree
    st.desc_sub(false)
        0 : 1af760275cafe9ea890bfa01b0acb1d1 : 25600 de:( 6  6) 1st:194249 PMT_3inch_pmt_solid
        1 : 0077df3ebff8aeec56c8a21518e3c887 : 12615 de:( 6  6) 1st: 70979 NNVTMCPPMTsMask_virtual
        2 : 1e410142530e54d54db8aaaccb63b834 :  4997 de:( 6  6) 1st: 70965 HamamatsuR12860sMask_virtual
        3 : 019f9eccb5cf94cce23ff7501c807475 :  2400 de:( 4  4) 1st:322253 mask_PMT_20inch_vetosMask_virtual
        4 : c051c1bb98b71ccb15b0cf9c67d143ee :   590 de:( 6  6) 1st: 68493 sStrutBallhead
        5 : 5e01938acb3e0df0543697fc023bffb1 :   590 de:( 6  6) 1st: 69083 uni1
        6 : cdc824bf721df654130ed7447fb878ac :   590 de:( 6  6) 1st: 69673 base_steel
        7 : 3fd85f9ee7ca8882c8caa747d0eef0b3 :   590 de:( 6  6) 1st: 70263 uni_acrylic1
        8 : c68bd8ca598e7b6eabad75f107da5132 :   504 de:( 7  7) 1st:    15 sPanel

    st.inst_nidx.size 48477 ins_idx INS_IDX 48476 nidx 65592
    stree::desc_m2w_product nidx 65592 reverse 0 num_nodes 8 nodes [ 0 1 5 6 12 64679 65201 65592]
     i 0 idx 0 so sWorld
             t                                             xform

         1.000      0.000      0.000      0.000                1.000      0.000      0.000      0.000 
         0.000      1.000      0.000      0.000                0.000      1.000      0.000      0.000 
         0.000      0.000      1.000      0.000                0.000      0.000      1.000      0.000 
         0.000      0.000      0.000      1.000                0.000      0.000      0.000      1.000 

     i 1 idx 1 so sTopRock
             t                                             xform

         1.000      0.000      0.000      0.000                1.000      0.000      0.000      0.000 
         0.000      1.000      0.000      0.000                0.000      1.000      0.000      0.000 
         0.000      0.000      1.000      0.000                0.000      0.000      1.000      0.000 
      3125.000      0.000  36750.000      1.000             3125.000      0.000  36750.000      1.000 

     i 2 idx 5 so sExpRockBox
             t                                             xform

         1.000      0.000      0.000      0.000                1.000      0.000      0.000      0.000 
         0.000      1.000      0.000      0.000                0.000      1.000      0.000      0.000 
         0.000      0.000      1.000      0.000                0.000      0.000      1.000      0.000 
         0.000      0.000  -9500.000      1.000             3125.000      0.000  27250.000      1.000 

     i 3 idx 6 so sExpHall
             t                                             xform

         1.000      0.000      0.000      0.000                1.000      0.000      0.000      0.000 
         0.000      1.000      0.000      0.000                0.000      1.000      0.000      0.000 
         0.000      0.000      1.000      0.000                0.000      0.000      1.000      0.000 
     -3125.000      0.000      0.000      1.000                0.000      0.000  27250.000      1.000 

     i 4 idx 12 so sAirTT
             t                                             xform

         1.000      0.000      0.000      0.000                1.000      0.000      0.000      0.000 
         0.000      1.000      0.000      0.000                0.000      1.000      0.000      0.000 
         0.000      0.000      1.000      0.000                0.000      0.000      1.000      0.000 
         0.000      0.000  -1298.000      1.000                0.000      0.000  25952.000      1.000 

     i 5 idx 64679 so sWall
             t                                             xform

         0.000      1.000      0.000      0.000                0.000      1.000      0.000      0.000 
        -1.000      0.000      0.000      0.000               -1.000      0.000      0.000      0.000 
         0.000      0.000      1.000      0.000                0.000      0.000      1.000      0.000 
    -20133.600   6711.200    545.000      1.000           -20133.600   6711.200  26497.000      1.000 

     i 6 idx 65201 so sPlane
             t                                             xform

         1.000      0.000      0.000      0.000                0.000      1.000      0.000      0.000 
         0.000      1.000      0.000      0.000               -1.000      0.000      0.000      0.000 
         0.000      0.000      1.000      0.000                0.000      0.000      1.000      0.000 
         0.000      0.000      7.150      1.000           -20133.600   6711.200  26504.150      1.000 

     i 7 idx 65592 so sPanel
             t                                             xform

         1.000      0.000      0.000      0.000                0.000      1.000      0.000      0.000 
         0.000      1.000      0.000      0.000               -1.000      0.000      0.000      0.000 
         0.000      0.000      1.000      0.000                0.000      0.000      1.000      0.000 
         0.000   2538.900      0.000      1.000           -22672.500   6711.200  26504.150      1.000 





Transform prep in old workflow
--------------------------------

::

     065 
      66 #include "NXform.hpp"  // header with the implementation
      67 template struct nxform<X4Nd> ;
      68 

     139 X4PhysicalVolume::X4PhysicalVolume(GGeo* ggeo, const G4VPhysicalVolume* const top)
     140     :
     141     X4Named("X4PhysicalVolume"),
     142     m_ggeo(ggeo),
     143     m_top(top),
     144     m_ok(m_ggeo->getOpticks()),
     145     m_lvsdname(m_ok->getLVSDName()),
     146     m_query(m_ok->getQuery()),
     147     m_gltfpath(m_ok->getGLTFPath()),
     148 
     149     m_mlib(m_ggeo->getMaterialLib()),
     150     m_sclib(m_ggeo->getScintillatorLib()),
     151     m_slib(m_ggeo->getSurfaceLib()),
     152     m_blib(m_ggeo->getBndLib()),
     153     m_hlib(m_ggeo->getMeshLib()),
     154     //m_meshes(m_hlib->getMeshes()), 
     155     m_xform(new nxform<X4Nd>(0,false)),
     156     m_verbosity(m_ok->getVerbosity()),


::

    117 /**
    118 nxform<N>::make_global_transform
    119 -----------------------------------
    120 
    121 node structs that can work with this require
    122 transform and parent members   
    123 
    124 1. collects nmat4triple pointers whilst following 
    125    parent links up the tree, ie in leaf-to-root order 
    126 
    127 2. returns the reversed product of those 
    128 
    129 
    130 **/
    131 
    132 template <typename N>
    133 const nmat4triple* nxform<N>::make_global_transform(const N* n) // static
    134 {
    135     std::vector<const nmat4triple*> tvq ;
    136     while(n)
    137     {
    138         if(n->transform) tvq.push_back(n->transform);
    139         n = n->parent ;
    140     }
    141     bool reverse = true ; // as tvq in leaf-to-root order
    142     return tvq.size() == 0 ? NULL : nmat4triple::product(tvq, reverse) ;
    143 }



Current stree transforms match the CF transforms from aug5
-------------------------------------------------------------

* this suggests that something has broken the CF transforms since then and the stree ones are OK

::

    In [6]: cf 
    Out[6]: 
    /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry
    min_stamp:2022-08-15 10:09:17.554576
    max_stamp:2022-08-15 10:09:20.473688
    age_stamp:0:07:30.637037
             node :        (23518, 4, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/node.npy 
             itra :         (8159, 4, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/itra.npy 
         meshname :               (139,)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/meshname.txt 
             meta :                 (7,)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/meta.txt 
         primname :              (3248,)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/primname.txt 
          mmlabel :                (10,)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/mmlabel.txt 
             tran :         (8159, 4, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/tran.npy 
             inst :        (48477, 4, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/inst.npy 
            solid :           (10, 3, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/solid.npy 
             prim :         (3248, 4, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/prim.npy 

    In [7]: a_inst[0]
    Out[7]: 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    In [8]: b_inst[0]
    Out[8]: 
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1., nan],
           [ 0.,  0.,  0., nan]], dtype=float32)

    In [9]: a_inst[:,:,:3]
    Out[9]: 
    array([[[     1.   ,      0.   ,      0.   ],
            [     0.   ,      1.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [     0.   ,      0.   ,      0.   ]],

           [[     0.877,     -0.431,      0.215],
            [    -0.441,     -0.897,      0.   ],
            [     0.193,     -0.095,     -0.977],
            [ -3734.247,   1835.066,  18932.178]],

           [[     0.879,     -0.432,      0.2  ],
            [    -0.441,     -0.897,      0.   ],
            [     0.179,     -0.088,     -0.98 ],
            [ -3470.825,   1705.616,  18994.307]],

           [[    -0.338,      0.92 ,      0.2  ],
            [     0.939,      0.345,      0.   ],
            [    -0.069,      0.187,     -0.98 ],
            [  1333.472,  -3630.097,  18994.307]],

           [[    -0.337,      0.917,      0.215],
            [     0.939,      0.345,      0.   ],
            [    -0.074,      0.201,     -0.977],
            [  1434.678,  -3905.607,  18932.178]],

           ...,

           [[     1.   ,      0.   ,      0.   ],
            [     0.   ,      1.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [-20133.6  ,   9250.1  ,  26489.85 ]],

           [[     0.   ,      1.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [-17594.7  ,   6711.2  ,  26504.15 ]],

           [[     0.   ,      1.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [-19287.299,   6711.2  ,  26504.15 ]],

           [[     0.   ,      1.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [-20979.9  ,   6711.2  ,  26504.15 ]],

           [[     0.   ,      1.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [-22672.5  ,   6711.2  ,  26504.15 ]]], dtype=float32)

    In [10]: b_inst[:,:,:3]
    Out[10]: 
    array([[[     1.   ,      0.   ,      0.   ],
            [     0.   ,      1.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [     0.   ,      0.   ,      0.   ]],

           [[     0.877,     -0.431,      0.215],
            [    -0.441,     -0.897,      0.   ],
            [     0.193,     -0.095,     -0.977],
            [ -3734.247,   1835.066,  18932.178]],

           [[     0.879,     -0.432,      0.2  ],
            [    -0.441,     -0.897,      0.   ],
            [     0.179,     -0.088,     -0.98 ],
            [ -3470.825,   1705.616,  18994.307]],

           [[    -0.338,      0.92 ,      0.2  ],
            [     0.939,      0.345,      0.   ],
            [    -0.069,      0.187,     -0.98 ],
            [  1333.472,  -3630.097,  18994.307]],

           [[    -0.337,      0.917,      0.215],
            [     0.939,      0.345,      0.   ],
            [    -0.074,      0.201,     -0.977],
            [  1434.678,  -3905.607,  18932.178]],

           ...,

           [[     1.   ,      0.   ,      0.   ],
            [     0.   ,      1.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [-20133.6  ,   9250.101,  26489.85 ]],

           [[     0.   ,      1.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [-17594.7  ,   6711.2  ,  26504.15 ]],

           [[     0.   ,      1.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [-19287.299,   6711.2  ,  26504.15 ]],

           [[     0.   ,      1.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [-20979.9  ,   6711.2  ,  26504.15 ]],

           [[     0.   ,      1.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ],
            [-22672.5  ,   6711.2  ,  26504.15 ]]], dtype=float32)

    In [11]: np.abs( a_inst[:,:,:3] - b_inst[:,:,:3] ).max()
    Out[11]: 0.0009765625

    In [12]: np.abs( a_inst[:,:,:3] - b_inst[:,:,:3] ).min()
    Out[12]: 0.0

    In [13]: f
    Out[13]: 
    f

    CMDLINE:/Users/blyth/opticks/sysrap/tests/stree_test.py
    f.base:/tmp/blyth/opticks/ntds3/G4CXOpticks/stree

      : f.subs                                             :               336653 : 13:21:45.538088 
      : f.sensor_id                                        :             (45612,) : 13:21:45.570969 
      : f.soname                                           :                  139 : 13:21:45.570567 
      : f.iinst_f4                                         :        (48477, 4, 4) : 13:21:46.834187 
      : f.nds                                              :         (336653, 11) : 13:21:45.573755 
      : f.digs                                             :               336653 : 13:21:47.201406 
      : f.m2w                                              :       (336653, 4, 4) : 13:21:45.626505 
      : f.inst                                             :        (48477, 4, 4) : 13:21:46.811051 
      : f.inst_f4                                          :        (48477, 4, 4) : 13:21:46.606344 
      : f.inst_nidx                                        :             (48477,) : 13:21:46.078044 
      : f.mtname                                           :                   20 : 13:21:45.623294 
      : f.iinst                                            :        (48477, 4, 4) : 13:21:46.848103 
      : f.w2m                                              :       (336653, 4, 4) : 13:21:44.752833 
      : f.factor                                           :              (9, 11) : 13:21:47.200370 

     min_stamp : 2022-08-14 21:00:03.656953 
     max_stamp : 2022-08-14 21:00:06.105526 
     dif_stamp : 0:00:02.448573 
     age_stamp : 13:21:44.752833 

    In [14]:                                   



Hmm : given that GGeo has not long to live better to get the transform stack from Geant4 model ?
---------------------------------------------------------------------------------------------------

BUT that is almost what stree is doing, so will probably not help.  

* contrast the simple stree transform approach with the GGeo/GNode approach and 
  add what is needed to allow easy access to stack ?


* HMM: howabout debug discrepancy by populating an stree from X4PhysicalVolume::convertNode
  analogously to U4Tree::convertNodes_r 


sysrap/tests/stree_test.sh::

     10 export STBASE=/tmp/$USER/opticks/ntds3/G4CXOpticks
     15 export FOLD=$STBASE/stree

sysrap/tests/stree_test.py::

     89     f = Fold.Load(symbol="f")
     90     print(repr(f))
     91 
     92     g = Fold.Load(os.path.join(os.path.dirname(f.base), "GGeo/stree"), symbol="g")
     93 

::

    In [9]: np.abs( f.m2w - g.m2w ).max()
    Out[9]: 0.0009731784812174737


Collecting the base transforms into stree.h during GGeo creation shows 
no significant diffs. 

How to proceed with debug.  

Bring over more of the stree recording to collection within GGeo, eg from stree::add_inst




Check GVolume::getTransform by collection of analogous transforms into GGeo/stree and U4Tree/stree
-----------------------------------------------------------------------------------------------------

Collect all the GVolume::getTransform into GGeo/stree/gtd.npy from X4PhysicalVolume::convertStructure_r
and do the analogous collection in U4Tree::convertNodes_r. 


::

    032 /**
     33 GTree::makeInstanceTransformsBuffer
     34 -------------------------------------
     35 
     36 Returns transforms array of shape (num_placements, 4, 4)
     37 
     38 Collects transforms from GNode placement instances into a buffer.
     39 getPlacement for ridx=0 just returns m_root (which always has identity transform)
     40 for ridx > 0 returns all GNode subtree bases of the ridx repeats.
     41 
     42 Just getting transforms from one place to another, 
     43 not multiplying them so float probably OK. 
     44 
     45 TODO: faster to allocate in one go and set, instead of using NPY::add
     46 
     47 **/
     48 
     49 NPY<float>* GTree::makeInstanceTransformsBuffer(const std::vector<const GNode*>& placements) // static
     50 {
     51     LOG(LEVEL) << "[" ;
     52     unsigned numPlacements = placements.size();
     53     NPY<float>* buf = NPY<float>::make(0, 4, 4);
     54     for(unsigned i=0 ; i < numPlacements ; i++)
     55     {
     56         const GNode* place = placements[i] ;
     57         GMatrix<float>* t = place->getTransform();
     58         buf->add(t->getPointer(), 4*4*sizeof(float) );
     59     }
     60     assert(buf->getNumItems() == numPlacements);
     61     LOG(LEVEL) << "]" ;
     62     return buf ;
     63 }
     64 

    141 GMatrixF* GNode::getTransform() const
    142 {
    143    return m_transform ;
    144 }

    045 GNode::GNode(unsigned int index, GMatrixF* transform, const GMesh* mesh)
     46     :
     47     m_selfdigest(true),
     48     m_csgskip(false),
     49     m_selected(true),
     50     m_index(index),
     51     m_parent(NULL),
     52     m_description(NULL),
     53     m_transform(transform),
     54     m_ltransform(NULL),
     55     m_gtriple(NULL),


::

    1433 GVolume* X4PhysicalVolume::convertStructure_r(const G4VPhysicalVolume* const pv,
    1434         GVolume* parent, int depth, int sibdex, int parent_nidx,
    1435         const G4VPhysicalVolume* const parent_pv, bool& recursive_select )
    1436 {
    ....
    1465      glm::tmat4x4<double> tr_gtd(1.) ;   // GGeo Transform Debug   
    1466      GMatrixF* transform = volume->getTransform();
    1467      float* tr = (float*)transform->getPointer() ;
    1468      strid::Read(tr_gtd, tr, false );   // transpose:false the getPointer does a transpose
    1469 
    ....
    1474      snode nd ;
    1475      nd.index = nidx ;
    1476      nd.depth = depth ;
    1477      nd.sibdex = sibdex ;
    1478      nd.parent = parent_nidx ;
    1479 
    1480      nd.num_child = num_child ;
    1481      nd.first_child = -1 ;     // gets changed inplace from lower recursion level 
    1482      nd.next_sibling = -1 ;
    1483      nd.lvid = lvid ;
    1484      nd.copyno = copyno ;
    1485 
    1486      nd.sensor_id = -1 ;
    1487      nd.sensor_index = -1 ;
    1488 
    1489      m_tree->nds.push_back(nd);
    1490      m_tree->m2w.push_back(tr_m2w);
    1491      m_tree->gtd.push_back(tr_gtd);
    1492 
    1493      if(sibdex == 0 && nd.parent > -1) m_tree->nds[nd.parent].first_child = nd.index ;
    1494      // record first_child nidx into parent snode



    192 inline int U4Tree::convertNodes_r( const G4VPhysicalVolume* const pv, int depth, int sibdex, int parent )
    193 {
    ...
    210     int nidx = st->nds.size() ;  // 0-based node index
    211 
    212     snode nd ;
    213 
    214     nd.index = nidx ;
    215     nd.depth = depth ;
    216     nd.sibdex = sibdex ;
    217     nd.parent = parent ;
    218 
    219     nd.num_child = num_child ;
    220     nd.first_child = -1 ;     // gets changed inplace from lower recursion level 
    221     nd.next_sibling = -1 ;
    222     nd.lvid = lvid ;
    223     nd.copyno = copyno ;
    224 
    225     nd.sensor_id = -1 ;     // changed later by U4Tree::identifySensitiveInstances
    226     nd.sensor_index = -1 ;  // changed later by U4Tree::identifySensitiveInstances and stree::reorderSensors
    227 
    228 
    229     pvs.push_back(pv);
    230     st->nds.push_back(nd);
    231     st->digs.push_back(dig);
    232     st->m2w.push_back(tr_m2w);
    233     st->w2m.push_back(tr_w2m);
    234 
    235 
    236     glm::tmat4x4<double> tr_gtd(1.) ;          // "GGeo Transform Debug" comparison
    237     st->get_m2w_product(tr_gtd, nidx, false );  // NB this must be after push back of nd and tr_m2w
    238     st->gtd.push_back(tr_gtd);
    239 
    240 
    241 





::

    cd ~/opticks/sysrap/tests
    ./stree_test.sh ana

    In [18]: f.base
    Out[18]: '/tmp/blyth/opticks/ntds3/G4CXOpticks/stree'

    In [19]: g.base
    Out[19]: '/tmp/blyth/opticks/ntds3/G4CXOpticks/GGeo/stree'

    In [20]: f.gtd.shape
    Out[20]: (336653, 4, 4)

    In [21]: g.gtd.shape
    Out[21]: (336653, 4, 4)


No very large differences between the transforms::

    In [8]: np.abs( g.gtd - f.gtd ).max()    
    Out[8]: 0.0015625000014551915


    In [12]: w = np.where( np.abs( g.gtd - f.gtd ) > 0.001 )[0]

    In [13]: g.gtd[w]
    Out[13]: 
    array([[[     0.   ,      1.   ,      0.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [ 23451.301,  -6711.2  ,  23504.15 ,      1.   ]],

           [[     0.   ,      1.   ,      0.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [ 23451.301,  -6711.2  ,  23504.15 ,      1.   ]],

           [[     0.   ,      1.   ,      0.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [ 23319.301,  -6711.2  ,  23504.15 ,      1.   ]],

           [[     0.   ,      1.   ,      0.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [ 23319.301,  -6711.2  ,  23504.15 ,      1.   ]],

           [[     0.   ,      1.   ,      0.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [ 23187.301,  -6711.2  ,  23504.15 ,      1.   ]],

           ...,

           [[     0.   ,      1.   ,      0.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [-23187.301,   6711.2  ,  26504.15 ,      1.   ]],

           [[     0.   ,      1.   ,      0.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [-23319.301,   6711.2  ,  26504.15 ,      1.   ]],

           [[     0.   ,      1.   ,      0.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [-23319.301,   6711.2  ,  26504.15 ,      1.   ]],

           [[     0.   ,      1.   ,      0.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [-23451.301,   6711.2  ,  26504.15 ,      1.   ]],

           [[     0.   ,      1.   ,      0.   ,      0.   ],
            [    -1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [-23451.301,   6711.2  ,  26504.15 ,      1.   ]]])

    In [14]: f.gtd[w]
    Out[14]: 
    array([[[     0.  ,      1.  ,      0.  ,      0.  ],
            [    -1.  ,      0.  ,      0.  ,      0.  ],
            [     0.  ,      0.  ,      1.  ,      0.  ],
            [ 23451.3 ,  -6711.2 ,  23504.15,      1.  ]],

           [[     0.  ,      1.  ,      0.  ,      0.  ],
            [    -1.  ,      0.  ,      0.  ,      0.  ],
            [     0.  ,      0.  ,      1.  ,      0.  ],
            [ 23451.3 ,  -6711.2 ,  23504.15,      1.  ]],

           [[     0.  ,      1.  ,      0.  ,      0.  ],
            [    -1.  ,      0.  ,      0.  ,      0.  ],
            [     0.  ,      0.  ,      1.  ,      0.  ],
            [ 23319.3 ,  -6711.2 ,  23504.15,      1.  ]],

           [[     0.  ,      1.  ,      0.  ,      0.  ],
            [    -1.  ,      0.  ,      0.  ,      0.  ],
            [     0.  ,      0.  ,      1.  ,      0.  ],
            [ 23319.3 ,  -6711.2 ,  23504.15,      1.  ]],

           [[     0.  ,      1.  ,      0.  ,      0.  ],
            [    -1.  ,      0.  ,      0.  ,      0.  ],
            [     0.  ,      0.  ,      1.  ,      0.  ],
            [ 23187.3 ,  -6711.2 ,  23504.15,      1.  ]],

           ...,

           [[     0.  ,      1.  ,      0.  ,      0.  ],
            [    -1.  ,      0.  ,      0.  ,      0.  ],
            [     0.  ,      0.  ,      1.  ,      0.  ],
            [-23187.3 ,   6711.2 ,  26504.15,      1.  ]],

           [[     0.  ,      1.  ,      0.  ,      0.  ],
            [    -1.  ,      0.  ,      0.  ,      0.  ],
            [     0.  ,      0.  ,      1.  ,      0.  ],
            [-23319.3 ,   6711.2 ,  26504.15,      1.  ]],

           [[     0.  ,      1.  ,      0.  ,      0.  ],
            [    -1.  ,      0.  ,      0.  ,      0.  ],
            [     0.  ,      0.  ,      1.  ,      0.  ],
            [-23319.3 ,   6711.2 ,  26504.15,      1.  ]],

           [[     0.  ,      1.  ,      0.  ,      0.  ],
            [    -1.  ,      0.  ,      0.  ,      0.  ],
            [     0.  ,      0.  ,      1.  ,      0.  ],
            [-23451.3 ,   6711.2 ,  26504.15,      1.  ]],

           [[     0.  ,      1.  ,      0.  ,      0.  ],
            [    -1.  ,      0.  ,      0.  ,      0.  ],
            [     0.  ,      0.  ,      1.  ,      0.  ],
            [-23451.3 ,   6711.2 ,  26504.15,      1.  ]]])

        In [15]: g.gtd[w] - f.gtd[w]
        Out[15]: 
        array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.001,  0.   ,  0.   ,  0.   ]],

               [[ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.001,  0.   ,  0.   ,  0.   ]],

               [[ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.001,  0.   ,  0.   ,  0.   ]],

               [[ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.001,  0.   ,  0.   ,  0.   ]],

               [[ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.001,  0.   ,  0.   ,  0.   ]],

               ...,

               [[ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [-0.001,  0.   ,  0.   ,  0.   ]],

               [[ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [-0.001,  0.   ,  0.   ,  0.   ]],

               [[ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [-0.001,  0.   ,  0.   ,  0.   ]],

               [[ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [-0.001,  0.   ,  0.   ,  0.   ]],

               [[ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  0.   ],
                [-0.001,  0.   ,  0.   ,  0.   ]]])




Hmm the ggeo inst transforms should exactly match the gtd ones ? 


HUH, dont see the large differences anymore::

    In [9]: np.where( np.abs(cf.inst[:,:,:3] - f.inst_f4[:,:,:3]) > 0.001 )
    Out[9]: (array([], dtype=int64), array([], dtype=int64), array([], dtype=int64))

    In [10]: np.where( np.abs(cf.inst[:,:,:3] - f.inst_f4[:,:,:3]) > 0.0001 )
    Out[10]: 
    (array([47973, 47981, 47989, 47993, 47997, 48005, 48012, 48013, 48021, 48049, 48068, 48088, 48096, 48104, 48105, 48112, 48120, 48124, 48128, 48136, 48141, 48149, 48157, 48161, 48165, 48173, 48180,
            48181, 48189, 48217, 48236, 48256, 48264, 48272, 48273, 48280, 48288, 48292, 48296, 48304, 48309, 48317, 48325, 48329, 48333, 48341, 48348, 48349, 48357, 48385, 48404, 48424, 48432, 48440,
            48441, 48448, 48456, 48460, 48464, 48472]),
     array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
     array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]))

    In [11]: np.abs(cf.inst[:,:,:3] - f.inst_f4[:,:,:3]).max()
    Out[11]: 0.0009765625

    In [12]: cf.cfbase
    Out[12]: '/tmp/blyth/opticks/ntds3/G4CXOpticks'

    In [13]: f.base
    Out[13]: '/tmp/blyth/opticks/ntds3/G4CXOpticks/stree'




