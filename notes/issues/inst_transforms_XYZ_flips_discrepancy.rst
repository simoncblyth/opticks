inst_transforms_XYZ_flips_discrepancy
=======================================

Previous :doc:`sensor_info_into_new_workflow` showed getting inst identity info to match.
BUT: now getting mismatch between the transforms. 

* HMM: I thought the transforms were matching previously, so recent changes may have broken them 
* AHAH : could this be --gparts_transform_offset yet again ? 


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


