review_sensor_id
====================

Context
----------

* from :doc:`QSimTest_shakedown_following_QPMT_extension`

Overview
-----------

* DONE : at U4Tree.h/stree.h snode level the sensor info looks OK
* DONE : persist GGeo, added "GEOM ggeo" 
* DONE : examine GGeo : find that MM 2,3 lack any sensor_id : SMOKING GUN 

  * SMOKING GUN : AS OLD/NEW MUST HAVE SAME IDEA OF SENSORS 
  * reason established : it is the PMT geometry change

* WHY IS GGEO STILL INVOLVED FOR SENSORS ? 

  * GGeo is still involved for everything, as still going via CSG_GGeo_Convert 

* WIP : HMM GGeo has its own stree ? 
* TODO : compare old and new inst and then switch to the new 

  * how to switch to U4Tree.h/stree.h inst ?  
  * CSGOptiX::Create just uploads the CSGFoundry fd 
  * CSGFoundry fd is still created with CSG_GGeo_Convert 


U4Tree/stree overview wrt inst and sensor info
------------------------------------------------

Everything done in U4Tree::Create from Geant4 traversals 
thru to adding inst with sensor info to stree with stree::add_inst 

* all looks fine and dandy 


U4Tree::Create
    Geant4 -> stree 

CSG_stree_Convert::Translate (INTENDED)
     stree -> CSGFoundry 



X4/GGeo overview wrt inst and sensor info
--------------------------------------------

Two step with GGeo model in the middle. 


X4Geo::Translate
   Geant4 -> GGeo

   All the heavy lifting done by X4PhysicalVolume instanciation 

   X4PhysicalVolume::convertNode 
        sensor decision captured with GVolume::setSensorIndex   


CSG_GGeo_Convert::Translate
   GGeo -> CSGFoundry  



Cycling on the conversion
----------------------------

::

   gxt ; ./G4CXOpticks_setGeometry_Test.sh


G4CXOpticks : how are the two workflows coordinated ? How to jump to new one ?
--------------------------------------------------------------------------------

::

    243 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    244 {
    245     LOG(LEVEL) << " G4VPhysicalVolume world " << world ;
    246     assert(world);
    247     wd = world ;
    248 
    249     assert(sim && "sim instance should have been created in ctor" );
    250     stree* st = sim->get_tree();
    251 
    252     tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    253 
    254     /**
    255     AIMING TO ELIMINATE GGeo, DEV IN CSG/tests/CSG_stree_Convert.h, ENABLING: 
    256     CSGFoundry* fd_ = CSG_stree_Convert::Translate( st );  
    257     setGeometry(fd_)
    258     **/
    259 
    260     // GGeo creation done when starting from a gdml or live G4,  still needs Opticks instance
    261     Opticks::Configure("--gparts_transform_offset --allownokey" );
    262     GGeo* gg_ = X4Geo::Translate(wd) ;
    263 
    264     setGeometry(gg_);
    265 }
    266 void G4CXOpticks::setGeometry(GGeo* gg_)
    267 {
    268     LOG(LEVEL);
    269     gg = gg_ ;
    270 
    271     CSGFoundry* fd_ = CSG_GGeo_Convert::Translate(gg) ;
    272     setGeometry(fd_);
    273 }





Local G4CXOpticks_setGeometry_Test.sh cycles to investigate
-------------------------------------------------------------



WIP : saved GGeo not going into expected dir
-----------------------------------------------


WIP : x4/GGeo has an stree ? Is that same instance as SSim ? NO
-----------------------------------------------------------------

The x4/stree looks to be a way to compare old/new by comparing x4/stree with u4/stree.  

Its foreign to GGeo, but tacked on in order to get saved presumably::

     159 void GGeo::setTree(stree* tree){ m_tree = tree ; }
     160 stree* GGeo::getTree() const {  return m_tree ; }

::

    epsilon:issues blyth$ opticks-f setTree 
    ./extg4/X4PhysicalVolume.cc:    m_ggeo->setTree(m_tree); 
    ./sysrap/stree.h:    GGeo:m_tree with setTree/getTree : but treated as foreign member, only GGeo::save saves it 
    ./sysrap/stree.h:    X4PhysicalVolume::convertStructure creates stree.h and setTree into GGeo 
    ./ggeo/GGeo.hh:        void setTree(stree* tree) ; 
    ./ggeo/GGeo.cc:void GGeo::setTree(stree* tree){ m_tree = tree ; }



But it is distinct from the SSim/stree::

    1401 void X4PhysicalVolume::convertStructure()
    1402 {
    1403     assert(m_top) ;
    1404     LOG(LEVEL) << "[ creating large tree of GVolume instances" ;
    1405 
    1406     m_tree = new stree ;
    1407     m_ggeo->setTree(m_tree);


Collects snode and transforms into the x4 stree::

    X4PhysicalVolume::convertStructure_r

    1533 
    1534      snode nd ;
    1535      nd.index = nidx ;
    1536      nd.depth = depth ;
    1537      nd.sibdex = sibdex ;
    1538      nd.parent = parent_nidx ;
    1539 
    1540      nd.num_child = num_child ;
    1541      nd.first_child = -1 ;     // gets changed inplace from lower recursion level 
    1542      nd.next_sibling = -1 ;
    1543      nd.lvid = lvid ;
    1544      nd.copyno = copyno ;
    1545 
    1546      nd.sensor_id = -1 ;
    1547      nd.sensor_index = -1 ;
    1548    
    1549      m_tree->nds.push_back(nd);
    1550      m_tree->m2w.push_back(tr_m2w);
    1551      m_tree->gtd.push_back(tr_gtd);
    1552      





WIP : create CSGFoundry from stree eliminating GGeo 
-------------------------------------------------------

Thinking of going direct from stree to CSGFoundry in::

   CSG/CSG_stree_Convert.h 
   CSG/tests/CSG_stree_Convert_test.sh 



Issue : Unexpected qat4.h sensor info. 
-------------------------------------------

::

    ct ; ./CSGFoundry_py_test.sh 


     16 if __name__ == '__main__':
     17     cf = CSGFoundry.Load()
     18     print(repr(cf))
     19 
     20     ins = cf.inst[:,0,3].view(np.int32)  # instance_idx      
     21     gas = cf.inst[:,1,3].view(np.int32)  # gas_idx      
     22     sid = cf.inst[:,2,3].view(np.int32)  # sensor_id       
     23     six = cf.inst[:,3,3].view(np.int32)  # sensor_idx       
     24 
     25     ugas,ngas = np.unique(gas, return_counts=True)
     26 
     27     EXPR = list(filter(None,textwrap.dedent(r"""
     28     cf.inst[:,:,3].view(np.int32)
     29     (sid.min(), sid.max())
     30     (six.min(), six.max())
     31     np.c_[ugas,ngas,cf.mmlabel] 


::

    ct ; ./CSGFoundry_py_test.sh


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
    [[     0      1]
     [     1      1]
     [     2      1]
     [     3      1]
     [     4      1]
     ...
     [307983      1]
     [307984      1]
     [307985      1]
     [307986      1]
     [307987      1]]
    np.c_[np.unique(sid[gas==2],return_counts=True)]     
    [[   -1 12615]]
    np.c_[np.unique(sid[gas==3],return_counts=True)]     
    [[  -1 4997]]
    np.c_[np.unique(sid[gas==4],return_counts=True)]     
    [[307988      1]
     [307989      1]
     [307990      1]
     [307991      1]
     [307992      1]
     ...
     [310383      1]
     [310384      1]
     [310385      1]
     [310386      1]
     [310387      1]]
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



Adding GSurfaceLib__SENSOR_SURFACE_LIST didnt move the needle
----------------------------------------------------------------

So add "GBndLib::descSensorBoundary" to see whats happening. 


X4PhysicalVolume::

    2035     ///////// sensor decision for the volume happens here  ////////////////////////
    2036     //////// TODO: encapsulate into a GBndLib::formSensorIndex ? 
    2037 
    2038     bool is_sensor = m_blib->isSensorBoundary(boundary) ; // this means that isurf/osurf has non-zero EFFICIENCY property 
    2039     unsigned sensorIndex = GVolume::SENSOR_UNSET ;
    2040     if(is_sensor)
    2041     {
    2042         sensorIndex = 1 + m_blib->getSensorCount() ;  // 1-based index
    2043         m_blib->countSensorBoundary(boundary);
    2044     }
    2045     volume->setSensorIndex(sensorIndex);   // must set to GVolume::SENSOR_UNSET for non-sensors, for sensor_indices array  
    2046 
    2047     ///////////////////////////////////////////////////////////////////////////


     663 bool GBndLib::isSensorBoundary(unsigned boundary) const
     664 {
     665     const guint4& bnd = m_bnd[boundary];
     666     bool osur_sensor = m_slib->isSensorIndex(bnd[OSUR]);
     667     bool isur_sensor = m_slib->isSensorIndex(bnd[ISUR]);
     668     bool is_sensor = osur_sensor || isur_sensor ;
     669     return is_sensor ;
     670 }




GGeo iid
-----------

::

    GEOM ggeo


    cd /tmp/blyth/opticks


    In [1]: np.load("GGeo/GMergedMesh/1/placement_iidentity.npy").shape
    Out[1]: (25600, 5, 4)

    In [16]: np.load("GGeo/GMergedMesh/1/placement_iidentity.npy")[0]
    Out[16]:
    array([[  244352, 16777216,  8323098,        0],
           [  244353, 16777217,  8192029,        0],
           [  244354, 16777218,  8060972,        1],
           [  244355, 16777219,  8126509,        0],
           [  244356, 16777220,  8257561,        0]], dtype=uint32)

    In [17]: np.load("GGeo/GMergedMesh/1/placement_iidentity.npy")[-1]
    Out[17]:
    array([[  372347, 23330560,  8323098,        0],
           [  372348, 23330561,  8192029,        0],
           [  372349, 23330562,  8060972,    25600],
           [  372350, 23330563,  8126509,        0],
           [  372351, 23330564,  8257561,        0]], dtype=uint32)

    In [18]: np.load("GGeo/GMergedMesh/1/placement_iidentity.npy")[100]
    Out[18]:
    array([[  244852, 16802816,  8323098,        0],
           [  244853, 16802817,  8192029,        0],
           [  244854, 16802818,  8060972,      101],
           [  244855, 16802819,  8126509,        0],
           [  244856, 16802820,  8257561,        0]], dtype=uint32)




    In [2]: np.load("GGeo/GMergedMesh/2/placement_iidentity.npy").shape
    Out[2]: (12615, 9, 4)

    In [22]: a = np.load("GGeo/GMergedMesh/2/placement_iidentity.npy")

    In [24]: np.unique( a[:,:,3], return_counts=True )
    Out[24]: (array([0], dtype=uint32), array([113535]))


    In [3]: np.load("GGeo/GMergedMesh/3/placement_iidentity.npy").shape
    Out[3]: (4997, 12, 4)

    In [25]: b = np.load("GGeo/GMergedMesh/3/placement_iidentity.npy")

    In [26]: b.shape
    Out[26]: (4997, 12, 4)

    In [29]: np.unique(b[:,:,3], return_counts=True)
    Out[29]: (array([0], dtype=uint32), array([59964]))


    ### THIS IS A SMOKING GUN : MM 2, 3 (the LPMT)  LACK ANY SENSOR_ID IN THE IID 


    In [4]: np.load("GGeo/GMergedMesh/4/placement_iidentity.npy").shape
    Out[4]: (2400, 6, 4)



    In [30]: c = np.load("GGeo/GMergedMesh/4/placement_iidentity.npy")

    In [31]: c.shape
    Out[31]: (2400, 6, 4)

    In [32]: c[0]
    Out[32]:
    array([[  372356, 67108864,  9109552,        0],
           [  372357, 67108865,  8781843,        0],
           [  372358, 67108866,  9043997,        0],
           [  372359, 67108867,  8978481,        0],
           [  372360, 67108868,  8847410,    25601],
           [  372361, 67108869,  8912947,        0]], dtype=uint32)

    In [33]: np.unique( c[:,:,3], return_counts=True )
    Out[33]:
    (array([    0, 25601, 25602, ..., 27998, 27999, 28000], dtype=uint32),
     array([12000,     1,     1, ...,     1,     1,     1]))








    In [5]: np.load("GGeo/GMergedMesh/5/placement_iidentity.npy").shape
    Out[5]: (590, 1, 4)

    In [6]: np.load("GGeo/GMergedMesh/6/placement_iidentity.npy").shape
    Out[6]: (590, 1, 4)

    In [7]: np.load("GGeo/GMergedMesh/7/placement_iidentity.npy").shape
    Out[7]: (590, 1, 4)

    In [8]: np.load("GGeo/GMergedMesh/8/placement_iidentity.npy").shape
    Out[8]: (590, 1, 4)

    In [9]: np.load("GGeo/GMergedMesh/9/placement_iidentity.npy").shape
    Out[9]: (504, 130, 4)

    In [10]: np.load("GGeo/GMergedMesh/0/placement_iidentity.npy").shape
    Out[10]: (1, 2977, 4)



Seems the GGeo::isSensor is no longer working as needed
----------------------------------------------------------

::

    epsilon:surface blyth$ pwd
    /Users/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/stree/surface

    epsilon:surface blyth$ find . -name EFFICIENCY.npy
    ./PMT_20inch_photocathode_logsurf1/EFFICIENCY.npy
    ./PMT_3inch_photocathode_logsurf2/EFFICIENCY.npy
    ./PMT_20inch_veto_photocathode_logsurf2/EFFICIENCY.npy
    ./PMT_20inch_photocathode_logsurf2/EFFICIENCY.npy
    ./PMT_20inch_veto_photocathode_logsurf1/EFFICIENCY.npy
    ./PMT_3inch_photocathode_logsurf1/EFFICIENCY.npy

    epsilon:surface blyth$ pwd
    /Users/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/stree/surface
    epsilon:surface blyth$
    epsilon:surface blyth$
    epsilon:surface blyth$ i

    In [1]: np.load("PMT_20inch_photocathode_logsurf1/EFFICIENCY.npy")[:,1].max()
    Out[1]: 0.8034280415921583

    In [2]: np.load("PMT_3inch_photocathode_logsurf2/EFFICIENCY.npy")[:,1].max()
    Out[2]: 1.0

    In [3]: np.load("PMT_20inch_photocathode_logsurf2/EFFICIENCY.npy")[:,1].max()
    Out[3]: 0.8034280415921583

    In [4]: np.load("PMT_20inch_veto_photocathode_logsurf2/EFFICIENCY.npy")[:,1].max()
    Out[4]: 1.0

    In [5]: np.load("PMT_20inch_veto_photocathode_logsurf1/EFFICIENCY.npy")[:,1].max()
    Out[5]: 1.0

    In [6]: np.load("PMT_3inch_photocathode_logsurf1/EFFICIENCY.npy")[:,1].max()
    Out[6]: 1.0

    In [7]:



* HMM the LPMT surfaces in the bnd are no longer the ones with the EFFICIENCY ?
* Thats an effect of the switch to simpler PMT and CustomART. 

::

    epsilon:standard blyth$ cat bnd_names.txt | grep Pyrex | grep Vacuum 
    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum

    Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
    Pyrex/PMT_3inch_absorb_logsurf2/PMT_3inch_absorb_logsurf1/Vacuum

    Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum
    Pyrex/PMT_20inch_veto_mirror_logsurf2/PMT_20inch_veto_mirror_logsurf1/Vacuum

    epsilon:standard blyth$ 


* so how can GGeo identify sensor surfaces ? As this is GGeo code which has not long to live 
  can just kludge it based on "photocathode" in the name perhaps.  

Added envvar sensitivity::

    export GSurfaceLib__SENSOR_SURFACE_LIST=HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf,NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf


Before using that::

    2023-07-14 02:40:30.326 INFO  [348012] [GSurfaceLib::collectSensorIndices@896]  ni 46
    2023-07-14 02:40:30.326 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 0 is_sensor_0 NO  is_listed NO  is_sensor NO  sn CDTyvekSurface
    2023-07-14 02:40:30.326 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 1 is_sensor_0 NO  is_listed NO  is_sensor NO  sn CDInnerTyvekSurface
    2023-07-14 02:40:30.326 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 2 is_sensor_0 NO  is_listed NO  is_sensor NO  sn VETOTyvekSurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 3 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_20inch_photocathode_logsurf1
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 4 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_20inch_photocathode_logsurf2
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 5 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_20inch_mirror_logsurf1
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 6 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_20inch_mirror_logsurf2
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 7 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_dynode_plate_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 8 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_inner_ring_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 9 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_outer_edge_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 10 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_inner_edge_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 11 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_dynode_tube_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 12 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_grid_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 13 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_shield_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 14 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 15 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 16 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 17 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMCPPMT_PMT_20inch_mcp_opsurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 18 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_20inch_veto_photocathode_logsurf1
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 19 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_20inch_veto_photocathode_logsurf2
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 20 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_20inch_veto_mirror_logsurf1
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 21 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_20inch_veto_mirror_logsurf2
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 22 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_3inch_photocathode_logsurf1
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 23 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_3inch_photocathode_logsurf2
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 24 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf1
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 25 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf2
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 26 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf3
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 27 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf4
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 28 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf5
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 29 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf6
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 30 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf7
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 31 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf8
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 32 is_sensor_0 NO  is_listed NO  is_sensor NO  sn UpperChimneyTyvekSurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 33 is_sensor_0 NO  is_listed NO  is_sensor NO  sn StrutAcrylicOpSurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 34 is_sensor_0 NO  is_listed NO  is_sensor NO  sn Strut2AcrylicOpSurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 35 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 36 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuMaskOpticalSurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 37 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 38 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMaskOpticalSurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 39 is_sensor_0 NO  is_listed NO  is_sensor NO  sn Steel_surface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 40 is_sensor_0 NO  is_listed NO  is_sensor NO  sn Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 41 is_sensor_0 NO  is_listed NO  is_sensor NO  sn Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 42 is_sensor_0 YES is_listed NO  is_sensor YES sn perfectDetectSurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 43 is_sensor_0 NO  is_listed NO  is_sensor NO  sn perfectAbsorbSurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 44 is_sensor_0 NO  is_listed NO  is_sensor NO  sn perfectSpecularSurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@915]  i 45 is_sensor_0 NO  is_listed NO  is_sensor NO  sn perfectDiffuseSurface
    2023-07-14 02:40:30.327 INFO  [348012] [GSurfaceLib::collectSensorIndices@924]  ni 46 sensor_surface_count 7


Added to jxv/ntds bash function::

    2023-07-14 02:50:54.291 INFO  [348306] [GSurfaceLib::collectSensorIndices@896]  ni 46
    2023-07-14 02:50:54.291 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 0 is_sensor_0 NO  is_listed NO  is_sensor NO  sn CDTyvekSurface
    2023-07-14 02:50:54.291 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 1 is_sensor_0 NO  is_listed NO  is_sensor NO  sn CDInnerTyvekSurface
    2023-07-14 02:50:54.291 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 2 is_sensor_0 NO  is_listed NO  is_sensor NO  sn VETOTyvekSurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 3 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_20inch_photocathode_logsurf1
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 4 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_20inch_photocathode_logsurf2
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 5 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_20inch_mirror_logsurf1
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 6 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_20inch_mirror_logsurf2
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 7 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_dynode_plate_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 8 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_inner_ring_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 9 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_outer_edge_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 10 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_inner_edge_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 11 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_dynode_tube_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 12 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_grid_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 13 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuR12860_PMT_20inch_shield_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 14 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 15 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 16 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 17 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMCPPMT_PMT_20inch_mcp_opsurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 18 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_20inch_veto_photocathode_logsurf1
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 19 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_20inch_veto_photocathode_logsurf2
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 20 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_20inch_veto_mirror_logsurf1
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 21 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_20inch_veto_mirror_logsurf2
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 22 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_3inch_photocathode_logsurf1
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 23 is_sensor_0 YES is_listed NO  is_sensor YES sn PMT_3inch_photocathode_logsurf2
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 24 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf1
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 25 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf2
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 26 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf3
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 27 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf4
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 28 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf5
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 29 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf6
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 30 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf7
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 31 is_sensor_0 NO  is_listed NO  is_sensor NO  sn PMT_3inch_absorb_logsurf8
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 32 is_sensor_0 NO  is_listed NO  is_sensor NO  sn UpperChimneyTyvekSurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 33 is_sensor_0 NO  is_listed NO  is_sensor NO  sn StrutAcrylicOpSurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 34 is_sensor_0 NO  is_listed NO  is_sensor NO  sn Strut2AcrylicOpSurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 35 is_sensor_0 NO  is_listed YES is_sensor YES sn HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 36 is_sensor_0 NO  is_listed NO  is_sensor NO  sn HamamatsuMaskOpticalSurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 37 is_sensor_0 NO  is_listed YES is_sensor YES sn NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 38 is_sensor_0 NO  is_listed NO  is_sensor NO  sn NNVTMaskOpticalSurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 39 is_sensor_0 NO  is_listed NO  is_sensor NO  sn Steel_surface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 40 is_sensor_0 NO  is_listed NO  is_sensor NO  sn Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 41 is_sensor_0 NO  is_listed NO  is_sensor NO  sn Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 42 is_sensor_0 YES is_listed NO  is_sensor YES sn perfectDetectSurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 43 is_sensor_0 NO  is_listed NO  is_sensor NO  sn perfectAbsorbSurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 44 is_sensor_0 NO  is_listed NO  is_sensor NO  sn perfectSpecularSurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@915]  i 45 is_sensor_0 NO  is_listed NO  is_sensor NO  sn perfectDiffuseSurface
    2023-07-14 02:50:54.292 INFO  [348306] [GSurfaceLib::collectSensorIndices@924]  ni 46 sensor_surface_count 9


That is with the envvar::

    export GSurfaceLib__SENSOR_SURFACE_LIST=HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf,NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf







GGeo::postDirectTranslation
----------------------------

::

    2023-07-13 22:05:19.023 INFO  [305489] [GGeo::postDirectTranslation@648] NOT SAVING : SAVE BY DEFINING ENVVAR: GGeo__postDirectTranslation_save


::

    pdig 4c29bcd2a52a397de5036b415af92efe ndig    504 nprog    129 placements    504 n pPanel_0_f_
    2023-07-13 22:10:11.890 INFO  [305655] [GGeo::postDirectTranslation@640] GGeo__postDirectTranslation_save
    2023-07-13 22:10:11.893 INFO  [305655] [GGeo::save@832]  idpath /tmp/blyth/opticks/GGeo
    2023-07-13 22:10:11.917 INFO  [305655] [BFile::preparePath@844] created directory /tmp/blyth/opticks/GGeo/GItemList
    2023-07-13 22:10:11.995 INFO  [305655] [BFile::preparePath@844] created directory /tmp/blyth/opticks/GGeo/GNodeLib
    2023-07-13 22:10:12.253 INFO  [305655] [BFile::preparePath@844] created directory /tmp/blyth/opticks/GGeo/GScintillatorLib/LS
    2023-07-13 22:10:12.255 INFO  [305655] [BFile::preparePath@844] created directory /tmp/blyth/opticks/GGeo/GScintillatorLib/LS_ori
     base /tmp/blyth/opticks/GGeo/stree/standard k wavelength.npy ERROR MISSING ARRAY FOR KEY
     base /tmp/blyth/opticks/GGeo/stree/standard k energy.npy ERROR MISSING ARRAY FOR KEY
     base /tmp/blyth/opticks/GGeo/stree/standard k rayleigh.npy ERROR MISSING ARRAY FOR KEY
     base /tmp/blyth/opticks/GGeo/stree/standard k mat.npy ERROR MISSING ARRAY FOR KEY
     base /tmp/blyth/opticks/GGeo/stree/standard k sur.npy ERROR MISSING ARRAY FOR KEY 
     base /tmp/blyth/opticks/GGeo/stree/standard k bd.npy ERROR MISSING ARRAY FOR KEY 
     base /tmp/blyth/opticks/GGeo/stree/standard k bnd.npy ERROR MISSING ARRAY FOR KEY 
     base /tmp/blyth/opticks/GGeo/stree/standard k optical.npy ERROR MISSING ARRAY FOR KEY 
     base /tmp/blyth/opticks/GGeo/stree/standard k icdf.npy ERROR MISSING ARRAY FOR KEY 
     base /tmp/blyth/opticks/GGeo/stree/subs_freq k key.npy ERROR MISSING ARRAY FOR KEY 
     base /tmp/blyth/opticks/GGeo/stree/subs_freq k val.npy ERROR MISSING ARRAY FOR KEY 
    2023-07-13 22:10:13.296 INFO  [305655] [G4CXOpticks::setGeometry@265] 
    2023-07-13 22:10:14.127 INFO  [305655] [G4CXOpticks::setGeometry_@324] [ fd 0x166d5f010







WIP : trace where qat4 inst identity info comes from
-----------------------------------------------------------

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











GGeo Level
------------

::

    1631 /**
    1632 GMergedMesh::getInstancedIdentityBuffer_SensorIndex
    1633 ----------------------------------------------------
    1634 
    1635 Extracts the sensor_index for each instance (as originally provided by GVolume::getIdentity 
    1636 GVolume::getSensorIndex) and appends to sensor_index in the order of the instances. 
    1637 
    1638 **/
    1639 
    1640 void GMergedMesh::getInstancedIdentityBuffer_SensorIndex(std::vector<int>& sensor_index, bool one_based_index ) const
    1641 {
    1642     NPY<unsigned>* iid = getInstancedIdentityBuffer();
    1643     Get3DFouthColumnNonZero(sensor_index, iid, one_based_index );
    1644 }




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



DONE : stree_py_test.sh : check sensor_name : MUST restrict to sensor nodes to avoid loadsa unset zeros
----------------------------------------------------------------------------------------------------------

::

    GEOM get 
    st ; ./stree_py_test.sh 

    In [12]: np.c_[np.unique(f.nds[:,14], return_counts=True)]
    Out[12]:
    array([[    -1,   2864],
           [     0, 363880],  ## THAT SHOULD BE 25600 : UNSET ZERO DEFAULT REMAINDER ?
           [     1,  12615],
           [     2,   4997],
           [     3,   2400]])

    In [21]: np.c_[np.unique(st.nds.sensor_name, return_counts=True )]
    Out[21]: 
    array([[    -1,   2864],
           [     0, 363880],
           [     1,  12615],
           [     2,   4997],
           [     3,   2400]])




Omitted to set snode::sensor_name for the remainder in  U4Tree::identifySensitiveGlobals


::

    In [14]: f.sensor_name_names
    Out[14]: array(['PMT_3inch_log_phys', 'pLPMT_NNVT_MCPPMT', 'pLPMT_Hamamatsu_R12860', 'mask_PMT_20inch_vetolMaskVirtual_phys'], dtype='<U37')

    In [15]: np.c_[f.sensor_name_names]
    Out[15]:
    array([['PMT_3inch_log_phys'],
           ['pLPMT_NNVT_MCPPMT'],
           ['pLPMT_Hamamatsu_R12860'],
           ['mask_PMT_20inch_vetolMaskVirtual_phys']], dtype='<U37')


But fixing that is not sufficient::

    In [1]: np.c_[np.unique(st.nds.sensor_name, return_counts=True )]
    Out[1]: 
    array([[    -1,   5841],
           [     0, 360903],
           [     1,  12615],
           [     2,   4997],
           [     3,   2400]])


The reason is that do not visit all the nodes so still loads unset zeros.
When restrict to sensor nodes get expected counts::

    In [4]: sna = st.nds.sensor_name[st.nds.sensor_index>-1]

    In [5]: np.unique(sna, return_counts=True)
    Out[5]: (array([0, 1, 2, 3], dtype=int32), array([25600, 12615,  4997,  2400]))


    In [6]: sna0 = st.nds.sensor_name[st.nds.sensor_index==-1]

    In [7]: sna0
    Out[7]: array([-1, -1, -1, -1, -1, ...,  0,  0,  0,  0,  0], dtype=int32)

    In [8]: np.unique(sna0, return_counts=True)
    Out[8]: (array([-1,  0], dtype=int32), array([  5841, 335303]))


    In [2]: st.sf
    Out[2]:
    sf   0 :   25600 : f2e4da325cbfc7582ff695f42b684930.
    sf   1 :   12615 : 2bf11f67d9cbcf2125907956fa5835fe.
    sf   2 :    4997 : 7c81a83fe61312ac0eb392cc3dc376fe.
    sf   3 :    2400 : edff08bf49c5dff191aa0e3c89e0f435.
    sf   4 :     590 : c051c1bb98b71ccb15b0cf9c67d143ee.
    sf   5 :     590 : 5e01938acb3e0df0543697fc023bffb1.
    sf   6 :     590 : cdc824bf721df654130ed7447fb878ac.





WIP : Check U4Tree.h/stree.h snode::sensor_id/name/index
-----------------------------------------------------------


::

    797 inline void U4Tree::identifySensitiveInstances()
    798 {
    799     unsigned num_factor = st->get_num_factor();
    800     if(level > 0) std::cerr
    801         << "[ U4Tree::identifySensitiveInstances"
    802         << " num_factor " << num_factor
    803         << " st.sensor_count " << st->sensor_count
    804         << std::endl
    805         ;
    806 
    807     for(unsigned i=0 ; i < num_factor ; i++)
    808     {
    809         std::vector<int> outer ;
    810         st->get_factor_nodes(outer, i );  // nidx of outer volumes of instances 
    811         sfactor& fac = st->get_factor_(i);
    812         fac.sensors = 0  ;
    813 
    814         for(unsigned j=0 ; j < outer.size() ; j++)
    815         {
    816             int nidx = outer[j] ;
    817             const G4VPhysicalVolume* pv = get_pv_(nidx) ;
    818             const char* pvn = pv->GetName().c_str() ;
    819 
    820             int sensor_id = sid->getInstanceIdentity(pv) ;
    821             int sensor_index = sensor_id > -1 ? st->sensor_count : -1 ;
    822             int sensor_name = -1 ;
    823 
    824             if(sensor_id > -1 )
    825             {
    826                 st->sensor_count += 1 ;  // count over all factors  
    827                 fac.sensors += 1 ;   // count sensors for each factor  
    828                 sensor_name = suniquename::Add(pvn, st->sensor_name ) ;
    829             }
    830             snode& nd = st->nds[nidx] ;
    831             nd.sensor_id = sensor_id ;
    832             nd.sensor_index = sensor_index ;
    833             nd.sensor_name = sensor_name ;
    834      
    835 



U4Tree.h collected sensor_id and sensor_index
-----------------------------------------------

The sensor_index just increments, so what it is for each sensor_id 
depends on the collection order. 

Starts as expected with 3 inch:: 
  
        sensor_id 300000 -> 325599     
        sensor_index 0->25599  

Then factor 1 NNVT copyno (HAMA gaps)::

        sensor_id      (between 0:17612 but with HAMA gaps)
        sensor_index   25600 ... 

Then factor 2 HAMA copyno (NNVT gaps)::

    .   sensor_id      (again between 0:17612 but with NNVT gaps)
        sensor_index   25600+12615 = 38215 ...  


    In [1]: 25600+12615+4997 
    Out[1]: 43212


Then factor 3 WPMT::

        sensor_id     30000 -> 30000+2400-1 
        sensor_index  43212 -> 43212+2400-1 = 45611


::

    In [5]: np.unique(st.nds.sensor_id, return_counts=True)
    Out[5]: 
    (array([    -1,      0,      1,      2,      3, ..., 325595, 325596, 325597, 325598, 325599], dtype=int32),
     array([341144,      1,      1,      1,      1, ...,      1,      1,      1,      1,      1]))

    In [9]: st.nds.sensor_index
    Out[9]: array([-1, -1, -1, -1, -1, ..., -1, -1, -1, -1, -1], dtype=int32)

    In [10]: st.nds.sensor_index[st.nds.sensor_index>-1]
    Out[10]: array([    0,     1,     2,     3,     4, ..., 45607, 45608, 45609, 45610, 45611], dtype=int32)

    In [11]: np.all( np.arange(45612) == st.nds.sensor_index[st.nds.sensor_index>-1] )
    Out[11]: True

    In [13]: np.count_nonzero(np.logical_and( st.nds.sensor_id > -1, st.nds.sensor_index > -1 ))
    Out[13]: 45612





::

    U4Tree::identifySensitiveInstances i       1 sensor_id   17588 sensor_index   38212
    U4SensorIdentifierDefault::getIdentity copyno 17589 num_sd 2 is_sensor 1 pvn pLPMT_NNVT_MCPPMT has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       1 sensor_id   17589 sensor_index   38213
    U4SensorIdentifierDefault::getIdentity copyno 17590 num_sd 2 is_sensor 1 pvn pLPMT_NNVT_MCPPMT has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       1 sensor_id   17590 sensor_index   38214
    U4Tree::identifySensitiveInstances factor 1 fac.sensors 12615
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 2 is_sensor 1 pvn pLPMT_Hamamatsu_R12860 has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       2 sensor_id       0 sensor_index   38215
    U4SensorIdentifierDefault::getIdentity copyno 1 num_sd 2 is_sensor 1 pvn pLPMT_Hamamatsu_R12860 has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       2 sensor_id       1 sensor_index   38216
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 2 is_sensor 1 pvn pLPMT_Hamamatsu_R12860 has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       2 sensor_id       3 sensor_index   38217

::

    U4Tree::identifySensitiveInstances i       2 sensor_id   17609 sensor_index   43209
    U4SensorIdentifierDefault::getIdentity copyno 17610 num_sd 2 is_sensor 1 pvn pLPMT_Hamamatsu_R12860 has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       2 sensor_id   17610 sensor_index   43210
    U4SensorIdentifierDefault::getIdentity copyno 17611 num_sd 2 is_sensor 1 pvn pLPMT_Hamamatsu_R12860 has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       2 sensor_id   17611 sensor_index   43211
    U4Tree::identifySensitiveInstances factor 2 fac.sensors 4997
    U4SensorIdentifierDefault::getIdentity copyno 30000 num_sd 2 is_sensor 1 pvn mask_PMT_20inch_vetolMaskVirtual_phys has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       3 sensor_id   30000 sensor_index   43212
    U4SensorIdentifierDefault::getIdentity copyno 30001 num_sd 2 is_sensor 1 pvn mask_PMT_20inch_vetolMaskVirtual_phys has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       3 sensor_id   30001 sensor_index   43213
    U4SensorIdentifierDefault::getIdentity copyno 30002 num_sd 2 is_sensor 1 pvn mask_PMT_20inch_vetolMaskVirtual_phys has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       3 sensor_id   30002 sensor_index   43214
                



::

    U4SensorIdentifierDefault::getIdentity copyno 325597 num_sd 2 is_sensor 1 pvn PMT_3inch_log_phys has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       0 sensor_id  325597 sensor_index   25597
    U4SensorIdentifierDefault::getIdentity copyno 325598 num_sd 2 is_sensor 1 pvn PMT_3inch_log_phys has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       0 sensor_id  325598 sensor_index   25598
    U4SensorIdentifierDefault::getIdentity copyno 325599 num_sd 2 is_sensor 1 pvn PMT_3inch_log_phys has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       0 sensor_id  325599 sensor_index   25599
    U4Tree::identifySensitiveInstances factor 0 fac.sensors 25600
    U4SensorIdentifierDefault::getIdentity copyno 2 num_sd 2 is_sensor 1 pvn pLPMT_NNVT_MCPPMT has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       1 sensor_id       2 sensor_index   25600
    U4SensorIdentifierDefault::getIdentity copyno 4 num_sd 2 is_sensor 1 pvn pLPMT_NNVT_MCPPMT has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       1 sensor_id       4 sensor_index   25601
    U4SensorIdentifierDefault::getIdentity copyno 6 num_sd 2 is_sensor 1 pvn pLPMT_NNVT_MCPPMT has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       1 sensor_id       6 sensor_index   25602
    U4SensorIdentifierDefault::getIdentity copyno 21 num_sd 2 is_sensor 1 pvn pLPMT_NNVT_MCPPMT has_PMT_pvn YES


::

    U4SensorIdentifierDefault::getIdentity copyno 32398 num_sd 2 is_sensor 1 pvn mask_PMT_20inch_vetolMaskVirtual_phys has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       3 sensor_id   32398 sensor_index   45610
    U4SensorIdentifierDefault::getIdentity copyno 32399 num_sd 2 is_sensor 1 pvn mask_PMT_20inch_vetolMaskVirtual_phys has_PMT_pvn YES
    U4Tree::identifySensitiveInstances i       3 sensor_id   32399 sensor_index   45611
    U4Tree::identifySensitiveInstances factor 3 fac.sensors 2400





::

    334 /**
    335 GVolume::setSensorIndex
    336 -------------------------
    337 
    338 sensorIndex is expected to be a 1-based contiguous index, with the 
    339 default value of SENSOR_UNSET (0)  meaning no sensor.
    340 
    341 This is canonically invoked from X4PhysicalVolume::convertNode during GVolume creation.
    342 
    343 * GNode::setSensorIndices duplicates the index to all faces of m_mesh triangulated geometry
    344 
    345 **/
    346 void GVolume::setSensorIndex(unsigned sensorIndex)
    347 {
    348     m_sensorIndex = sensorIndex ;
    349     setSensorIndices( m_sensorIndex );
    350 }


::

    epsilon:tests blyth$ opticks-f setSensorIndex
    ./extg4/X4PhysicalVolume.cc:    volume->setSensorIndex(sensorIndex);   // must set to GVolume::SENSOR_UNSET for non-sensors, for sensor_indices array  
    ./extg4/X4PhysicalVolume.cc:    volume->setSensorIndex(sensorIndex);   // must set to GVolume::SENSOR_UNSET for non-sensors, for sensor_indices array  
    ./ggeo/GGeoTest.cc:        volume->setSensorIndex(sensorIndex); // see notes/issues/GGeoTest_GMergedMesh_mergeVolumeFaces_assert_sensor_indices.rst 
    ./ggeo/GVolume.cc:GVolume::setSensorIndex
    ./ggeo/GVolume.cc:void GVolume::setSensorIndex(unsigned sensorIndex)
    ./ggeo/GVolume.hh:      void     setSensorIndex(unsigned sensorIndex) ;
    epsilon:opticks blyth$ 


From  X4PhysicalVolume::convertNode::

    2035     ///////// sensor decision for the volume happens here  ////////////////////////
    2036     //////// TODO: encapsulate into a GBndLib::formSensorIndex ? 
    2037 
    2038     bool is_sensor = m_blib->isSensorBoundary(boundary) ; // this means that isurf/osurf has non-zero EFFICIENCY property 
    2039     unsigned sensorIndex = GVolume::SENSOR_UNSET ;
    2040     if(is_sensor)
    2041     {
    2042         sensorIndex = 1 + m_blib->getSensorCount() ;  // 1-based index
    2043         m_blib->countSensorBoundary(boundary);
    2044     }
    2045     volume->setSensorIndex(sensorIndex);   // must set to GVolume::SENSOR_UNSET for non-sensors, for sensor_indices array  
    2046 
    2047     ///////////////////////////////////////////////////////////////////////////
    2048 


::

     654 /**
     655 GBndLib::isSensorBoundary
     656 --------------------------
     657 
     658 Canonically invoked from X4PhysicalVolume::convertNode 
     659 
     660 
     661 **/
     662 
     663 bool GBndLib::isSensorBoundary(unsigned boundary) const
     664 {
     665     const guint4& bnd = m_bnd[boundary];
     666     bool osur_sensor = m_slib->isSensorIndex(bnd[OSUR]);
     667     bool isur_sensor = m_slib->isSensorIndex(bnd[ISUR]);
     668     bool is_sensor = osur_sensor || isur_sensor ;
     669     return is_sensor ;
     670 }
     671 
     672 void GBndLib::countSensorBoundary(unsigned boundary)
     673 {
     674     m_boundary_sensor_count[boundary] += 1 ;
     675     m_sensor_count += 1 ;
     676 }
     678 /**
     679 GBndLib::getSensorCount (precache)
     680 ------------------------------------
     681 
     682 **/
     683 unsigned GBndLib::getSensorCount() const
     684 {
     685     return m_sensor_count ;
     686 }


::

     889 void GSurfaceLib::collectSensorIndices()
     890 {
     891     unsigned ni = getNumSurfaces();
     892     for(unsigned i=0 ; i < ni ; i++)
     893     {
     894         GPropertyMap<double>* surf = m_surfaces[i] ;
     895         bool is_sensor = surf->isSensor() ;
     896         if(is_sensor)
     897         {
     898             addSensorIndex(i);
     899             assert( isSensorIndex(i) == true ) ;
     900         }
     901     }
     902 }


     288 template <class T>
     289 bool GPropertyMap<T>::isSensor()
     290 {
     291     return hasNonZeroProperty(EFFICIENCY) || hasNonZeroProperty(detect) ;
     292 }



