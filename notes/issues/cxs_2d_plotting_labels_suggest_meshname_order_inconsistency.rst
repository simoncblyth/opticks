cxs_2d_plotting_labels_suggest_meshname_order_inconsistency
===============================================================


::

    gc        
    GEOM=body_phys ./run.sh 


    PMTSim::Traverse_r depth  0 pv           _body_phys lv            _body_log so      _body_solid_1_9 mt                Pyrex
    PMTSim::Traverse_r depth  1 pv         _inner1_phys lv          _inner1_log so      _inner1_solid_I mt               Vacuum
    PMTSim::Traverse_r depth  1 pv         _inner2_phys lv          _inner2_log so    _inner2_solid_1_9 mt               Vacuum


    [ fd.descMeshName 
    CSGFoundry::descMeshName meshname.size 3
        0 : _inner1_solid_I
        1 : _inner2_solid_1_9
        2 : _body_solid_1_9
    ] fd.descMeshName 


    cx
    ./cxs.sh


    (laptop)
    gc
    GEOM=body_phys ./grab.sh 

    cx
    GEOM=body_phys ./cxs.sh 




Labelling is clearly wrong::

    .                                               FROM POSITION OF THE INTERSECTS WHAT LABEL SHOULD BE 
    red   :          inner1_solid_I                 body_solid_1_9
    blue  :          inner2_solid_1_9               inner1_solid_I          
    green :          body_solid_1_9                 inner2_solid_1_9


CSGFoundry::meshname come from::

     56 void CSG_GGeo_Convert::init()
     57 {
     58     ggeo->getMeshNames(foundry->meshname);


::

    0927 void GGeo::getMeshNames(std::vector<std::string>& meshNames) const
     928 {
     929      m_meshlib->getMeshNames(meshNames);
     930 }

    741 void GMeshLib::getMeshNames(std::vector<std::string>& meshNames) const
    742 {
    743     meshNames.clear();
    744     unsigned numMeshes = getNumMeshes();
    745     for(unsigned midx=0 ; midx < numMeshes ; midx++)
    746     {
    747         const char* mname = getMeshName(midx);
    748         meshNames.push_back(mname);
    749     }
    750 }

    229 const char* GMeshLib::getMeshName(unsigned aindex) const
    230 {
    231     return m_meshnames->getKey(aindex);
    232 }

    164 const char* GItemList::getKey(unsigned index) const
    165 {
    166     return index < m_list.size() ? m_list[index].c_str() : NULL  ;
    167 }



    358 /**
    359 GMeshLib::add
    360 ----------------
    361 
    362 Invoked via GGeo::add from X4PhysicalVolume::convertSolids_r as each distinct 
    363 solid is encountered in the recursive traverse.
    364 
    365 **/
    366 
    367 void GMeshLib::add(const GMesh* mesh, bool alt )
    368 {
    369     const char* name = mesh->getName();
    370     unsigned index = mesh->getIndex();
    371     assert(name) ;
    372 
    373     m_meshnames->add(name);
    374 


    0872 void X4PhysicalVolume::convertSolids()
     873 {
     874     OK_PROFILE("_X4PhysicalVolume::convertSolids");
     875     LOG(LEVEL) << "[" ;
     876 
     877     const G4VPhysicalVolume* pv = m_top ;
     878     int depth = 0 ;
     879     convertSolids_r(pv, depth);
     880 

    0909 void X4PhysicalVolume::convertSolids_r(const G4VPhysicalVolume* const pv, int depth)
     910 {
     911     const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
     912 
     913     // G4LogicalVolume::GetNoDaughters returns 1042:G4int, 1062:size_t
     914     for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ;i++ )
     915     {
     916         const G4VPhysicalVolume* const daughter_pv = lv->GetDaughter(i);
     917         convertSolids_r( daughter_pv , depth + 1 );
     918     }
     919 
     920     // for newly encountered lv record the tail/postorder idx for the lv
     921     if(std::find(m_lvlist.begin(), m_lvlist.end(), lv) == m_lvlist.end())
     922     {
     923         convertSolid( lv );
     924     }
     925 }


::

        .
            _body_solid_1_9
           /                \
        _inner1_solid_I    _inner2_solid_1_9        


Meshnames are collected by X4PhysicalVolume::convertSolid in postorder::

    [ fd.descMeshName 
    CSGFoundry::descMeshName meshname.size 3
        0 : _inner1_solid_I
        1 : _inner2_solid_1_9
        2 : _body_solid_1_9
    ] fd.descMeshName 



Probably are making the bad assumption that primIdx matches meshIdx::

    228 CSGSolid* CSG_GGeo_Convert::convertSolid( unsigned repeatIdx )
    229 {
    230     unsigned nmm = ggeo->getNumMergedMesh();
    231     assert( repeatIdx < nmm );
    232     const GMergedMesh* mm = ggeo->getMergedMesh(repeatIdx);
    233     unsigned num_inst = mm->getNumITransforms() ;
    234 
    235     const GParts* comp = ggeo->getCompositeParts(repeatIdx) ;
    236     assert( comp );
    237     unsigned numPrim = comp->getNumPrim();
    238     std::string rlabel = CSGSolid::MakeLabel('r',repeatIdx) ;
    239 
    240     bool dump = dump_ridx > -1 && dump_ridx == int(repeatIdx) ;
    241 
    242     LOG(LEVEL)
    243         << " repeatIdx " << repeatIdx
    244         << " nmm " << nmm
    245         << " numPrim(GParts.getNumPrim) " << numPrim
    246         << " rlabel " << rlabel
    247         << " num_inst " << num_inst
    248         << " dump_ridx " << dump_ridx
    249         << " dump " << dump
    250         ;  
    251 
    252     CSGSolid* so = foundry->addSolid(numPrim, rlabel.c_str() );  // primOffset captured into CSGSolid 
    253     assert(so);
    254 
    255     AABB bb = {} ;
    256 
    257     // over the "layers/volumes" of the solid (eg multiple vols of PMT) 
    258     for(unsigned primIdx=0 ; primIdx < numPrim ; primIdx++)
    259     {  
    260         unsigned globalPrimIdx = so->primOffset + primIdx ;
    261         unsigned globalPrimIdx_0 = foundry->getNumPrim() ;
    262         assert( globalPrimIdx == globalPrimIdx_0 );
    263 
    264         CSGPrim* prim = convertPrim(comp, primIdx);
    265         bb.include_aabb( prim->AABB() );
    266 
    267         unsigned sbtIdx = prim->sbtIndexOffset() ;
    268         //assert( sbtIdx == globalPrimIdx  );  
    269         assert( sbtIdx == primIdx  );
    270 
    271         prim->setRepeatIdx(repeatIdx);
    272         prim->setPrimIdx(primIdx);
    273         //LOG(LEVEL) << prim->desc() ;
    274     }  
    275     so->center_extent = bb.center_extent() ;
    276 
    277     addInstances(repeatIdx);




::

    2351 unsigned GParts::getVolumeIndex(unsigned i) const
    2352 {
    2353     return getUIntIdx(i, VOL_IDX ) ;
    2354 }
    2355 unsigned GParts::getMeshIndex(unsigned i) const
    2356 {
    2357     return getUIntIdx(i, MESH_IDX ) ;
    2358 }
    2359 






::

    2021-11-14 15:16:39.426 INFO  [558662] [*CSG_GGeo_Convert::convertPrim@335]  primIdx    0 meshIdx    2 comp.getTypeMask 2 CSG::TypeMask un  CSG::IsPositiveMask 1
    2021-11-14 15:16:39.426 INFO  [558662] [*CSG_GGeo_Convert::convertPrim@335]  primIdx    1 meshIdx    0 comp.getTypeMask 0 CSG::TypeMask  CSG::IsPositiveMask 1
    2021-11-14 15:16:39.426 INFO  [558662] [*CSG_GGeo_Convert::convertPrim@335]  primIdx    2 meshIdx    1 comp.getTypeMask 2 CSG::TypeMask un  CSG::IsPositiveMask 1



primIdx appears to be inorder::

      .     0                 

        1      2


but meshIdx is postorder ::


      .    2

        0     1 



::

    epsilon:CSG blyth$ vi CSGFoundry.py 
    epsilon:CSG blyth$ ipython -i CSGFoundry.py 
    INFO:__main__:load /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry 
             node :           (31, 4, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/node.npy 
             itra :           (16, 4, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/itra.npy 
         meshname :                 (3,)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/meshname.txt 
             meta :                 (3,)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/meta.txt 
          bndname :                 (2,)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/bnd.txt 
             tran :           (16, 4, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/tran.npy 
             inst :            (1, 4, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/inst.npy 
              bnd :    (2, 4, 2, 761, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/bnd.npy 
            solid :            (1, 3, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/solid.npy 
             prim :            (3, 4, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/prim.npy 
    INFO:root:dump_node_boundary
        0 :     15 : Pyrex///Pyrex 
        1 :     16 : Pyrex///Vacuum 

    In [1]: cf.prim
    Out[1]: 
    array([[[   0.,    0.,    0.,    0.],
            [   0.,    0.,    0.,    0.],
            [-254., -254., -450.,  254.],
            [ 254.,  190.,    0.,    0.]],

           [[   0.,    0.,    0.,    0.],
            [   0.,    0.,    0.,    0.],
            [-249., -249.,    0.,  249.],
            [ 249.,  185.,    0.,    0.]],

           [[   0.,    0.,    0.,    0.],
            [   0.,    0.,    0.,    0.],
            [-249., -249., -445.,  249.],
            [ 249.,    0.,    0.,    0.]]], dtype=float32)

    In [2]: cf.prim.view(np.int32)
    Out[2]: 
    array([[[         15,           0,           0,           0],
            [          0,           2,           0,           0],
            [-1015152640, -1015152640, -1008664576,  1132331008],
            [ 1132331008,  1128136704,           0,           0]],

           [[          1,          15,           8,           0],
            [          1,           0,           0,           1],
            [-1015480320, -1015480320,           0,  1132003328],
            [ 1132003328,  1127809024,           0,           0]],

           [[         15,          16,           9,           0],
            [          2,           1,           0,           2],
            [-1015480320, -1015480320, -1008828416,  1132003328],
            [ 1132003328,           0,           0,           0]]], dtype=int32)

    In [3]: 


So need to use cf.prim[primIdx]::


    In [11]: cf.prim[0].view(np.uint32)[1,1]
    Out[11]: 2

    In [12]: cf.prim[1].view(np.uint32)[1,1]
    Out[12]: 0

    In [13]: cf.prim[2].view(np.uint32)[1,1]
    Out[13]: 1



::


     19 /**
     20 CSGPrim : contiguous sequence of *numNode* CSGNode starting from *nodeOffset* : complete binary tree of 1,3,7,15,... CSGNode
     21 ===============================================================================================================================
     22       
     23 * although CSGPrim is uploaded to GPU by CSGFoundry::upload, instances of CSGPrim at first glance 
     24   appear not to be needed GPU side because the Binding.h HitGroupData carries the same information.  
     25       
     26 * But that is disceptive as the uploaded CSGPrim AABB are essential for GAS construction 
     27       
     28 * vim replace : shift-R
     29       
     30       
     31     +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
     32     | q  |      x         |      y         |     z          |      w         |  notes                                          |
     33     +====+================+================+================+================+=================================================+
     34     |    |  numNode       |  nodeOffset    | tranOffset     | planOffset     |                                                 |
     35     | q0 |                |                | TODO:remove    | TODO: remove   |                                                 |
     36     |    |                |                |                |                |                                                 |
     37     +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
     38     |    | sbtIndexOffset |  meshIdx       | repeatIdx      | primIdx        |                                                 |
     39     |    |                |                |                |                |                                                 |
     40     | q1 |                |                |                |                |                                                 |
     41     |    |                |                |                |                |                                                 |
     42     |    |                |                |                |                |                                                 |
     43     +----+----------------+----------------+----------------+----------------+-------------------------------------------------+




::

    epsilon:CSG blyth$ ipython -i CSGFoundry.py 
    INFO:__main__:load /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry 
             node :           (31, 4, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/node.npy 
             itra :           (16, 4, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/itra.npy 
         meshname :                 (3,)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/meshname.txt 
             meta :                 (3,)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/meta.txt 
          bndname :                 (2,)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/bnd.txt 
             tran :           (16, 4, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/tran.npy 
             inst :            (1, 4, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/inst.npy 
              bnd :    (2, 4, 2, 761, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/bnd.npy 
            solid :            (1, 3, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/solid.npy 
             prim :            (3, 4, 4)  : /tmp/blyth/opticks/GeoChain_Darwin/body_phys/CSGFoundry/prim.npy 
    INFO:root:dump_node_boundary
        0 :     15 : Pyrex///Pyrex 
        1 :     16 : Pyrex///Vacuum 
    primIdx     0 midx     2 meshname _body_solid_1_9 
    primIdx     1 midx     0 meshname _inner1_solid_I 
    primIdx     2 midx     1 meshname _inner2_solid_1_9 
    {0: '_body_solid_1_9', 1: '_inner1_solid_I', 2: '_inner2_solid_1_9'}

    In [1]: 



Inconsistency seems resolved using CSG/CSGFoundry.py::

     27     def primIdx_meshname_dict(self):
     28         """
     29         See notes/issues/cxs_2d_plotting_labels_suggest_meshname_order_inconsistency.rst
     30         """
     31         d = {}
     32         for primIdx in range(len(self.prim)):
     33             midx = self.meshIdx (primIdx)
     34             assert midx < len(self.meshname)
     35             mnam = self.meshname[midx]
     36             d[primIdx] = mnam
     37             print("primIdx %5d midx %5d meshname %s " % (primIdx, midx, mnam))
     38         pass
     39         return d


