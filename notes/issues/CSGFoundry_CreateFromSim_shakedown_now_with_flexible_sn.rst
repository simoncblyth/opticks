CSGFoundry_CreateFromSim_shakedown_now_with_flexible_sn
==========================================================

Context
----------

* prev :doc:`CSGFoundry_CreateFromSim_shakedown`

A/B python comparison
------------------------

Python analysis, A/B comparison::

    ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh ana

Overview
----------

Two geometry routes::

     A0             X4      CSG_GGeo
     OLD : Geant4 -----> GGeo/NNode/NCSG/... ----->  CSGFoundry 
                         
     ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh   ## A0 : Loads GDML, pass world to G4CXOpticks  
     ## HMM : THIS ALREADY POPULATES AN stree : SO COULD CREATE THE TWO CSGFoundry GEOMETRIES WITH THIS ONE EXECUTABLE 


     B0+B1        U4Tree        CSGImport
     NEW : Geant4 ----->  stree ------>  CSGFoundry 
                          snd/sn

     ~/opticks/u4/tests/U4TreeCreateSSimTest.sh             ## B0 : Loads GDML, Create SSim/stree using U4Tree.h 
     ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh    ## B1 : Loads SSim/stree, uses CSGImport to create and save CSGFoundry


     B0           U4Tree        
     NEW : Geant4 ----->  stree 
                          snd/sn

     ~/opticks/u4/tests/U4TreeCreateSSimTest.sh  ## B0 : Loads GDML, Create SSim/stree using U4Tree.h 



     B1                        CSGImport
     NEW :                stree ------>  CSGFoundry 
                          snd/sn

     ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh    ## B1 : Loads SSim/stree, uses CSGImport to create and save CSGFoundry


Breakpoint debug examples
----------------------------

::

    BP=sn::Serialize    ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh
    BP=X4Solid::Balance ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh   ## not hit with GEOM V1J010

    BP=NTreeProcess::Process ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh
    


Workflow to create A and B CSGFoundry (and SSim/stree) 
-------------------------------------------------------

Create the A and B geometries::

     ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh   ## A0 : Loads GDML, pass world to G4CXOpticks, does everything  
     ~/opticks/u4/tests/U4TreeCreateSSimTest.sh             ## B0 : Loads GDML, Create SSim/stree using U4Tree.h 
     ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh    ## B1 : Loads SSim/stree, runs CSGImport creating CSGFoundry

Python analysis, A/B comparison::

    ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh ana

CAUTION regards what to rebuild + rerun after changes, in order to make a valid new A/B comparison

+----------------+-----------+
| U4Tree.h       | A0,B0,B1  |
+----------------+-----------+
| CSGImport.h    | B1        |
+----------------+-----------+
| G4CXOpticks.hh | A0        |
+----------------+-----------+


FIXED : BP=sn::Serialize for A shows all pool empty : probably this is an issue of the other stree
----------------------------------------------------------------------------------------------------

DONE : Replaced X4 stree with stree_standin to avoid confusion between "spy" stree in X4 and 
the "real" one in gx.  

::

    epsilon:ggeo blyth$ opticks-f setTree 
    ./extg4/X4PhysicalVolume.cc:    m_ggeo->setTree(m_tree); 
    ./sysrap/stree.h:    GGeo:m_tree with setTree/getTree : but treated as foreign member, only GGeo::save saves it 
    ./sysrap/stree.h:    X4PhysicalVolume::convertStructure creates stree.h and setTree into GGeo 
    ./ggeo/GGeo.hh:        void setTree(stree* tree) ; 
    ./ggeo/GGeo.cc:void GGeo::setTree(stree* tree){ m_tree = tree ; }
    ./ggeo/GGeoTest.cc:    m_csglist->setTree( index, const_cast<NCSG*>(replacement_solid) ); 
    ./npy/NCSGList.cpp:void NCSGList::setTree(unsigned index, NCSG* replacement) 
    ./npy/NCSGList.hpp:        void         setTree(unsigned index, NCSG* replacement );
    epsilon:opticks blyth$ 


All pool empty at sn::Serialize so all refs -1::

    (lldb) f 0 
    frame #0: 0x000000010b7db82e libSysRap.dylib`sn::Serialize(n=0x000000099ab4d000, x=0x0000000113e19cd0) at sn.h:592
       589 	    n.xform = s_tv::pool->index(x->xform) ;  
       590 	    n.param = s_pa::pool->index(x->param) ;  
       591 	    n.aabb = s_bb::pool->index(x->aabb) ;  
    -> 592 	    n.parent = pool->index(x->parent);  
       593 	
       594 	#ifdef WITH_CHILD
       595 	    n.sibdex = x->sibling_index();  // 0 for root 
    (lldb) p s_bb::pool->pool
    (s_pool<s_bb, _s_bb>::POOL) $4 = size=0 {}
    (lldb) p s_pa::pool->pool
    (s_pool<s_pa, _s_pa>::POOL) $5 = size=0 {}
    (lldb) p s_tv::pool->pool
    (s_pool<s_tv, _s_tv>::POOL) $6 = size=0 {}
    (lldb) p pool->pool
    (s_pool<sn, _sn>::POOL) $7 = size=0 {}
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
      * frame #0: 0x000000010b7db82e libSysRap.dylib`sn::Serialize(n=0x000000099ab4d000, x=0x0000000113e19cd0) at sn.h:592
        frame #1: 0x000000010b7db358 libSysRap.dylib`s_pool<sn, _sn>::serialize_(this=0x000000010c400c80, buf=size=553) const at s_pool.h:317
        frame #2: 0x000000010b7da932 libSysRap.dylib`NP* s_pool<sn, _sn>::serialize<int>(this=0x000000010c400c80) const at s_pool.h:342
        frame #3: 0x000000010b7d819f libSysRap.dylib`s_csg::serialize(this=0x000000010c4000a0) const at s_csg.h:172
        frame #4: 0x000000010b7a87cf libSysRap.dylib`stree::serialize(this=0x000000010c4012d0) const at stree.h:2035
        frame #5: 0x000000010b7a7ada libSysRap.dylib`SSim::serialize(this=0x000000010c33a490) at SSim.cc:365
        frame #6: 0x0000000100122600 libG4CX.dylib`G4CXOpticks::init_SEvt(this=0x000000010c33a5c0) at G4CXOpticks.cc:384
        frame #7: 0x0000000100121ab5 libG4CX.dylib`G4CXOpticks::setGeometry_(this=0x000000010c33a5c0, fd_=0x000000099d82aef0) at G4CXOpticks.cc:339
        frame #8: 0x0000000100120fcf libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010c33a5c0, fd_=0x000000099d82aef0) at G4CXOpticks.cc:310
        frame #9: 0x000000010012161e libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010c33a5c0, gg_=0x0000000131c5ea70) at G4CXOpticks.cc:277
        frame #10: 0x000000010011f622 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010c33a5c0, world=0x000000010c34adc0) at G4CXOpticks.cc:269
        frame #11: 0x0000000100120f49 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010c33a5c0, gdmlpath="/Users/blyth/.opticks/GEOM/V1J010/origin.gdml") at G4CXOpticks.cc:222
        frame #12: 0x000000010011efde libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010c33a5c0) at G4CXOpticks.cc:202
        frame #13: 0x000000010011e3a0 libG4CX.dylib`G4CXOpticks::SetGeometry() at G4CXOpticks.cc:64
        frame #14: 0x000000010000fa6f G4CXOpticks_setGeometry_Test`main(argc=1, argv=0x00007ffeefbfe6f0) at G4CXOpticks_setGeometry_Test.cc:16
        frame #15: 0x00007fff56c7c015 libdyld.dylib`start + 1
    (lldb) 


::

    (lldb) f 6
    frame #6: 0x0000000100122600 libG4CX.dylib`G4CXOpticks::init_SEvt(this=0x000000010c33a5c0) at G4CXOpticks.cc:384
       381 	
       382 	void G4CXOpticks::init_SEvt()
       383 	{
    -> 384 	    sim->serialize() ;  
       385 	    SEvt* sev = SEvt::CreateOrReuse(SEvt::EGPU) ; 
       386 	    sev->setGeo((SGeo*)fd);   
       387 	    smeta::Collect(sev->meta, "G4CXOpticks::init_SEvt"); 
    (lldb) 





FIXED : A.SSim.stree._csg.sn all references are -1 : THIS WAS THE EXTRANEOUS stree EMPTY POOL ISSUE
-----------------------------------------------------------------------------------------------------

::

    In [10]: at = A.SSim.stree
    In [11]: bt = B.SSim.stree

    In [18]: an = A.SSim.stree._csg.sn
    In [19]: bn = B.SSim.stree._csg.sn          

    In [23]: np.unique( np.where( an != bn )[1] )
    Out[23]: array([ 3,  4,  5,  6,  9, 10, 11])

    In [13]: np.all( an[:,3:7] == -1 )
    Out[13]: True


::

    In [1]: sn.Doc()
    Out[1]: 
     0 : tc :        typecode : <i4 
     1 : cm :      complement : <i4 
     2 : lv :            lvid : <i4 
     3 : xf :           xform : <i4 
     4 : pa :           param : <i4 
     5 : bb :            aabb : <i4 
     6 : pr :          parent : <i4 
     7 : sx :          sibdex : <i4 
     8 : nc :       num_child : <i4 
     9 : fc :     first_child : <i4 
    10 : ns :    next_sibling : <i4 
    11 : ix :           index : <i4 
    12 : dp :           depth : <i4 
    13 : l0 :          label0 : <i4 
    14 : l1 :          label1 : <i4 
    15 : l2 :          label2 : <i4 
    16 : l3 :          label3 : <i4 


    In [6]: w = np.where( an != bn ) 

    In [7]: an[w]
    Out[7]: array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ..., -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=int32)

    In [8]: np.all( an[w] == -1 )
    Out[8]: True


DONE : After A:stree_standin FIX getting stree._csg perfect match as expected
-------------------------------------------------------------------------------

::

    In [1]: an = A.SSim.stree._csg.sn
    In [2]: bn = B.SSim.stree._csg.sn

    In [4]: an.shape
    Out[4]: (553, 17)

    In [5]: bn.shape
    Out[5]: (553, 17)

    In [6]: np.where( an != bn )
    Out[6]: (array([], dtype=int64), array([], dtype=int64))

    In [7]: ac = A.SSim.stree._csg

    In [8]: bc = B.SSim.stree._csg

    In [9]: ac
    Out[9]:
    _csg

    CMDLINE:/Users/blyth/opticks/CSG/tests/CSGFoundryAB.py
    _csg.base:/tmp/blyth/opticks/G4CXOpticks_setGeometry_Test/CSGFoundry/SSim/stree/_csg

      : _csg.s_bb                                          :             (346, 6) : 0:05:26.161790
      : _csg.sn                                            :            (553, 17) : 0:05:26.162039
      : _csg.s_pa                                          :             (346, 6) : 0:05:26.161520
      : _csg.NPFold_index                                  :                 (4,) : 0:05:26.162296
      : _csg.s_tv                                          :       (205, 2, 4, 4) : 0:05:26.160863

     min_stamp : 2023-08-19 11:59:09.712616
     max_stamp : 2023-08-19 11:59:09.714049
     dif_stamp : 0:00:00.001433
     age_stamp : 0:05:26.160863

    In [10]: bc
    Out[10]:
    _csg

    CMDLINE:/Users/blyth/opticks/CSG/tests/CSGFoundryAB.py
    _csg.base:/tmp/blyth/opticks/CSGFoundry_CreateFromSimTest/CSGFoundry/SSim/stree/_csg

      : _csg.s_bb                                          :             (346, 6) : 16:12:59.395056
      : _csg.sn                                            :            (553, 17) : 16:12:59.395231
      : _csg.s_pa                                          :             (346, 6) : 16:12:59.394893
      : _csg.NPFold_index                                  :                 (4,) : 16:12:59.395422
      : _csg.s_tv                                          :       (205, 2, 4, 4) : 16:12:59.394722

     min_stamp : 2023-08-18 19:51:39.647552
     max_stamp : 2023-08-18 19:51:39.648252
     dif_stamp : 0:00:00.000700
     age_stamp : 16:12:59.394722

    In [11]: np.all( ac.s_bb == bc.s_bb )
    Out[11]: True

    In [12]: np.all( ac.s_pa == bc.s_pa )
    Out[12]: True

    In [13]: np.all( ac.s_tv == bc.s_tv )
    Out[13]: True

    In [14]: np.all( ac.sn == bc.sn )
    Out[14]: True


DONE : A/B stree now looking identical, as they should : just the same code running in different places
---------------------------------------------------------------------------------------------------------

::

    In [15]: at = A.SSim.stree

    In [16]: bt = B.SSim.stree

    In [20]: at.nds.shape
    Out[20]: (386112, 15)

    In [21]: bt.nds.shape
    Out[21]: (386112, 15)

    In [22]: np.all( at.nds == bt.nds )
    Out[22]: True

    In [28]: np.all( at.soname_names == bt.soname_names )
    Out[28]: True

    In [32]: np.all( at.w2m == bt.w2m )
    Out[32]: True


WIP : A/B CSGFoundry primname difference
--------------------------------------------

A/B primname 0x suffix difference ? Where does B trim the suffix ?::

    In [48]: a.primname[:3]
    Out[48]: array(['sWorld0x59f2060', 'sTopRock0x59f4600', 'sDomeRockBox0x59f4770'], dtype=object)

    In [49]: b.primname[:3]
    Out[49]: array(['sWorld', 'sTopRock', 'sDomeRockBox'], dtype=object)

    In [50]: a_primname = np.char.partition(a.primname.astype("U"),"0x")[:,0]

    In [51]: b_primname = b.primname.astype("U")

After trimming the A suffix with np.char.partition can see that the 
prim count difference is from three "sMask_virtual" that are skipped in A::

    In [42]: a_primname.shape
    Out[42]: (3085,)

    In [43]: b_primname.shape
    Out[43]: (3088,)

    In [44]: set(a_primname)-set(b_primname)
    Out[44]: set()

    In [45]: set(b_primname)-set(a_primname)
    Out[45]:
    {'HamamatsuR12860sMask_virtual',
     'NNVTMCPPMTsMask_virtual',
     'mask_PMT_20inch_vetosMask_virtual'}

B::

     56 void CSGImport::importNames()
     57 {
     58     assert(st);
     59     st->get_mmlabel( fd->mmlabel);
     60     st->get_meshname(fd->meshname);
     61 }
     62 



     421 void CSGFoundry::addMeshName(const char* name)
     422 {
     423     meshname.push_back(name);
     424 }



     274 void CSGFoundry::getPrimName( std::vector<std::string>& pname ) const
     275 {
     276     unsigned num_prim = prim.size();
     277     for(unsigned i=0 ; i < num_prim ; i++)
     278     {
     279         const CSGPrim& pr = prim[i] ;
     280         unsigned midx = num_prim == 1 ? 0 : pr.meshIdx();  // kludge avoid out-of-range for single prim CSGFoundry
     281         
     282         if(midx == UNDEFINED)
     283         {
     284             pname.push_back("err-midx-undefined");   // avoid FAIL  with CSGMakerTest 
     285         }   
     286         else
     287         {
     288             const char* mname = getMeshName(midx);
     289             LOG(debug) << " primIdx " << std::setw(4) << i << " midx " << midx << " mname " << mname  ;
     290             pname.push_back(mname);   
     291         }   
     292     }   
     293 }   
     294 
     295 const char* CSGFoundry::getMeshName(unsigned midx) const
     296 {
     297     bool in_range = midx < meshname.size() ;
     298     
     299     LOG_IF(fatal, !in_range) << " not in range midx " << midx << " meshname.size()  " << meshname.size()  ;
     300     assert(in_range);
     301 
     302     return meshname[midx].c_str() ;
     303 }




A/B meta : A has some cxskiplv I had forgotten about ?
---------------------------------------------------------

::

    In [6]: A.meta
    Out[6]: 
    array(['source:CSGFoundry::init', 'creator:G4CXOpticks_setGeometry_Test', 'stamp:1692387173174430', 'stampFmt:2023-08-18T20:32:53.174430',
           'uname:Darwin epsilon.local 17.7.0 Darwin Kernel Version 17.7.0: Thu Jun 21 22:53:14 PDT 2018; root:xnu-4570.71.2~1/RELEASE_X86_64 x86_64', 'HOME:/Users/blyth', 'USER:blyth',
           'PWD:/Users/blyth/opticks/g4cx/tests', 'GEOM:V1J010', '${GEOM}_GEOMList:V1J010_GEOMList',
           'cxskiplv:NNVTMCPPMTsMask_virtual0x,HamamatsuR12860sMask_virtual0x,mask_PMT_20inch_vetosMask_virtual0x', 'cxskiplv_idxlist:117,108,134'], dtype='<U136')

    In [7]: B.meta
    Out[7]: 
    array(['source:CSGFoundry::init', 'creator:CSGFoundry_CreateFromSimTest', 'stamp:1692384698522289', 'stampFmt:2023-08-18T19:51:38.522289',
           'uname:Darwin epsilon.local 17.7.0 Darwin Kernel Version 17.7.0: Thu Jun 21 22:53:14 PDT 2018; root:xnu-4570.71.2~1/RELEASE_X86_64 x86_64', 'HOME:/Users/blyth', 'USER:blyth',
           'PWD:/Users/blyth/opticks/CSG/tests', 'GEOM:V1J010', '${GEOM}_GEOMList:V1J010_GEOMList'], dtype='<U136')

    In [8]:                      




DONE : Disabled the SGeoConfig::GeometrySpecificSetup skipping in A
-----------------------------------------------------------------------

In first instance its better to compare simulations with the same geometry
without any skips : SO DISABLED THE  SGeoConfig::GeometrySpecificSetup


::

    257 void SGeoConfig::GeometrySpecificSetup(const SName* id)  // static
    258 {
    259     const char* JUNO_names = "HamamatsuR12860sMask,HamamatsuR12860_PMT_20inch,NNVTMCPPMT_PMT_20inch" ;
    260     bool JUNO_detected = id->hasNames(JUNO_names); 
    261     LOG(LEVEL) << " JUNO_detected " << JUNO_detected ;
    262     if(JUNO_detected)
    263     {
    264         const char* skips = nullptr ;"NNVTMCPPMTsMask_virtual,HamamatsuR12860sMask_virtual,mask_PMT_20inch_vetosMask_virtual" ;
    265         SetCXSkipLV(skips); 
    266         SetCXSkipLV_IDXList(id);
    267         
    268         // USING dynamic ELVSelection here would be inappropriate : as dynamic selection 
    269         // means the persisted geometry does not match the used geometry.   
    270     }   
    271     
    272 }   
    273 

    0091 void CSG_GGeo_Convert::init()
      92 {   
      93     int consistent = CheckConsistent(ggeo, tree) ; 
      94     LOG_IF(fatal, consistent != 0 ) << DescConsistent(ggeo, tree);
      95     LOG(info) << DescConsistent(ggeo, tree);
      96     assert( consistent == 0 );
      97     
      98     ggeo->getMeshNames(foundry->meshname); 
      99     ggeo->getMergedMeshLabels(foundry->mmlabel); 
     100     // boundary names now travel with the NP bnd.names 
     101     
     102     SGeoConfig::GeometrySpecificSetup(foundry->id);
     103     
     104     const char* cxskiplv = SGeoConfig::CXSkipLV() ; 
     105     const char* cxskiplv_idxlist = SGeoConfig::CXSkipLV_IDXList() ;  
     106     foundry->setMeta<std::string>("cxskiplv", cxskiplv ? cxskiplv : "-" ); 
     107     foundry->setMeta<std::string>("cxskiplv_idxlist", cxskiplv_idxlist ? cxskiplv_idxlist : "-" );
     108     LOG(LEVEL) 
     109         << " cxskiplv  " << cxskiplv 
     110         << " cxskiplv   " << cxskiplv
     111         << " foundry.meshname.size " << foundry->meshname.size()
     112         << " foundry.id.getNumName " << foundry->id->getNumName()
     113         ;
     114 }



WIP : A/B CSGPrim prim content : ints match, B lacks bbox
------------------------------------------------------------


A/B first 8 ints match::


    In [38]: np.all( a.prim[:,:2].view(np.int32).reshape(-1,8) == b.prim[:,:2].view(np.int32).reshape(-1,8) )                                    
    Out[38]: True


B:prim misses bbox::

    In [39]: a.prim[0]                                                                                                                           
    Out[39]: 
    array([[     0.,      0.,      0.,      0.],
           [     0.,      0.,      0.,      0.],
           [-60000., -60000., -60000.,  60000.],
           [ 60000.,  60000.,      0.,      0.]], dtype=float32)

    In [40]: b.prim[0]                                                                                                                           
    Out[40]: 
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], dtype=float32)



::

    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    | q  |      x         |      y         |     z          |      w         |  notes                                          |
    +====+================+================+================+================+=================================================+
    |    |  numNode       |  nodeOffset    | tranOffset     | planOffset     |                                                 |
    | q0 |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    | sbtIndexOffset |  meshIdx       | repeatIdx      | primIdx        |                                                 |
    |    |                |  (lvid)        |                |                |                                                 |
    | q1 |                |  (1,1)         |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    | q2 |  BBMin_x       |  BBMin_y       |  BBMin_z       |  BBMax_x       |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    | q3 |  BBMax_y       |  BBMax_z       |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+




    


A/B CSGNode node content
-------------------------

::

    In [23]: a.node.shape                                                                                                                        
    Out[23]: (15968, 4, 4)

    In [24]: b.node.shape                                                                                                                        
    Out[24]: (15968, 4, 4)


    In [18]: np.where(ab.node > 1e-2)
    Out[18]: (array([    15679, 15680, 
                         15720, 15721, 
                         15750, 15753, 
                         15765, 15768, 
                         15827, 15829, 15830, 15834]),)

    In [19]: w = np.where(ab.node > 1e-2)[0]



    In [22]: np.c_[a.node[w], b.node[w], a.node[w] - b.node[w]]
    Out[22]:
    array([[[   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,    0.   ],
            [-183.225,    1.   ,    0.   ,    0.   , -183.225,    0.   ,    0.   ,    0.   ,    0.   ,    1.   ,   -0.   ,   -0.   ],
            [-264.05 , -264.05 , -183.225,  264.05 , -264.05 , -264.05 , -183.225,  264.05 ,    0.   ,    0.   ,    0.   ,    0.   ],
            [ 264.05 ,    1.   ,    0.   ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,   98.   ,    0.   ,    0.   ,    0.   ,   97.   ,    0.   ,    0.   ,    0.   ,    1.   ,   -0.   ,   -0.   ],
            [-264.05 , -264.05 ,    0.   ,  264.05 , -264.05 , -264.05 ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,    0.   ],
            [ 264.05 ,   98.   ,    0.   ,    0.   ,  264.05 ,   97.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,    0.   ],
            [-183.225,    1.   ,    0.   ,    0.   , -183.225,    0.   ,    0.   ,    0.   ,    0.   ,    1.   ,   -0.   ,   -0.   ],
            [-264.05 , -264.05 , -183.225,  264.05 , -264.05 , -264.05 , -183.225,  264.05 ,    0.   ,    0.   ,    0.   ,    0.   ],
            [ 264.05 ,    1.   ,    0.   ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,  101.   ,    0.   ,    0.   ,    0.   ,  100.   ,    0.   ,    0.   ,    0.   ,    1.   ,   -0.   ,   -0.   ],
            [-264.05 , -264.05 ,    0.   ,  264.05 , -264.05 , -264.05 ,    0.   ,  264.05 ,    0.   ,    0.   ,    0.   ,    0.   ],
            [ 264.05 ,  101.   ,    0.   ,    0.   ,  264.05 ,  100.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,  190.001,    0.   ,    0.   ,    0.   ,  190.001,    0.   ,    0.   ,    0.   ,    0.   ],
            [-168.226,    1.   ,    0.   ,    0.   , -168.226,    0.   ,    0.   ,    0.   ,    0.   ,    1.   ,   -0.   ,   -0.   ],
            [-254.001, -254.001, -173.226,  254.001, -254.001, -254.001, -173.226,  254.001,    0.   ,    0.   ,    0.   ,    0.   ],
            [ 254.001,   -4.   ,    0.   ,    0.   ,  254.001,   -5.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,  190.001,    0.   ,    0.   ,    0.   ,  190.001,    0.   ,    0.   ,    0.   ,    0.   ],
            [  -1.   ,  190.101,    0.   ,    0.   ,    0.   ,  190.101,    0.   ,    0.   ,   -1.   ,    0.   ,   -0.   ,   -0.   ],
            [-254.001, -254.001,   -1.   ,  254.001, -254.001, -254.001,    0.   ,  254.001,    0.   ,    0.   ,   -1.   ,    0.   ],
            [ 254.001,  190.101,    0.   ,    0.   ,  254.001,  190.101,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,  185.   ,    0.   ,    0.   ,    0.   ,  185.   ,    0.   ,    0.   ,    0.   ,    0.   ],
            [-163.225,    1.   ,    0.   ,    0.   , -163.225,    0.   ,    0.   ,    0.   ,    0.   ,    1.   ,   -0.   ,   -0.   ],
            [-249.   , -249.   , -168.225,  249.   , -249.   , -249.   , -168.225,  249.   ,    0.   ,    0.   ,    0.   ,    0.   ],
            [ 249.   ,   -4.   ,    0.   ,    0.   ,  249.   ,   -5.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,  185.   ,    0.   ,    0.   ,    0.   ,  185.   ,    0.   ,    0.   ,    0.   ,    0.   ],
            [  -1.   ,  185.1  ,    0.   ,    0.   ,    0.   ,  185.1  ,    0.   ,    0.   ,   -1.   ,    0.   ,   -0.   ,   -0.   ],
            [-249.   , -249.   ,   -1.   ,  249.   , -249.   , -249.   ,    0.   ,  249.   ,    0.   ,    0.   ,   -1.   ,    0.   ],
            [ 249.   ,  185.1  ,    0.   ,    0.   ,  249.   ,  185.1  ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,   70.   ,    0.   ,    0.   ,    0.   ,   70.   ,    0.   ,    0.   ,    0.   ,    0.   ],
            [-101.   ,  -14.   ,    0.   ,    0.   , -101.   ,  -15.   ,    0.   ,    0.   ,    0.   ,    1.   ,   -0.   ,   -0.   ],
            [ -70.   ,  -70.   , -101.   ,   70.   ,  -70.   ,  -70.   , -101.   ,   70.   ,    0.   ,    0.   ,    0.   ,    0.   ],
            [  70.   ,  -14.   ,    0.   ,    0.   ,   70.   ,  -15.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,   55.5  ,    0.   ,    0.   ,    0.   ,   55.5  ,    0.   ,    0.   ,    0.   ,    0.   ],
            [-102.   ,  -15.   ,    0.   ,    0.   , -101.   ,  -15.   ,    0.   ,    0.   ,   -1.   ,    0.   ,   -0.   ,   -0.   ],
            [ -55.5  ,  -55.5  , -102.   ,   55.5  ,  -55.5  ,  -55.5  , -101.   ,   55.5  ,    0.   ,    0.   ,   -1.   ,    0.   ],
            [  55.5  ,  -15.   ,    0.   ,   -0.   ,   55.5  ,  -15.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,   -0.   ]],

           [[   0.   ,    0.   ,    0.   ,   43.   ,    0.   ,    0.   ,    0.   ,   43.   ,    0.   ,    0.   ,    0.   ,    0.   ],
            [ -16.   ,    1.   ,    0.   ,    0.   ,  -15.   ,    0.   ,    0.   ,    0.   ,   -1.   ,    1.   ,   -0.   ,   -0.   ],
            [ -43.   ,  -43.   ,  -16.   ,   43.   ,  -43.   ,  -43.   ,  -15.   ,   43.   ,    0.   ,    0.   ,   -1.   ,    0.   ],
            [  43.   ,    1.   ,    0.   ,   -0.   ,   43.   ,    0.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,   -0.   ]],

           [[ 200.   , -140.   ,  451.786,    1.   ,  200.   , -140.   ,  450.   ,    0.   ,    0.   ,    0.   ,    1.786,    1.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ,   -0.   ,   -0.   ],
            [-451.786, -451.786, -140.   ,  451.786, -450.   , -450.   , -140.   ,  450.   ,   -1.786,   -1.786,    0.   ,    1.786],
            [ 451.786,    1.   ,    0.   ,    0.   ,  450.   ,    0.   ,    0.   ,    0.   ,    1.786,    1.   ,    0.   ,    0.   ]]], dtype=float32)


Find primname of prims with deviant nodes::

    In [58]: %cpaste                                                                                                                             
    Pasting code; enter '--' alone on the line to stop or use Ctrl-D.
    :    w = np.where(ab.node > 1e-2)[0] 
    :    nn = a.prim[:,0,0].view(np.int32)  
    :    no = a.prim[:,0,1].view(np.int32) 
    :
    :    for v in w:
    :        wv = np.where(np.logical_and( v >= no, v <= no+nn ))[0][0]      
    :        print(v, wv, a.primname[wv])
    :    pass
    :
    :
    :--
    15679 2928 NNVTMCPPMTsMask_virtual0x6173a40
    15680 2928 NNVTMCPPMTsMask_virtual0x6173a40
    15720 2937 HamamatsuR12860sMask_virtual0x6163d90
    15721 2937 HamamatsuR12860sMask_virtual0x6163d90
    15750 2940 HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280
    15753 2940 HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280
    15765 2941 HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0
    15768 2941 HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0
    15827 2956 base_steel0x5aa4870
    15829 2956 base_steel0x5aa4870
    15830 2956 base_steel0x5aa4870
    15834 2957 uni_acrylic10x5ba6710

    In [59]:                                    



All familiar names of solids with coincidence issues.
Encapsulate finding primIdx from nodeIdx into CSGFoundry.py::


    In [4]: w = np.where(ab.node > 1e-2)[0] ; w
    Out[4]: array([15679, 15680, 15720, 15721, 15750, 15753, 15765, 15768, 15827, 15829, 15830, 15834])


    In [8]: np.c_[b.primname[np.unique(a.find_primIdx_from_nodeIdx(w))]]
    Out[8]:
    array([['NNVTMCPPMTsMask_virtual'],
           ['HamamatsuR12860sMask_virtual'],            ## 
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_4'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_4'],   ## jcv Hamamatsu_R12860_PMTSolid : union of PMT bulb and polycone neck
           ['base_steel'],                     ## jcv UpperAcrylicConstruction    : polycone
           ['uni_acrylic1']], dtype=object)    ## jcv AdditionAcrylicConstruction : huge sphere with polycone subtracted







* HMM: THEY LOOK LIKE THEY ALL INCLUDE POLYCONE CONVERSIONS 

::

    In [10]: lvs = a.prim[pp,1,1].view(np.int32)
    In [12]: np.c_[lvs,a.meshname[lvs]]
    Out[12]:
    array([[117, 'NNVTMCPPMTsMask_virtual0x6173a40'],
           [117, 'NNVTMCPPMTsMask_virtual0x6173a40'],
           [108, 'HamamatsuR12860sMask_virtual0x6163d90'],
           [108, 'HamamatsuR12860sMask_virtual0x6163d90'],
           [107, 'HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           [107, 'HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           [106, 'HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           [106, 'HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           [95, 'base_steel0x5aa4870'],
           [95, 'base_steel0x5aa4870'],
           [95, 'base_steel0x5aa4870'],
           [96, 'uni_acrylic10x5ba6710']], dtype=object)


* 117,108,107,106,95,96

* DONE : improved nudge logging,

::

    2023-08-19 17:34:55.839 ERROR [32807060] [X4Solid::convertPolycone@1725] all_z_descending detected, reversing base_steel0x5aa4870
    2023-08-19 17:34:55.839 FATAL [32807060] [*X4Solid::Polycone_MakeInner@1843]  EXPERIMENTAL num_R_inner > 1 handling :  name base_steel0x5aa4870 num_R_inner 2 lvIdx 95
    2023-08-19 17:34:55.839 ERROR [32807060] [*X4Solid::Polycone_MakeInner@1854]  inner_prims.size 2 lvIdx 95
    2023-08-19 17:34:55.839 ERROR [32807060] [*X4Solid::Polycone_MakeInner@1869]  lower.is_znudge_capable lvIdx 95
    2023-08-19 17:34:55.839 ERROR [32807060] [*X4Solid::Polycone_MakeInner@1872]  upper.is_znudge_capable lvIdx 95
    2023-08-19 17:34:55.839 ERROR [32807060] [*X4Solid::Polycone_MakeInner@1925]  after znudges lvIdx 95
    2023-08-19 17:34:55.843 ERROR [32807060] [X4Solid::convertPolycone@1725] all_z_descending detected, reversing solidAddition_down0x5ba5d90
    2023-08-19 17:34:55.843 INFO  [32807060] [ncone::increase_z2@119]  new_z2 1 new_r2 451.786
    2023-08-19 17:34:56.048 ERROR [32807060] [X4Solid::convertPolycone@1725] all_z_descending detected, reversing PMT_3inch_cntr_solid0x68fff60
    2023-08-19 17:34:56.049 ERROR [32807060] [X4Solid::convertPolycone@1725] all_z_descending detected, reversing PMT_3inch_pmt_solid_cyl0x68fdd10
    2023-08-19 17:35:38.548 INFO  [32807060] [GInstancer::dumpRepeatCandidates@464]  num_repcan 8 dmax 20



DONE : Added logging : looking for  95,96, 106,107,108, 117
--------------------------------------------------------------

::

    2023-08-19 20:46:11.509 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 94 lvIdx 94 treeidx 94
    2023-08-19 20:46:11.509 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 94 lvIdx 94


    2023-08-19 20:46:11.510 ERROR [33035526] [X4Solid::convertPolycone@1725] all_z_descending detected, reversing base_steel0x5aa4870
    2023-08-19 20:46:11.510 FATAL [33035526] [*X4Solid::Polycone_MakeInner@1843]  EXPERIMENTAL num_R_inner > 1 handling :  name base_steel0x5aa4870 num_R_inner 2 lvIdx 95
    2023-08-19 20:46:11.510 ERROR [33035526] [*X4Solid::Polycone_MakeInner@1854]  inner_prims.size 2 lvIdx 95
    2023-08-19 20:46:11.510 ERROR [33035526] [*X4Solid::Polycone_MakeInner@1869]  lower.is_znudge_capable lvIdx 95
    2023-08-19 20:46:11.511 INFO  [33035526] [ncylinder::decrease_z1@139]  treeidx -1 _z1 -101 dz 1 new_z1 -102
    2023-08-19 20:46:11.511 ERROR [33035526] [*X4Solid::Polycone_MakeInner@1872]  upper.is_znudge_capable lvIdx 95
    2023-08-19 20:46:11.511 INFO  [33035526] [ncylinder::increase_z2@122]  treeidx -1 _z2 0 dz 1 new_z2 1
    2023-08-19 20:46:11.511 INFO  [33035526] [ncylinder::decrease_z1@139]  treeidx -1 _z1 -15 dz 1 new_z1 -16
    2023-08-19 20:46:11.511 ERROR [33035526] [*X4Solid::Polycone_MakeInner@1925]  after znudges lvIdx 95
    2023-08-19 20:46:11.511 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 95 lvIdx 95 treeidx 95
    2023-08-19 20:46:11.511 INFO  [33035526] [ncylinder::increase_z2@122]  treeidx 95 _z2 -15 dz 1 new_z2 -14
    2023-08-19 20:46:11.514 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 95 lvIdx 95
    2023-08-19 20:46:11.514 ERROR [33035526] [X4Solid::convertPolycone@1725] all_z_descending detected, reversing solidAddition_down0x5ba5d90


    2023-08-19 20:46:11.515 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 96 lvIdx 96 treeidx 96
    2023-08-19 20:46:11.515 INFO  [33035526] [ncone::increase_z2@119]  treeidx 96 dz 1 _r1 200 _z1 -140 _r2 450 _z2 0 new_z2 1 new_r2 451.786
    2023-08-19 20:46:11.519 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 96 lvIdx 96


    2023-08-19 20:46:11.581 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 106 lvIdx 106 treeidx 106
    2023-08-19 20:46:11.581 INFO  [33035526] [nzsphere::decrease_z1@111]  treeidx 106 dz 1
    2023-08-19 20:46:11.581 INFO  [33035526] [nzsphere::increase_z2@99]  treeidx 106 dz 1
    2023-08-19 20:46:11.584 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 106 lvIdx 106


    2023-08-19 20:46:11.590 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 107 lvIdx 107 treeidx 107
    2023-08-19 20:46:11.590 INFO  [33035526] [nzsphere::decrease_z1@111]  treeidx 107 dz 1
    2023-08-19 20:46:11.590 INFO  [33035526] [nzsphere::increase_z2@99]  treeidx 107 dz 1
    2023-08-19 20:46:11.593 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 107 lvIdx 107


    2023-08-19 20:46:11.599 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 108 lvIdx 108 treeidx 108
    2023-08-19 20:46:11.599 INFO  [33035526] [ncylinder::increase_z2@122]  treeidx 108 _z2 0 dz 1 new_z2 1
    2023-08-19 20:46:11.599 INFO  [33035526] [ncylinder::increase_z2@122]  treeidx 108 _z2 100 dz 1 new_z2 101
    2023-08-19 20:46:11.600 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 108 lvIdx 108

    2023-08-19 20:46:11.648 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 117 lvIdx 117 treeidx 117
    2023-08-19 20:46:11.648 INFO  [33035526] [ncylinder::increase_z2@122]  treeidx 117 _z2 0 dz 1 new_z2 1
    2023-08-19 20:46:11.648 INFO  [33035526] [ncylinder::increase_z2@122]  treeidx 117 _z2 97 dz 1 new_z2 98
    2023-08-19 20:46:11.650 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 117 lvIdx 117


    2023-08-19 20:46:11.730 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 118 lvIdx 118 treeidx 118
    2023-08-19 20:46:11.732 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 118 lvIdx 118
    2023-08-19 20:46:11.732 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 119 lvIdx 119 treeidx 119
    2023-08-19 20:46:11.732 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 119 lvIdx 119
    2023-08-19 20:46:11.733 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 120 lvIdx 120 treeidx 120
    2023-08-19 20:46:11.734 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 120 lvIdx 120
    2023-08-19 20:46:11.734 ERROR [33035526] [X4Solid::convertPolycone@1725] all_z_descending detected, reversing PMT_3inch_cntr_solid0x68fff60
    2023-08-19 20:46:11.734 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 121 lvIdx 121 treeidx 121
    2023-08-19 20:46:11.735 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 121 lvIdx 121
    2023-08-19 20:46:11.735 ERROR [33035526] [X4Solid::convertPolycone@1725] all_z_descending detected, reversing PMT_3inch_pmt_solid_cyl0x68fdd10
    2023-08-19 20:46:11.735 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 122 lvIdx 122 treeidx 122
    2023-08-19 20:46:11.737 INFO  [33035526] [*NCSG::Adopt@186]  ]  soIdx 122 lvIdx 122
    2023-08-19 20:46:11.805 INFO  [33035526] [*NCSG::Adopt@169]  [  soIdx 123 lvIdx 123 treeidx 123


DONE : Confirmed that uncoincidence shifts in A explains all the above CSGNode diffs
-------------------------------------------------------------------------------------------------

Where is Uncoincidence done in A side, how to disable ?::

    epsilon:opticks blyth$ opticks-fl NTreeProcess
    ./extg4/X4Solid.cc
    ./extg4/X4PhysicalVolume.cc
    ./extg4/tests/X4GDMLBalanceTest.cc
    ./extg4/X4CSG.cc
    ./GeoChain/GeoChain.hh
    ./GeoChain/tests/GeoChainSolidTest.cc
    ./GeoChain/translate.sh
    ./optickscore/Opticks.cc
    ./npy/NTreePositive.hpp
    ./npy/CMakeLists.txt
    ./npy/tests/NTreeBalanceTest.cc
    ./npy/NTreeProcess.hpp
    ./npy/NTreeProcess.cpp
    epsilon:opticks blyth$ 

    epsilon:npy blyth$ opticks-fl NNodeNudger
    ./ana/geocache.bash
    ./bin/ab.bash
    ./integration/tests/tboolean.bash
    ./extg4/X4Solid.cc
    ./extg4/X4PhysicalVolume.cc
    ./GeoChain/translate.sh
    ./u4/U4Solid.h
    ./optickscore/OpticksCfg.cc
    ./npy/NNodeUncoincide.hpp
    ./npy/CMakeLists.txt
    ./npy/NNodeNudger.cpp
    ./npy/NCSG.cpp
    ./npy/NNodeNudger.hpp
    ./npy/NCSG.hpp
    ./npy/NNodeUncoincide.cpp
    ./npy/NNodeEnum.cpp
    epsilon:opticks blyth$ 


::

    1837 nnode* X4Solid::Polycone_MakeInner(const std::vector<zplane>& zp, const char* name, unsigned num_R_inner) // static 
    1838 {
    1839     LOG(fatal) << " EXPERIMENTAL num_R_inner > 1 handling "  << name << " num_R_inner " << num_R_inner  ;
    1840 
    1841     std::vector<nnode*> inner_prims ;
    1842     Polycone_MakePrims( zp, inner_prims, name, false  );
    1843 
    1844     unsigned num_prims = inner_prims.size() ;
    1845     LOG(error) << " inner_prims.size " << num_prims ;
    1846 
    1847     nnode* lower = inner_prims[0] ;
    1848     nnode* upper = inner_prims[inner_prims.size()-1] ;
    1849 
    1850     // polycone made up of cone and cylinder so should all be znudge capable
    1851     // HUH: should be looping over pairs when num_prims > 2 
    1852 
    1853     if( lower->is_znudge_capable() &&  upper->is_znudge_capable()  )
    1854     {
    1855         float dz = 1.0 ;
    1856 
    1857         LOG(error) << " lower.is_znudge_capable " ;
    1858         lower->decrease_z1(dz);
    1859 
    1860         LOG(error) << " upper.is_znudge_capable " ;
    1861         upper->increase_z2(dz);
    1862 
    1863         if( num_prims == 2 )
    1864         {
    1865             // see NNodeNudger::znudge_union_maxmin  expand the z on the smaller r side
    1866 
    1867             nnode* j = upper ;
    1868             nnode* i = lower ;
    1869 






FIXED : CSGNode typecode diffs by calling sn::positivize from U4Solid::init_Tree
----------------------------------------------------------------------------------

After FIX::

    In [1]: a_tc
    Out[1]: array([110, 110, 110,   2, 105, 110,   2, 105, 110, 110, ..., 110, 110, 110, 110, 110, 110, 110, 110, 110, 110], dtype=int32)

    In [2]: b_tc
    Out[2]: array([110, 110, 110,   2, 105, 110,   2, 105, 110, 110, ..., 110, 110, 110, 110, 110, 110, 110, 110, 110, 110], dtype=int32)

    In [3]: np.where(a_tc != b_tc)
    Out[3]: (array([], dtype=int64),)


Before FIX::

    In [59]: a_tc = a.node[:,3,2].view(np.int32)

    In [61]: b_tc = b.node[:,3,2].view(np.int32)

    In [62]: np.where( a_tc != b_tc )
    Out[62]:
    (array([    3,     6,    10,    13,    18,    21,    24, 15662, 15666, 15683, 15685, 15690, 15692, 15695, 15707, 15710, 15713, 15724, 15726, 15731, 15733, 15736, 15776, 15779, 15782, 15785, 15788,
            15792, 15796, 15798, 15816, 15824, 15826, 15831]),)

    In [63]: w_tc = np.where( a_tc != b_tc )[0]

    In [64]: w_tc.shape
    Out[64]: (34,)

    In [65]: np.c_[a_tc[w_tc],b_tc[w_tc]]
    Out[65]:
    array([[2, 3],
           [2, 3],
           ..
           [2, 3],
           [2, 3],
           [2, 3],
           [2, 1],
           [2, 3],
           [2, 1],
           [2, 1],

Mostly::

    B:3:CSG_DIFFERENCE => A:2:CSG_INTERSECTION


    In [68]: np.unique( np.c_[a_tc[w_tc],b_tc[w_tc]], axis=0, return_counts=True )
    Out[68]:
    (array([[2, 1],
            [2, 3]], dtype=int32),
     array([ 8, 26]))


::

     22 typedef enum {
     23     CSG_ZERO=0,
     24     CSG_OFFSET_LIST=4,
     25     CSG_OFFSET_LEAF=7,
     26
     27     CSG_TREE=1,
     28         CSG_UNION=1,
     29         CSG_INTERSECTION=2,
     30         CSG_DIFFERENCE=3,
     31


Resonable to presume those are all CSG trees that start having CSG_DIFFERENCE, 
which is positivized in A but not yet B::

    In [21]: np.c_[b.primname[np.unique(b.find_primIdx_from_nodeIdx(w_tc))]]
    Out[21]: 
    array([['sTopRock_dome'],
           ['sTopRock_domeAir'],
           ['sExpHall'],
           ['PoolCoversub'],
           ['Upper_Steel_tube'],
           ['Upper_Tyvek_tube'],
           ['sAirTT'],
           ['sChimneyAcrylic'],
           ['sChimneySteel'],
           ['NNVTMCPPMTsMask'],
           ['NNVTMCPPMTTail'],
           ['NNVTMCPPMT_PMT_20inch_edge_solid'],
           ['NNVTMCPPMT_PMT_20inch_plate_solid'],
           ['NNVTMCPPMT_PMT_20inch_tube_solid'],
           ['HamamatsuR12860sMask'],
           ['HamamatsuR12860Tail'],
           ['HamamatsuR12860_PMT_20inch_plate_solid'],
           ['HamamatsuR12860_PMT_20inch_outer_edge_solid'],
           ['HamamatsuR12860_PMT_20inch_inner_edge_solid'],
           ['HamamatsuR12860_PMT_20inch_inner_ring_solid'],
           ['HamamatsuR12860_PMT_20inch_dynode_tube_solid'],
           ['HamamatsuR12860_PMT_20inch_shield_solid'],
           ['mask_PMT_20inch_vetosMask'],
           ['PMT_20inch_veto_inner2_solid'],
           ['base_steel'],
           ['uni_acrylic1']], dtype=object)


DONE : Added sn::positivize to U4Solid::init_Tree
------------------------------------------------------

::

    269 inline void U4Solid::init()
    270 {
    271     init_Constituents();
    272     init_Check();
    273     init_Tree() ;
    274 }
    275 
    276 inline void U4Solid::init_Tree()
    277 {
    278 #ifdef WITH_SND
    279     assert( root > -1 );
    280     snd::SetLVID(root, lvid );
    281     std::cerr << "U4Solid::init_Tree.WITH_SND.FATAL snd.hh does not provide positivize" << std::endl ;   
    282     assert(0); 
    283 #else
    284     assert( root); 
    285     root->set_lvid(lvid);
    286     root->positivize() ;
    287 #endif
    288 }


A/B bnd are off : THIS IS osur ?
-----------------------------------

::

    In [9]: a_bnd =  a.node[:,1,2].view(np.int32)

    In [10]: b_bnd = b.node[:,1,2].view(np.int32)

    In [11]: a_bnd.min()
    Out[11]: 0

    In [12]: a_bnd.max()
    Out[12]: 42

    In [13]: b_bnd.max()
    Out[13]: 123

    In [14]: b_bnd.min()
    Out[14]: 0


::

    In [22]: A.SSim.stree.standard.bnd_names.shape   ## A using GGeo bndname not this ?
    Out[22]: (124,)

    In [23]: B.SSim.stree.standard.bnd_names.shape
    Out[23]: (124,)

    In [24]: np.all( A.SSim.stree.standard.bnd_names == B.SSim.stree.standard.bnd_names )
    Out[24]: True



A : nudge stack
-----------------


::

    BP=ncone::increase_z2 ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh dbg


::

    lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x000000010a7105bb libNPY.dylib`ncone::increase_z2(this=0x0000000132ecaa70, dz=1) at NCone.cpp:109
        frame #1: 0x000000010a6f5c25 libNPY.dylib`NNodeNudger::znudge_union_maxmin(this=0x0000000132ecbb90, coin=0x0000000132ecb200) at NNodeNudger.cpp:491
        frame #2: 0x000000010a6f4d12 libNPY.dylib`NNodeNudger::znudge(this=0x0000000132ecbb90, coin=0x0000000132ecb200) at NNodeNudger.cpp:299
        frame #3: 0x000000010a6f2bbc libNPY.dylib`NNodeNudger::uncoincide(this=0x0000000132ecbb90) at NNodeNudger.cpp:286
        frame #4: 0x000000010a6f11ed libNPY.dylib`NNodeNudger::init(this=0x0000000132ecbb90) at NNodeNudger.cpp:92
        frame #5: 0x000000010a6f0c36 libNPY.dylib`NNodeNudger::NNodeNudger(this=0x0000000132ecbb90, root_=0x0000000132ecb300, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:66
        frame #6: 0x000000010a6f15bd libNPY.dylib`NNodeNudger::NNodeNudger(this=0x0000000132ecbb90, root_=0x0000000132ecb300, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:64
        frame #7: 0x000000010a7843a3 libNPY.dylib`NCSG::MakeNudger(msg="Adopt root ctor", root=0x0000000132ecb300, surface_epsilon=0.00000999999974) at NCSG.cpp:284
        frame #8: 0x000000010a7844f7 libNPY.dylib`NCSG::NCSG(this=0x0000000132ecbad0, root=0x0000000132ecb300) at NCSG.cpp:317
        frame #9: 0x000000010a7836ed libNPY.dylib`NCSG::NCSG(this=0x0000000132ecbad0, root=0x0000000132ecb300) at NCSG.cpp:332
        frame #10: 0x000000010a7834c1 libNPY.dylib`NCSG::Adopt(root=0x0000000132ecb300, config=0x0000000000000000, soIdx=96, lvIdx=96) at NCSG.cpp:177
        frame #11: 0x00000001008ad8d7 libExtG4.dylib`X4PhysicalVolume::ConvertSolid_FromRawNode(ok=0x0000000132d989f0, lvIdx=96, soIdx=96, solid=0x000000010c464d50, soname="uni_acrylic10x5ba6710", lvname="lAddition0x5ba7be0", balance_deep_tree=true, raw=0x0000000132ecb300) at X4PhysicalVolume.cc:1226
        frame #12: 0x00000001008ad2fc libExtG4.dylib`X4PhysicalVolume::ConvertSolid_(ok=0x0000000132d989f0, lvIdx=96, soIdx=96, solid=0x000000010c464d50, soname="uni_acrylic10x5ba6710", lvname="lAddition0x5ba7be0", balance_deep_tree=true) at X4PhysicalVolume.cc:1192
        frame #13: 0x00000001008ac33d libExtG4.dylib`X4PhysicalVolume::ConvertSolid(ok=0x0000000132d989f0, lvIdx=96, soIdx=96, solid=0x000000010c464d50, soname="uni_acrylic10x5ba6710", lvname="lAddition0x5ba7be0") at X4PhysicalVolume.cc:1090
        frame #14: 0x00000001008aac05 libExtG4.dylib`X4PhysicalVolume::convertSolid(this=0x00007ffeefbfc978, lv=0x000000010c196f30) at X4PhysicalVolume.cc:1035
        frame #15: 0x00000001008a9b25 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfc978, pv=0x000000010c1cc870, depth=6) at X4PhysicalVolume.cc:994
        frame #16: 0x00000001008a9844 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfc978, pv=0x0000000124a10040, depth=5) at X4PhysicalVolume.cc:988
        frame #17: 0x00000001008a9844 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfc978, pv=0x00000001244bdf70, depth=4) at X4PhysicalVolume.cc:988
        frame #18: 0x00000001008a9844 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfc978, pv=0x0000000124a928d0, depth=3) at X4PhysicalVolume.cc:988
        frame #19: 0x00000001008a9844 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfc978, pv=0x0000000124a92a00, depth=2) at X4PhysicalVolume.cc:988
        frame #20: 0x00000001008a9844 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfc978, pv=0x0000000124988ca0, depth=1) at X4PhysicalVolume.cc:988
        frame #21: 0x00000001008a9844 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfc978, pv=0x000000010f9135d0, depth=0) at X4PhysicalVolume.cc:988
        frame #22: 0x00000001008a422a libExtG4.dylib`X4PhysicalVolume::convertSolids(this=0x00007ffeefbfc978) at X4PhysicalVolume.cc:946
        frame #23: 0x00000001008a2f43 libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfc978) at X4PhysicalVolume.cc:212
        frame #24: 0x00000001008a2bba libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfc978, ggeo=0x0000000132dafe60, top=0x000000010f9135d0) at X4PhysicalVolume.cc:191
        frame #25: 0x00000001008a30a5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfc978, ggeo=0x0000000132dafe60, top=0x000000010f9135d0) at X4PhysicalVolume.cc:182
        frame #26: 0x000000010089b9a8 libExtG4.dylib`X4Geo::Translate(top=0x000000010f9135d0) at X4Geo.cc:25
        frame #27: 0x0000000100120218 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010c4574b0, world=0x000000010f9135d0) at G4CXOpticks.cc:267
        frame #28: 0x0000000100121b59 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010c4574b0, gdmlpath="/Users/blyth/.opticks/GEO



    (lldb) f 8 
    frame #8: 0x000000010a7844f7 libNPY.dylib`NCSG::NCSG(this=0x0000000132ecbad0, root=0x0000000132ecb300) at NCSG.cpp:317
       314 	    m_root(root),
       315 	    m_points(NULL),
       316 	    m_uncoincide(make_uncoincide()),
    -> 317 	    m_nudger(MakeNudger("Adopt root ctor", root, SURFACE_EPSILON)),
       318 	    m_csgdata(new NCSGData),
       319 	    m_adopted(true), 
       320 	    m_boundary(NULL),
    (lldb) p root->get_treeidx()
    (int) $2 = 96
    (lldb) 

     274 NNodeNudger* NCSG::MakeNudger(const char* msg, nnode* root, float surface_epsilon )   // static  
     275 {
     276     int treeidx = root->get_treeidx();
     277     bool nudgeskip = root->is_nudgeskip() ;
     278 
     279     LOG(LEVEL)
     280         << " treeidx " << treeidx
     281         << " nudgeskip " << nudgeskip
     282          ;
     283 
     284     NNodeNudger* nudger = nudgeskip ? nullptr : new NNodeNudger(root, surface_epsilon, root->verbosity);
     285     return nudger ;
     286 }


    (lldb) f 4
    frame #4: 0x000000010a6f11ed libNPY.dylib`NNodeNudger::init(this=0x0000000132ecbb90) at NNodeNudger.cpp:92
       89  	    collect_coincidence();
       90  	
       91  	    if(enabled)
    -> 92  	       uncoincide();
       93  	
       94  	    bool out = listed || nudges.size() > 0 ; 
       95  	    LOG_IF(LEVEL, out ) << brief() ;  
    (lldb) 


Add disable NNodeNudger__DISABLE::

    057 NNodeNudger::NNodeNudger(nnode* root_, float epsilon_, unsigned /*verbosity*/)
     58     :
     59     root(root_),
     60     epsilon(epsilon_),
     61     verbosity(SSys::getenvint("VERBOSITY",1)),
     62     listed(false),
     63     enabled(!SSys::getenvbool("NNodeNudger__DISABLE"))
     64 {
     65     root->check_tree( FEATURE_GTRANSFORMS | FEATURE_PARENT_LINKS );
     66     init();
     67 }

    (lldb) p root->get_treeidx()
    (int) $6 = 96
    (lldb) f 3
    frame #3: 0x000000010a6f2bbc libNPY.dylib`NNodeNudger::uncoincide(this=0x0000000132ecbb90) at NNodeNudger.cpp:286
       283 	   unsigned num_coincidence = coincidence.size();
       284 	   for(unsigned i=0 ; i < num_coincidence ; i++)
       285 	   {
    -> 286 	       znudge(&coincidence[i]);
       287 	   }
       288 	}
       289 	
    (lldb) 

    (lldb) f 2
    frame #2: 0x000000010a6f4d12 libNPY.dylib`NNodeNudger::znudge(this=0x0000000132ecbb90, coin=0x0000000132ecb200) at NNodeNudger.cpp:299
       296 	    if( can_znudge_union_maxmin(coin) ) 
       297 	    {
       298 	        LOG(LEVEL) << "proceed znudge_union_maxmin " << desc_znudge_union_maxmin(coin) ; 
    -> 299 	        znudge_union_maxmin(coin);
       300 	    }
       301 	    else
       302 	    {
    (lldb) 

    (lldb) f 1
    frame #1: 0x000000010a6f5c25 libNPY.dylib`NNodeNudger::znudge_union_maxmin(this=0x0000000132ecbb90, coin=0x0000000132ecb200) at NNodeNudger.cpp:491
       488 	
       489 	        **/ 
       490 	
    -> 491 	        i->increase_z2( dz ); 
       492 	        coin->n = NUDGE_I_INCREASE_Z2 ; 
       493 	    }
       494 	
    (lldb) p i 
    (ncone *) $7 = 0x0000000132ecaa70
    (lldb) p i->get_treeidx()
    (int) $8 = -1
    (lldb) 


HMM : THE set_treeidx needs to recurse



::

    1205 GMesh* X4PhysicalVolume::ConvertSolid_FromRawNode( const Opticks* ok, int lvIdx, int soIdx, const G4VSolid* const solid, c     onst char* soname, const char* lvname, bool balance_deep_tree,
    1206      nnode* raw)
    1207 {
    1208     bool is_x4balanceskip = ok->isX4BalanceSkip(lvIdx) ;
    1209     bool is_x4polyskip = ok->isX4PolySkip(lvIdx);   // --x4polyskip 211,232
    1210     bool is_x4nudgeskip = ok->isX4NudgeSkip(lvIdx) ;
    1211     bool is_x4pointskip = ok->isX4PointSkip(lvIdx) ;
    1212     bool do_balance = balance_deep_tree && !is_x4balanceskip ;
    1213 
    1214     nnode* root = do_balance ? NTreeProcess<nnode>::Process(raw, soIdx, lvIdx) : raw ;
    1215 
    1216     LOG(LEVEL) << " after NTreeProcess:::Process " ;
    1217 
    1218     root->other = raw ;
    1219     root->set_nudgeskip( is_x4nudgeskip );
    1220     root->set_pointskip( is_x4pointskip );
    1221     root->set_treeidx( lvIdx );
    1222 
    1223     const NSceneConfig* config = NULL ;
    1224 
    1225     LOG(LEVEL) << "[ before NCSG::Adopt " ;
    1226     NCSG* csg = NCSG::Adopt( root, config, soIdx, lvIdx );   // Adopt exports nnode tree to m_nodes buffer in NCSG instanc     e
    1227     LOG(LEVEL) << "] after NCSG::Adopt " ;
    1228     assert( csg ) ;
    1229     assert( csg->isUsedGlobally() );
    1230 
    1231     bool is_balanced = root != raw ;
    1232     if(is_balanced) assert( balance_deep_tree == true );
    1233 
    1234     csg->set_balanced(is_balanced) ;
    1235     csg->set_soname( soname ) ;
    1236     csg->set_lvname( lvname ) ;




With "export NNodeNudger__DISABLE=1" : get 2 base_steel CSGNode discrepant LV:95
-----------------------------------------------------------------------------------


::

    In [2]: w = np.where(ab.node > 1e-2 )[0]

    In [3]: w
    Out[3]: array([15829, 15830])

    In [4]: a.node.shape
    Out[4]: (15968, 4, 4)

    In [5]: b.node.shape
    Out[5]: (15968, 4, 4)

    In [6]: a.find_primIdx_from_nodeIdx(w)
    Out[6]: array([2956, 2956], dtype=int32)

    In [7]: a.primname[a.find_primIdx_from_nodeIdx(w)]
    Out[7]: array(['base_steel0x5aa4870', 'base_steel0x5aa4870'], dtype=object)

    In [8]: b.primname[b.find_primIdx_from_nodeIdx(w)]
    Out[8]: array(['base_steel', 'base_steel'], dtype=object)


    In [11]: np.c_[a.node[w],b.node[w],a.node[w]-b.node[w]]
    Out[11]:
    array([[[   0. ,    0. ,    0. ,   55.5,    0. ,    0. ,    0. ,   55.5,    0. ,    0. ,    0. ,    0. ],
            [-102. ,  -15. ,    0. ,    0. , -101. ,  -15. ,    0. ,    0. ,   -1. ,    0. ,   -0. ,   -0. ],
            [ -55.5,  -55.5, -102. ,   55.5,  -55.5,  -55.5, -101. ,   55.5,    0. ,    0. ,   -1. ,    0. ],
            [  55.5,  -15. ,    0. ,   -0. ,   55.5,  -15. ,    0. ,   -0. ,    0. ,    0. ,    0. ,    0. ]],

           [[   0. ,    0. ,    0. ,   43. ,    0. ,    0. ,    0. ,   43. ,    0. ,    0. ,    0. ,    0. ],
            [ -16. ,    1. ,    0. ,    0. ,  -15. ,    0. ,    0. ,    0. ,   -1. ,    1. ,   -0. ,   -0. ],
            [ -43. ,  -43. ,  -16. ,   43. ,  -43. ,  -43. ,  -15. ,   43. ,    0. ,    0. ,   -1. ,    0. ],
            [  43. ,    1. ,    0. ,   -0. ,   43. ,    0. ,    0. ,   -0. ,    0. ,    1. ,    0. ,    0. ]]], dtype=float32)

    In [12]:

    In [12]: a.node[w]
    Out[12]:
    array([[[   0. ,    0. ,    0. ,   55.5],       ## cx,cy,cz,radius 
            [-102. ,  -15. ,    0. ,    0. ],       ## z1,z2,i:bd,i:idx
            [ -55.5,  -55.5, -102. ,   55.5],
            [  55.5,  -15. ,    0. ,   -0. ]],

           [[   0. ,    0. ,    0. ,   43. ],
            [ -16. ,    1. ,    0. ,    0. ],
            [ -43. ,  -43. ,  -16. ,   43. ],
            [  43. ,    1. ,    0. ,   -0. ]]], dtype=float32)

    ## HMM -102 : grown lower edge ? 

    In [13]: b.node[w]
    Out[13]:
    array([[[   0. ,    0. ,    0. ,   55.5],
            [-101. ,  -15. ,    0. ,    0. ],
            [ -55.5,  -55.5, -101. ,   55.5],
            [  55.5,  -15. ,    0. ,   -0. ]],

           [[   0. ,    0. ,    0. ,   43. ],
            [ -15. ,    0. ,    0. ,    0. ],
            [ -43. ,  -43. ,  -15. ,   43. ],
            [  43. ,    0. ,    0. ,   -0. ]]], dtype=float32)

    In [14]: a.node[w] - b.node[w]
    Out[14]:
    array([[[ 0.,  0.,  0.,  0.],
            [-1.,  0., -0., -0.],
            [ 0.,  0., -1.,  0.],
            [ 0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [-1.,  1., -0., -0.],
            [ 0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.]]], dtype=float32)

    In [15]:

    In [15]: a.node.view(np.int32)[w,3,2]
    Out[15]: array([105, 105], dtype=int32)

    In [16]: b.node.view(np.int32)[w,3,2]
    Out[16]: array([105, 105], dtype=int32)

::

     38     CSG_LEAF=101,
     39         CSG_SPHERE=101,
     40         CSG_BOX=102,
     41         CSG_ZSPHERE=103,
     42         CSG_TUBS=104,
     43         CSG_CYLINDER=105,
     44         CSG_SLAB=106,
     45         CSG_PLANE=107,
     46         CSG_CONE=108,
     47         CSG_EXBB=109,
     48         CSG_BOX3=110,



::

    In [18]: b.prim[b.find_primIdx_from_nodeIdx(w)].view(np.int32)
    Out[18]:
    array([[[    7, 15824,  7254,     0],   ##  numNode, nodeOffset, tranOffset, planOffset
            [    0,    95,     6,     0],   ##  sbt    , lvIdx,    , repeatIdx , primIdx 
            [    0,     0,     0,     0],
            [    0,     0,     0,     0]],

           [[    7, 15824,  7254,     0],
            [    0,    95,     6,     0],
            [    0,     0,     0,     0],
            [    0,     0,     0,     0]]], dtype=int32)


    In [20]: b.meshname[95]
    Out[20]: 'base_steel'

    In [21]: a.meshname[95]
    Out[21]: 'base_steel0x5aa4870'







::

    2023-08-20 13:35:31.605 INFO  [33305488] [*NCSG::Adopt@186]  ]  soIdx 93 lvIdx 93
    2023-08-20 13:35:31.606 INFO  [33305488] [*NCSG::Adopt@169]  [  soIdx 94 lvIdx 94 treeidx 94
    2023-08-20 13:35:31.606 INFO  [33305488] [*NCSG::Adopt@186]  ]  soIdx 94 lvIdx 94
    2023-08-20 13:35:31.607 ERROR [33305488] [X4Solid::convertPolycone@1725] all_z_descending detected, reversing base_steel0x5aa4870
    2023-08-20 13:35:31.607 FATAL [33305488] [*X4Solid::Polycone_MakeInner@1843]  EXPERIMENTAL num_R_inner > 1 handling :  name base_steel0x5aa4870 num_R_inner 2 lvIdx 95
    2023-08-20 13:35:31.607 ERROR [33305488] [*X4Solid::Polycone_MakeInner@1854]  inner_prims.size 2 lvIdx 95
    2023-08-20 13:35:31.607 ERROR [33305488] [*X4Solid::Polycone_MakeInner@1869]  lower.is_znudge_capable lvIdx 95
    2023-08-20 13:35:31.607 INFO  [33305488] [ncylinder::decrease_z1@139]  treeidx -1 _z1 -101 dz 1 new_z1 -102
    2023-08-20 13:35:31.607 ERROR [33305488] [*X4Solid::Polycone_MakeInner@1872]  upper.is_znudge_capable lvIdx 95
    2023-08-20 13:35:31.607 INFO  [33305488] [ncylinder::increase_z2@122]  treeidx -1 _z2 0 dz 1 new_z2 1
    2023-08-20 13:35:31.607 INFO  [33305488] [ncylinder::decrease_z1@139]  treeidx -1 _z1 -15 dz 1 new_z1 -16
    2023-08-20 13:35:31.607 ERROR [33305488] [*X4Solid::Polycone_MakeInner@1925]  after znudges lvIdx 95
    2023-08-20 13:35:31.608 INFO  [33305488] [*NCSG::Adopt@169]  [  soIdx 95 lvIdx 95 treeidx 95
    2023-08-20 13:35:31.608 INFO  [33305488] [*NCSG::Adopt@186]  ]  soIdx 95 lvIdx 95
    2023-08-20 13:35:31.609 ERROR [33305488] [X4Solid::convertPolycone@1725] all_z_descending detected, reversing solidAddition_down0x5ba5d90
    2023-08-20 13:35:31.609 INFO  [33305488] [*NCSG::Adopt@169]  [  soIdx 96 lvIdx 96 treeidx 96
    2023-08-20 13:35:31.611 INFO  [33305488] [*NCSG::Adopt@186]  ]  soIdx 96 lvIdx 96
    2023-08-20 13:35:31.613 INFO  [33305488] [*NCSG::Adopt@169]  [  soIdx 97 lvIdx 97 treeidx 97





DONE : Get CSGNode A/B match with NNodeNudger__DISABLE=1 X4Solid__convertPolycone_nudge_mode=0
-------------------------------------------------------------------------------------------------


::

     66 export NNodeNudger__DISABLE=1
     67 export X4Solid__convertPolycone_nudge_mode=0 # DISABLE 

Hence those two nudge locations explain all the A/B CSGNode deviation.::

    In [4]: np.where(ab.node > 1e-2)                                                                                                
    Out[4]: (array([], dtype=int64),)



DONE  : review more general nudging in A and decide how to implement in B : IMPLEMENTED IN SN::UNCOINCIDE
--------------------------------------------------------------------------------------------------------------

* A:X4Solid.cc/NNodeNudger.cpp --> B:U4Solid.h/sn.h

DONE  : review Polycone nudging in A and decide how to implement in B  : IMPLEMENTED IN U4POLYCONE USING SN::ZNUDGE METHODS
------------------------------------------------------------------------------------------------------------------------------

Started with the simpler Polycone nudging as unlike the general case 
that has no transforms and no need to be concerned with finding coincidences 
as every joint is coincident and every inner end is also coincident
in the subtraction. 


* A:X4Solid::Polycone_Nudge --> B:U4Polycone.h/sn.h 

Started with rationalizing U4Polycone.h and filling out sn nudge methods::

     336     static void ZNudgeEnds(  std::vector<sn*>& prims);
     337     static void ZNudgeJoints(std::vector<sn*>& prims);
     338     static std::string ZDesc(const std::vector<sn*>& prims);


::

    In [1]: w = np.where(ab.node > 1e-2)[0]

    In [2]: w
    Out[2]: array([15829, 15830])

    In [4]: b.find_primIdx_from_nodeIdx(w)
    Out[4]: array([2956, 2956], dtype=int32)

    In [5]: b.primname[2956]
    Out[5]: 'base_steel'


::

    In [13]: a.node[w]    ## two subtracted cylinders of different radii
    Out[13]: 
    array([[[   0. ,    0. ,    0. ,   55.5],
            [-102. ,  -15. ,    0. ,    0. ],
            [ -55.5,  -55.5, -102. ,   55.5],
            [  55.5,  -15. ,    0. ,   -0. ]],

           [[   0. ,    0. ,    0. ,   43. ],
            [ -16. ,    1. ,    0. ,    0. ],
            [ -43. ,  -43. ,  -16. ,   43. ],
            [  43. ,    1. ,    0. ,   -0. ]]], dtype=float32)


    In [14]: b.node[w]
    Out[14]: 
    array([[[   0. ,    0. ,    0. ,   55.5],
            [-101. ,  -15. ,    0. ,    0. ],
            [ -55.5,  -55.5, -101. ,   55.5],
            [  55.5,  -15. ,    0. ,   -0. ]],

           [[   0. ,    0. ,    0. ,   43. ],
            [ -15. ,    0. ,    0. ,    0. ],
            [ -43. ,  -43. ,  -15. ,   43. ],
            [  43. ,    0. ,    0. ,   -0. ]]], dtype=float32)

    In [15]: a.node[w] - b.node[w]                                                                                                  
    Out[15]: 
    array([[[ 0.,  0.,  0.,  0.],
            [-1.,  0., -0., -0.],
            [ 0.,  0., -1.,  0.],
            [ 0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [-1.,  1., -0., -0.],
            [ 0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.]]], dtype=float32)

    In [16]: a.node[w].view(np.int32)
    Out[16]: 
    array([[[          0,           0,           0,  1113456640],
            [-1026818048, -1049624576,          23,           5],
            [-1034027008, -1034027008, -1026818048,  1113456640],
            [ 1113456640, -1049624576,         105, -2147476391]],

           [[          0,           0,           0,  1110179840],
            [-1048576000,  1065353216,          23,           6],
            [-1037303808, -1037303808, -1048576000,  1110179840],
            [ 1110179840,  1065353216,         105, -2147476390]]], dtype=int32)





DONE  : After implementing U4Polycone use of sn::ZNudgeExpandEnds : establish baseline
---------------------------------------------------------------------------------------

1. disable nudging from both A and B:: 

A ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh::

    export NNodeNudger__DISABLE=1
    export X4Solid__convertPolycone_nudge_mode=0 # 0:DISABLE 
    export U4Polycone__DISABLE_NUDGE=1 

B ~/opticks/u4/tests/U4TreeCreateSSimTest.sh::

    export U4Polycone__DISABLE_NUDGE=1 


As expected that gives no large CSGNode deviations (but it would have terrible coincidence issues)::

    In [3]: w = np.where(ab.node > 1e-2)[0] ; w
    Out[3]: array([], dtype=int64)


2. Now enable X4Solid__convertPolycone_nudge_mode=1 (the default) by commenting the above A line.   

As expected base_steel polycone becomes discrepant::

    In [1]: w = np.where(ab.node > 1e-2)[0] ; w
    Out[1]: array([15829, 15830])

    In [2]: a.find_primIdx_from_nodeIdx(w)
    Out[2]: array([2956, 2956], dtype=int32)

    In [3]: a.primname[2956]
    Out[3]: 'base_steel0x5aa4870'


3. Then enable the new B side U4Polycone nudging  by commenting the above B line::

    #export U4Polycone__DISABLE_NUDGE=1 
   

Not matching yet because : joints not done, bbox not updated::

    In [4]: a.node[w]
    Out[4]: 
    array([[[   0. ,    0. ,    0. ,   55.5],
            [-102. ,  -15. ,    0. ,    0. ],
            [ -55.5,  -55.5, -102. ,   55.5],
            [  55.5,  -15. ,    0. ,   -0. ]],

           [[   0. ,    0. ,    0. ,   43. ],
            [ -16. ,    1. ,    0. ,    0. ],
            [ -43. ,  -43. ,  -16. ,   43. ],
            [  43. ,    1. ,    0. ,   -0. ]]], dtype=float32)

    In [5]: b.node[w]
    Out[5]: 
    array([[[   0. ,    0. ,    0. ,   55.5],
            [-102. ,  -15. ,    0. ,    0. ],
            [ -55.5,  -55.5, -101. ,   55.5],
            [  55.5,  -15. ,    0. ,   -0. ]],

           [[   0. ,    0. ,    0. ,   43. ],
            [ -15. ,    1. ,    0. ,    0. ],
            [ -43. ,  -43. ,  -15. ,   43. ],
            [  43. ,    0. ,    0. ,   -0. ]]], dtype=float32)

    In [6]: a.node[w]-b.node[w]
    Out[6]: 
    array([[[ 0.,  0.,  0.,  0.],
            [ 0.,  0., -0., -0.],
            [ 0.,  0., -1.,  0.],
            [ 0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [-1.,  0., -0., -0.],
            [ 0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.]]], dtype=float32)



After enable joint and end nudging in B, but with only polycone in A::

    In [1]: w = np.where(ab.node > 1e-2)[0] ; w
    Out[1]: array([15679, 15680, 15720, 15721, 15827, 15829, 15830, 15834])
     

    In [2]: a.find_primIdx_from_nodeIdx(w)
    Out[2]: array([2928, 2928, 2937, 2937, 2956, 2956, 2956, 2957], dtype=int32)


    In [4]: np.c_[a.primname[a.find_primIdx_from_nodeIdx(w)]]
    Out[4]:
    array([['NNVTMCPPMTsMask_virtual0x6173a40'],
           ['NNVTMCPPMTsMask_virtual0x6173a40'],
           ['HamamatsuR12860sMask_virtual0x6163d90'],
           ['HamamatsuR12860sMask_virtual0x6163d90'],
           ['base_steel0x5aa4870'],
           ['base_steel0x5aa4870'],
           ['base_steel0x5aa4870'],
           ['uni_acrylic10x5ba6710']], dtype=object)

    In [5]:

::

    In [8]: a.node.view(np.int32)[w,3,2]
    Out[8]: array([105, 105, 105, 105, 105, 105, 105, 108], dtype=int32)

    In [9]: b.node.view(np.int32)[w,3,2]
    Out[9]: array([105, 105, 105, 105, 105, 105, 105, 108], dtype=int32)  ## cylinders and one cone


Perhaps some diffs from nudger disabled in A::

    In [6]: a.node[w].reshape(-1,16)[:,:6]
    Out[6]: 
    array([[   0.   ,    0.   ,    0.   ,  264.05 , -183.225,    0.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,   97.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 , -183.225,    0.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,  100.   ],
           [   0.   ,    0.   ,    0.   ,   70.   , -101.   ,  -15.   ],
           [   0.   ,    0.   ,    0.   ,   55.5  , -102.   ,  -15.   ],
           [   0.   ,    0.   ,    0.   ,   43.   ,  -16.   ,    1.   ],
           [ 200.   , -140.   ,  450.   ,    0.   ,    0.   ,    0.   ]], dtype=float32)

    In [7]: b.node[w].reshape(-1,16)[:,:6]
    Out[7]: 
    array([[   0.   ,    0.   ,    0.   ,  264.05 , -183.225,    1.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,   98.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 , -183.225,    1.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,  101.   ],
           [   0.   ,    0.   ,    0.   ,   70.   , -101.   ,  -14.   ],
           [   0.   ,    0.   ,    0.   ,   55.5  , -102.   ,  -15.   ],
           [   0.   ,    0.   ,    0.   ,   43.   ,  -16.   ,    1.   ],
           [ 200.   , -140.   ,  450.   ,    1.   ,    0.   ,    0.   ]], dtype=float32)



Now with nudger enabled in A::


    In [1]: w = np.where(ab.node > 1e-2)[0] ; w
    Out[1]: array([15679, 15680, 15720, 15721, 15750, 15753, 15765, 15768, 15827, 15829, 15830, 15834])

    In [2]: a.find_primIdx_from_nodeIdx(w)
    Out[2]: array([2928, 2928, 2937, 2937, 2940, 2940, 2941, 2941, 2956, 2956, 2956, 2957], dtype=int32)

    In [3]: np.c_[a.primname[a.find_primIdx_from_nodeIdx(w)]]
    Out[3]:
    array([['NNVTMCPPMTsMask_virtual0x6173a40'],
           ['NNVTMCPPMTsMask_virtual0x6173a40'],
           ['HamamatsuR12860sMask_virtual0x6163d90'],
           ['HamamatsuR12860sMask_virtual0x6163d90'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['base_steel0x5aa4870'],
           ['base_steel0x5aa4870'],
           ['base_steel0x5aa4870'],
           ['uni_acrylic10x5ba6710']], dtype=object)


    In [5]: a.node[w,:2].reshape(-1,8)
    Out[5]:
    array([[   0.   ,    0.   ,    0.   ,  264.05 , -183.225,    1.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,   98.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 , -183.225,    1.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,  101.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  190.001, -168.226,    1.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  190.001,   -1.   ,  190.101,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  185.   , -163.225,    1.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  185.   ,   -1.   ,  185.1  ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,   70.   , -101.   ,  -14.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,   55.5  , -102.   ,  -15.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,   43.   ,  -16.   ,    1.   ,    0.   ,    0.   ],
           [ 200.   , -140.   ,  451.786,    1.   ,    0.   ,    0.   ,    0.   ,    0.   ]], dtype=float32)

    In [6]: b.node[w,:2].reshape(-1,8)
    Out[6]:
    array([[   0.   ,    0.   ,    0.   ,  264.05 , -183.225,    1.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,   98.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 , -183.225,    1.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  264.05 ,    0.   ,  101.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  190.001, -168.226,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  190.001,    0.   ,  190.101,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  185.   , -163.225,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  185.   ,    0.   ,  185.1  ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,   70.   , -101.   ,  -14.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,   55.5  , -102.   ,  -15.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,   43.   ,  -16.   ,    1.   ,    0.   ,    0.   ],
           [ 200.   , -140.   ,  450.   ,    1.   ,    0.   ,    0.   ,    0.   ,    0.   ]], dtype=float32)

    In [7]: a.node[w,:2].reshape(-1,8) - b.node[w,:2].reshape(-1,8)
    Out[7]: 
    array([[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   , -1.   ,  0.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   , -1.   ,  0.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.   , -0.   ],
           [ 0.   ,  0.   ,  1.786,  0.   ,  0.   ,  0.   , -0.   , -0.   ]], dtype=float32)



Restrict to comparing param, gives 5 discrepant nodes. This is before general 
Nudging implemented in B::

    In [14]: abpar = np.max(np.abs(apar-bpar), axis=1 )

    In [15]: abpar.shape
    Out[15]: (15968,)

    In [16]: a.node.shape
    Out[16]: (15968, 4, 4)

    In [17]: np.where( abpar > 1e-2)[0]
    Out[17]: array([15750, 15753, 15765, 15768, 15834])

    In [18]: w = np.where( abpar > 1e-2)[0]

    In [19]: w
    Out[19]: array([15750, 15753, 15765, 15768, 15834])

    In [20]: a.find_primIdx_from_nodeIdx(w)
    Out[20]: array([2940, 2940, 2941, 2941, 2957], dtype=int32)

    In [21]: np.c_[a.primname[a.find_primIdx_from_nodeIdx(w)]]
    Out[21]:
    array([['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['uni_acrylic10x5ba6710']], dtype=object)

    In [23]: a.node[w,3,2].view(np.int32)
    Out[23]: array([103, 103, 103, 103, 108], dtype=int32)   # 4 cylinders and one cone


    In [23]: a.node[w,3,2].view(np.int32)
    Out[23]: array([103, 103, 103, 103, 108], dtype=int32)

    In [24]: apar[w]
    Out[24]:
    array([[   0.   ,    0.   ,    0.   ,  190.001, -168.226,    1.   ],
           [   0.   ,    0.   ,    0.   ,  190.001,   -1.   ,  190.101],
           [   0.   ,    0.   ,    0.   ,  185.   , -163.225,    1.   ],
           [   0.   ,    0.   ,    0.   ,  185.   ,   -1.   ,  185.1  ],
           [ 200.   , -140.   ,  451.786,    1.   ,    0.   ,    0.   ]], dtype=float32)

    In [25]: bpar[w]
    Out[25]:
    array([[   0.   ,    0.   ,    0.   ,  190.001, -168.226,    0.   ],
           [   0.   ,    0.   ,    0.   ,  190.001,    0.   ,  190.101],
           [   0.   ,    0.   ,    0.   ,  185.   , -163.225,    0.   ],
           [   0.   ,    0.   ,    0.   ,  185.   ,    0.   ,  185.1  ],
           [ 200.   , -140.   ,  450.   ,    1.   ,    0.   ,    0.   ]], dtype=float32)


    In [29]: a.prim[a.find_primIdx_from_nodeIdx(w)].view(np.int32)[:,1,1]
    Out[29]: array([107, 107, 106, 106,  96], dtype=int32)

    In [30]: a.prim[a.find_primIdx_from_nodeIdx(w)].view(np.int32)[:,0,0]
    Out[30]: array([15, 15, 15, 15,  7], dtype=int32)


AHAH : NEED ZSphere nudging for lvid 107, 106::

    2023-08-21 20:41:01.522 INFO  [34056294] [nzsphere::decrease_z1@111]  treeidx 106 dz 1
    2023-08-21 20:41:01.522 INFO  [34056294] [nzsphere::increase_z2@99]  treeidx 106 dz 1
    2023-08-21 20:41:01.530 INFO  [34056294] [nzsphere::decrease_z1@111]  treeidx 107 dz 1
    2023-08-21 20:41:01.530 INFO  [34056294] [nzsphere::increase_z2@99]  treeidx 107 dz 1

::

    2023-08-21 20:41:01.460 INFO  [34056294] [ncone::increase_z2@119]  treeidx 96 dz 1 _r1 200 _z1 -140 _r2 450 _z2 0 new_z2 1 new_r2 451.786





CSGNode Deviations
-------------------

::


    In [10]: w = np.where(ab.node > 1e-2)[0]  ; w
    Out[10]: array([15679, 15680, 15720, 15721, 15750, 15753, 15765, 15768, 15827, 15829, 15830, 15834])

    In [11]: np.c_[a.primname[a.find_primIdx_from_nodeIdx(w)]]
    Out[11]:
    array([['NNVTMCPPMTsMask_virtual0x6173a40'],
           ['NNVTMCPPMTsMask_virtual0x6173a40'],
           ['HamamatsuR12860sMask_virtual0x6163d90'],
           ['HamamatsuR12860sMask_virtual0x6163d90'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['base_steel0x5aa4870'],
           ['base_steel0x5aa4870'],
           ['base_steel0x5aa4870'],
           ['uni_acrylic10x5ba6710']], dtype=object)


FIXED : CSGNode bbox deviants
--------------------------------

Where does the A side update bbox after nudges ? Where to do that on A side ?

::

    In [39]: w = np.where(ab.nbb > 1e-2)[0] ; w
    Out[39]: array([15679, 15680, 15720, 15721, 15750, 15753, 15765, 15768, 15827, 15829, 15830, 15834])

    In [40]: a.find_primIdx_from_nodeIdx(w)
    Out[40]: array([2928, 2928, 2937, 2937, 2940, 2940, 2941, 2941, 2956, 2956, 2956, 2957], dtype=int32)

    In [42]: np.c_[a.primname[np.unique(a.find_primIdx_from_nodeIdx(w))]]
    Out[42]:
    array([['NNVTMCPPMTsMask_virtual0x6173a40'],
           ['HamamatsuR12860sMask_virtual0x6163d90'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['base_steel0x5aa4870'],
           ['uni_acrylic10x5ba6710']], dtype=object)


::

    In [44]: a.node[w].reshape(-1,16)[:,8:14]
    Out[44]: 
    array([[-264.05 , -264.05 , -183.225,  264.05 ,  264.05 ,    1.   ],
           [-264.05 , -264.05 ,    0.   ,  264.05 ,  264.05 ,   98.   ],
           [-264.05 , -264.05 , -183.225,  264.05 ,  264.05 ,    1.   ],
           [-264.05 , -264.05 ,    0.   ,  264.05 ,  264.05 ,  101.   ],
           [-254.001, -254.001, -173.226,  254.001,  254.001,   -4.   ],
           [-254.001, -254.001,   -1.   ,  254.001,  254.001,  190.101],
           [-249.   , -249.   , -168.225,  249.   ,  249.   ,   -4.   ],
           [-249.   , -249.   ,   -1.   ,  249.   ,  249.   ,  185.1  ],
           [ -70.   ,  -70.   , -101.   ,   70.   ,   70.   ,  -14.   ],
           [ -55.5  ,  -55.5  , -102.   ,   55.5  ,   55.5  ,  -15.   ],
           [ -43.   ,  -43.   ,  -16.   ,   43.   ,   43.   ,    1.   ],
           [-451.786, -451.786, -140.   ,  451.786,  451.786,    1.   ]], dtype=float32)

    In [45]: b.node[w].reshape(-1,16)[:,8:14]
    Out[45]: 
    array([[-264.05 , -264.05 , -183.225,  264.05 ,  264.05 ,    0.   ],
           [-264.05 , -264.05 ,    0.   ,  264.05 ,  264.05 ,   97.   ],
           [-264.05 , -264.05 , -183.225,  264.05 ,  264.05 ,    0.   ],
           [-264.05 , -264.05 ,    0.   ,  264.05 ,  264.05 ,  100.   ],
           [-254.001, -254.001, -173.226,  254.001,  254.001,   -5.   ],
           [-254.001, -254.001,    0.   ,  254.001,  254.001,  190.101],
           [-249.   , -249.   , -168.225,  249.   ,  249.   ,   -5.   ],
           [-249.   , -249.   ,    0.   ,  249.   ,  249.   ,  185.1  ],
           [ -70.   ,  -70.   , -101.   ,   70.   ,   70.   ,  -15.   ],
           [ -55.5  ,  -55.5  , -101.   ,   55.5  ,   55.5  ,  -15.   ],
           [ -43.   ,  -43.   ,  -15.   ,   43.   ,   43.   ,    0.   ],
           [-450.   , -450.   , -140.   ,  450.   ,  450.   ,    0.   ]], dtype=float32)




FIXED : CSGNode parameter deviations
---------------------------------------

::

    In [7]: w = np.where(ab.npa > 1e-2)[0] ; w
    Out[7]: array([15750, 15753, 15765, 15768, 15834])

    In [8]: np.c_[a.primname[a.find_primIdx_from_nodeIdx(w)]]
    Out[8]:
    array([['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['uni_acrylic10x5ba6710']], dtype=object)


* parameter deviations from lack of ZSphere nudging for first 4 
  and last one is from lack is specialized cone nudging 


CSGNode parameter deviations, now with specialized cone nudging 
------------------------------------------------------------------

The specialized cone handling avoids param deviation in uni_acrylic1::

    In [1]: w = np.where(ab.npa > 1e-2)[0] ; w
    Out[1]: array([15750, 15753, 15765, 15768])

    In [2]: np.c_[a.primname[a.find_primIdx_from_nodeIdx(w)]]
    Out[2]: 
    array([['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0']], dtype=object)




FIXED : CSGNode parameter deviation for cone in uni_acrylic1 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is not from lack of general nudger because that cone
comes from a polycone. 

AHHA there is special handling ncone::decrease_z1 ncone::increase_z2
for growing cones that avoids changing the cone angle by changing radii too. 


Find the prim::

    In [37]: a_prim_lvid = a.prim.view(np.int32)[:,1,1]

    In [38]: b_prim_lvid = b.prim.view(np.int32)[:,1,1]

    In [39]: np.all( a_prim_lvid == b_prim_lvid )
    Out[39]: True

    In [45]: p = np.where( a_prim_lvid == 96 )[0][0] ; p
    Out[45]: 2957

    In [46]: a.prim[p].view(np.int32)
    Out[46]:
    array([[          7,       15831,        7258,           0],
           [          0,          96,           7,           0],
           [-1008606062, -1008606062, -1022623744,  1138877586],
           [ 1138877586,  1085695590,           0,           0]], dtype=int32)

    In [47]: a.prim[p]
    Out[47]:
    array([[   0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ],
           [-451.786, -451.786, -140.   ,  451.786],
           [ 451.786,    5.7  ,    0.   ,    0.   ]], dtype=float32)

    In [48]: b.prim[p]
    Out[48]:
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], dtype=float32)

    In [49]: b.prim[p].view(np.int32)
    Out[49]:
    array([[    7, 15831,  7258,     0],
           [    0,    96,     7,     0],
           [    0,     0,     0,     0],
           [    0,     0,     0,     0]], dtype=int32)


    In [52]: a.node[15831:15831+7].view(np.int32)[:,3,2]
    Out[52]: array([  2,   1, 101, 108, 105,   0,   0], dtype=int32)

    In [53]: b.node[15831:15831+7].view(np.int32)[:,3,2]
    Out[53]: array([  2,   1, 101, 108, 105,   0,   0], dtype=int32)
                     INT  UNI  !SPH  CON  CYL
                               

                 INT

             UNI      SPH  

          CON   CYL  


Deviation is all in the CONE::

    In [63]: a.node[15831+3:15831+4]
    Out[63]: 
    array([[[ 200.   , -140.   ,  451.786,    1.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [-451.786, -451.786, -140.   ,  451.786],
            [ 451.786,    1.   ,    0.   ,    0.   ]]], dtype=float32)

    In [64]: b.node[15831+3:15831+4]
    Out[64]: 
    array([[[ 200., -140.,  450.,    1.],
            [   0.,    0.,    0.,    0.],
            [-450., -450., -140.,  450.],
            [ 450.,    0.,    0.,    0.]]], dtype=float32)


Polycone that becomes union of cone and cylinder subtracting huge sphere. 



jcv AdditionAcrylicConstruction::

    112     } else if (option=="simple") {
    113 
    114         double ZNodes3[3];
    115         double RminNodes3[3];
    116         double RmaxNodes3[3];
    117         ZNodes3[0] = 5.7*mm; RminNodes3[0] = 0*mm; RmaxNodes3[0] = 450.*mm;
    118         ZNodes3[1] = 0.0*mm; RminNodes3[1] = 0*mm; RmaxNodes3[1] = 450.*mm;
    119         ZNodes3[2] = -140.0*mm; RminNodes3[2] = 0*mm; RmaxNodes3[2] = 200.*mm;
    120 
    121         solidAddition_down = new G4Polycone("solidAddition_down",0.0*deg,360.0*deg,3,ZNodes3,RminNodes3,RmaxNodes3);
    122 
    123     }
    124 
    125 
    126 //    solidAddition_down = new G4Tubs("solidAddition_down",0,199.67*mm,140*mm,0.0*deg,360.0*deg);
    127 //    solidAddition_down = new G4Cons("solidAddition_down",0.*mm,450.*mm,0.*mm,200*mm,70.*mm,0.*deg,360.*deg);
    128     solidAddition_up = new G4Sphere("solidAddition_up",0*mm,m_radAcrylic,0.0*deg,360.0*deg,0.0*deg,180.*deg);
    129 
    130     uni_acrylic1 = new G4SubtractionSolid("uni_acrylic1",solidAddition_down,solidAddition_up,0,G4ThreeVector(0*mm,0*mm,+m_radAcrylic));
    131 
    132     solidAddition_up1 = new G4Tubs("solidAddition_up1",120*mm,208*mm,15.2*mm,0.0*deg,360.0*deg);
    133     uni_acrylic2 = new G4SubtractionSolid("uni_acrylic2",uni_acrylic1,solidAddition_up1,0,G4ThreeVector(0.*mm,0.*mm,-20*mm));



WIP : CSGNode parameter deviation for HAMA solids : Remaining all from lack of ZSPHERE nudge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* difference is lack of ZSPHERE nudge : for ZSPH in union 

::

    In [1]: w = np.where(ab.npa > 1e-2)[0] ; w
    Out[1]: array([15750, 15753, 15765, 15768])

    In [21]: pp = a.find_primIdx_from_nodeIdx(w) ; pp
    Out[21]: array([2940, 2940, 2941, 2941], dtype=int32)

    In [25]: upp = np.unique(pp) ; upp
    Out[25]: array([2940, 2941], dtype=int32)


Param from two nodes in each of these prim is discrepant::

    In [22]: np.c_[a.primname[upp]]
    Out[22]: 
    array([['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0']], dtype=object)


    In [16]: p = a.prim[upp]  

    In [28]: p.view(np.int32)[:,:2].reshape(-1,8)
    Out[28]: 
    array([[   15, 15746,  7217,     0,     3,   107,     3,     3],
           [   15, 15761,  7221,     0,     4,   106,     3,     4]], dtype=int32)
            numNode nodeOff tranOff       sbt   lvid    ridx  primIdx


    In [31]: a.node[15746:15746+15,3,2].view(np.int32)
    Out[31]: array([  1,   1, 108,   1, 103,   0,   0, 103, 105,   0,   0,   0,   0,   0,   0], dtype=int32)
                    UNI   UNI  CONE UNI  ZSP          ZSP   CYL



                   UNI
               UNI      CONE
            UNI   ZSP        
         ZSP  CYL 



::

    In [36]: a.node[15746:15746+15].reshape(-1,16)[:,:6]
    Out[36]: 
    array([[   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],      UNI
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],      UNI
           [ 139.245,    5.99 ,  142.968,   17.17 ,    0.   ,    0.   ],      CONE
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],      UNI
           [   0.   ,    0.   ,    0.   ,  190.001, -168.226,   *1.*  ],      ZSP : zmax increased
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],      ZER
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],      ZER
           [   0.   ,    0.   ,    0.   ,  190.001,  *-1.*  ,  190.101],      ZSP : zmin decreased  
           [   0.   ,    0.   ,    0.   ,  254.001,   -2.5  ,    2.5  ],      CYL
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ]], dtype=float32)

    In [37]: b.node[15746:15746+15].reshape(-1,16)[:,:6]
    Out[37]: 
    array([[   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [ 139.245,    5.99 ,  142.968,   17.17 ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  190.001, -168.226,   *0.*  ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,  190.001,   *0.*  ,  190.101],
           [   0.   ,    0.   ,    0.   ,  254.001,   -2.5  ,    2.5  ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    0.   ,    0.   ,    0.   ,    0.   ]], dtype=float32)

    In [38]: a.node[15746:15746+15].reshape(-1,16)[:,:6] - b.node[15746:15746+15].reshape(-1,16)[:,:6]
    Out[38]: 
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)



WIP : CSGNode bbox deviations
----------------------------------


::

    In [2]: w = np.where(ab.nbb > 1e-2)[0]

    In [3]: w
    Out[3]: array([15679, 15680, 15720, 15721, 15750, 15753, 15765, 15768, 15827, 15829, 15830, 15834])

    In [4]: a.find_primIdx_from_nodeIdx(w)
    Out[4]: array([2928, 2928, 2937, 2937, 2940, 2940, 2941, 2941, 2956, 2956, 2956, 2957], dtype=int32)

    In [5]: np.c_[a.primname[a.find_primIdx_from_nodeIdx(w)]]
    Out[5]:
    array([['NNVTMCPPMTsMask_virtual0x6173a40'],
           ['NNVTMCPPMTsMask_virtual0x6173a40'],
           ['HamamatsuR12860sMask_virtual0x6163d90'],
           ['HamamatsuR12860sMask_virtual0x6163d90'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['base_steel0x5aa4870'],
           ['base_steel0x5aa4870'],
           ['base_steel0x5aa4870'],
           ['uni_acrylic10x5ba6710']], dtype=object)

    In [6]:



Blanket setting bbox in CSGImport looks to cause lots of deviation, but many are CSG_ZERO placeholders::

    In [7]: w = np.where(ab.nbb > 1e-2)[0] ; w
    Out[7]: array([  224,   225,   231,   232,   238,   239,   245,   246,   252,   253, ..., 15772, 15773, 15774, 15775, 15814, 15815, 15821, 15822, 15836, 15837])

    In [8]: w.shape
    Out[8]: (4278,)


After change the CSG_ZERO default to 0.::

    +//const float CSGNode::UNBOUNDED_DEFAULT_EXTENT = 100.f ; 
    +const float CSGNode::UNBOUNDED_DEFAULT_EXTENT = 0.f ; 

Get down to 4 nodes with discrepant bb::

    In [1]: w = np.where(ab.nbb > 1e-2)[0] ; w
    Out[1]: array([15750, 15753, 15765, 15768])


    In [2]: a.find_primIdx_from_nodeIdx(w)
    Out[2]: array([2940, 2940, 2941, 2941], dtype=int32)

    In [3]: np.c_[a.primname[np.unique(a.find_primIdx_from_nodeIdx(w))]]
    Out[3]:
    array([['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0']], dtype=object)


Those nodes are the same as the parameter discrepant 4 CSG_ZSPHERE, 2 from each LV 106,107
Thats HAMA Pyrex and Vacuum:: 

    In [1]: w = np.where(ab.nbb > 1e-2)[0] ; w
    Out[1]: array([15750, 15753, 15765, 15768])

    In [6]: w = np.where(ab.npa > 1e-2)[0] ; w
    Out[6]: array([15750, 15753, 15765, 15768])

    In [2]: a.find_primIdx_from_nodeIdx(w)
    Out[2]: array([2940, 2940, 2941, 2941], dtype=int32)

    In [3]: np.c_[a.primname[np.unique(a.find_primIdx_from_nodeIdx(w))]]
    Out[3]:
    array([['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0']], dtype=object)

    In [4]: w
    Out[4]: array([15750, 15753, 15765, 15768])

    In [5]: a.node.view(np.int32)[w,3,2]
    Out[5]: array([103, 103, 103, 103], dtype=int32)


Encapsulate lvid lookup in CSGFoundry.py::

    In [3]: a.find_lvid_from_nodeIdx(w)
    Out[3]: array([107, 107, 106, 106], dtype=int32)

    In [5]: np.c_[a.find_lvname_from_nodeIdx(w)]
    Out[5]:
    array([['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0'],
           ['HamamatsuR12860_PMT_20inch_inner_solid_1_40x61578a0']], dtype=object)



::

    sn::uncoincide lvid 107 num_prim 4
     idx  457 lvid 107 zs[-190.001,-190.001,  0.000,190.001,190.001,190.101]
     idx  458 lvid 107 cy[-254.001,-254.001, -2.500,254.001,254.001,  2.500]
     idx  460 lvid 107 zs[-190.001,-190.001,-168.226,190.001,190.001,  0.000]
     idx  462 lvid 107 co[-142.968,-142.968,  5.990,142.968,142.968, 17.170]
    ]U4Solid::init 107/0/Uni/463


After using the transforms to stra<double>::Transform_AABB_Inplace::

    sn::uncoincide sn__uncoincide_dump_lvid 107 (-ve for transform details)  lvid 107 num_prim 4
     idx  457 lvid 107 zs[-254.001,-254.001,  0.000,254.001,254.001,190.101]   ## zs coincident with cy at z=0.000
     idx  458 lvid 107 cy[-254.001,-254.001, -5.000,254.001,254.001,  0.000]
     idx  460 lvid 107 zs[-254.001,-254.001,-173.226,254.001,254.001, -5.000]  ## zs coincident with cy at z=-5.000 
     idx  462 lvid 107 co[-142.968,-142.968,-173.226,142.968,142.968,-162.045]

    ]U4Solid::init 107/0/Uni/463


Similar with 106::

    sn::uncoincide sn__uncoincide_dump_lvid 106 (-ve for transform details)  lvid 106 num_prim 4
     idx  450 lvid 106 zs[-249.000,-249.000,  0.000,249.000,249.000,185.100]
     idx  451 lvid 106 cy[-249.000,-249.000, -5.000,249.000,249.000,  0.000]
     idx  453 lvid 106 zs[-249.000,-249.000,-168.225,249.000,249.000, -5.000]
     idx  455 lvid 106 co[-139.625,-139.625,-168.225,139.625,139.625,-158.178]

    ]U4Solid::init 106/0/Uni/456





                   UNI 
               UNI      CONE          CONE IS THE PMT NECK   
            UNI   ZSP        
         ZSP  CYL 



HMM: clearly need to transform the AABB::


    In [3]: print(a.descLVDetail(107))
    descLV lvid:107 meshname:HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280 pidxs:[2940]
    pidx 2940 lv 107 pxl    3 :  HamamatsuR12860_PMT_20inch_pmt_solid_1_40x6152280 : no 15746 nn   15 tcn 3(1:union) 1(108:cone) 2(103:zsphere) 8(0:zero) 1(105:cylinder) tcs [  1   1 108   1 103   0   0 103 105   0   0   0   0   0   0] : bnd 27 : vetoWater/Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLw1.up07_up08_HBeam_phys//LatticedShellSteel
    a.node[15746:15746+15].reshape(-1,16)[:,:6] # descNodeParam
    [[   0.       0.       0.       0.       0.       0.   ]
     [   0.       0.       0.       0.       0.       0.   ]
     [ 139.245    5.99   142.968   17.17     0.       0.   ]
     [   0.       0.       0.       0.       0.       0.   ]
     [   0.       0.       0.     190.001 -168.226    1.   ]
     [   0.       0.       0.       0.       0.       0.   ]
     [   0.       0.       0.       0.       0.       0.   ]
     [   0.       0.       0.     190.001   -1.     190.101]
     [   0.       0.       0.     254.001   -2.5      2.5  ]
     [   0.       0.       0.       0.       0.       0.   ]
     [   0.       0.       0.       0.       0.       0.   ]
     [   0.       0.       0.       0.       0.       0.   ]
     [   0.       0.       0.       0.       0.       0.   ]
     [   0.       0.       0.       0.       0.       0.   ]
     [   0.       0.       0.       0.       0.       0.   ]]
    a.node[15746:15746+15].reshape(-1,16)[:,8:14] # descNodeBB
    [[  -0.      -0.      -0.       0.       0.       0.   ]
     [  -0.      -0.      -0.       0.       0.       0.   ]
     [-142.968 -142.968 -173.226  142.968  142.968 -162.045]
     [  -0.      -0.      -0.       0.       0.       0.   ]
     [-254.001 -254.001 -173.226  254.001  254.001   -4.   ]
     [  -0.      -0.      -0.       0.       0.       0.   ]
     [  -0.      -0.      -0.       0.       0.       0.   ]
     [-254.001 -254.001   -1.     254.001  254.001  190.101]
     [-254.001 -254.001   -5.     254.001  254.001    0.   ]
     [  -0.      -0.      -0.       0.       0.       0.   ]
     [  -0.      -0.      -0.       0.       0.       0.   ]
     [  -0.      -0.      -0.       0.       0.       0.   ]
     [  -0.      -0.      -0.       0.       0.       0.   ]
     [  -0.      -0.      -0.       0.       0.       0.   ]
     [  -0.      -0.      -0.       0.       0.       0.   ]]
    a.node[15746:15746+15].reshape(-1,16).view(np.int32)[:,6:8] # descNodeBoundaryIndex
    [[27 29]
     [27 30]
     [27 31]
     [27 32]
     [27 33]
     [27 34]
     [27 35]
     [27 36]
     [27 37]
     [27 38]
     [27 39]
     [27 40]
     [27 41]
     [27 42]
     [27 43]]
    a.node[15746:15746+15].reshape(-1,16).view(np.int32)[:,14:16] # descNodeTCTran
    [[   1    0]
     [   1    0]
     [ 108 7218]
     [   1    0]
     [ 103 7219]
     [   0    0]
     [   0    0]
     [ 103 7220]
     [ 105 7221]
     [   0    0]
     [   0    0]
     [   0    0]
     [   0    0]
     [   0    0]
     [   0    0]]
    a.node[15746:15746+15].reshape(-1,16).view(np.int32)[:,14:16] & 0x7ffffff  # descNodeTCTran
    [[   1    0]
     [   1    0]
     [ 108 7218]
     [   1    0]
     [ 103 7219]
     [   0    0]
     [   0    0]
     [ 103 7220]
     [ 105 7221]
     [   0    0]
     [   0    0]
     [   0    0]
     [   0    0]
     [   0    0]
     [   0    0]]

    In [4]: a.tran[7218-1]
    Out[4]:
    array([[   1.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    1.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    1.   ,    0.   ],
           [   0.   ,    0.   , -179.216,    1.   ]], dtype=float32)

    In [5]: a.tran[7219-1]
    Out[5]:
    array([[ 1.337,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  1.337,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  1.   ,  0.   ],
           [ 0.   ,  0.   , -5.   ,  1.   ]], dtype=float32)

    In [6]: a.tran[7220-1]
    Out[6]:
    array([[1.337, 0.   , 0.   , 0.   ],
           [0.   , 1.337, 0.   , 0.   ],
           [0.   , 0.   , 1.   , 0.   ],
           [0.   , 0.   , 0.   , 1.   ]], dtype=float32)

    In [7]: a.tran[7221-1]
    Out[7]:
    array([[ 1. ,  0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ],
           [ 0. ,  0. , -2.5,  1. ]], dtype=float32)

    In [8]:




::

    ]U4Solid::init 107/1/Dis/462
    sn::uncoincide lvid 107 num_prim 4

    sn::getNodeTransformProduct idx 457 reverse 0 num_nds 4
     i 0 j 3 ii.idx 463 jj.idx 457 ixf N jxf Y

     (jxf.t)                                                (jxf.v)                                                (jxf.t)*(jxf.v)                                       
     1.3368     0.0000     0.0000     0.0000                0.7480     -0.0000    0.0000     -0.0000               1.0000     0.0000     0.0000     0.0000    
     0.0000     1.3368     0.0000     0.0000                -0.0000    0.7480     -0.0000    0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     -0.0000    1.0000     -0.0000               0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     0.0000     1.0000                -0.0000    0.0000     -0.0000    1.0000                0.0000     0.0000     0.0000     1.0000    

     i 1 j 2 ii.idx 461 jj.idx 459 ixf N jxf N
     i 2 j 1 ii.idx 459 jj.idx 461 ixf N jxf N
     i 3 j 0 ii.idx 457 jj.idx 463 ixf Y jxf N

     (ixf.t)                                                (ixf.v)                                                (ixf.t)*(ixf.v)                                       
     1.3368     0.0000     0.0000     0.0000                0.7480     -0.0000    0.0000     -0.0000               1.0000     0.0000     0.0000     0.0000    
     0.0000     1.3368     0.0000     0.0000                -0.0000    0.7480     -0.0000    0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     -0.0000    1.0000     -0.0000               0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     0.0000     1.0000                -0.0000    0.0000     -0.0000    1.0000                0.0000     0.0000     0.0000     1.0000    


     tp                                                     vp                                                     tp*vp                                                 
     1.3368     0.0000     0.0000     0.0000                0.7480     0.0000     0.0000     0.0000                1.0000     0.0000     0.0000     0.0000    
     0.0000     1.3368     0.0000     0.0000                0.0000     0.7480     0.0000     0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     0.0000     1.0000     0.0000                0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     0.0000     1.0000                0.0000     0.0000     0.0000     1.0000                0.0000     0.0000     0.0000     1.0000    

     idx  457 lvid 107 zs[-190.001,-190.001,  0.000,190.001,190.001,190.101]

    sn::getNodeTransformProduct idx 458 reverse 0 num_nds 4
     i 0 j 3 ii.idx 463 jj.idx 458 ixf N jxf Y

     (jxf.t)                                                (jxf.v)                                                (jxf.t)*(jxf.v)                                       
     1.0000     0.0000     0.0000     0.0000                1.0000     -0.0000    0.0000     -0.0000               1.0000     0.0000     0.0000     0.0000    
     0.0000     1.0000     0.0000     0.0000                -0.0000    1.0000     -0.0000    0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     -0.0000    1.0000     -0.0000               0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     -2.5000    1.0000                -0.0000    0.0000     2.5000     1.0000                0.0000     0.0000     0.0000     1.0000    

     i 1 j 2 ii.idx 461 jj.idx 459 ixf N jxf N
     i 2 j 1 ii.idx 459 jj.idx 461 ixf N jxf N
     i 3 j 0 ii.idx 458 jj.idx 463 ixf Y jxf N

     (ixf.t)                                                (ixf.v)                                                (ixf.t)*(ixf.v)                                       
     1.0000     0.0000     0.0000     0.0000                1.0000     -0.0000    0.0000     -0.0000               1.0000     0.0000     0.0000     0.0000    
     0.0000     1.0000     0.0000     0.0000                -0.0000    1.0000     -0.0000    0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     -0.0000    1.0000     -0.0000               0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     -2.5000    1.0000                -0.0000    0.0000     2.5000     1.0000                0.0000     0.0000     0.0000     1.0000    


     tp                                                     vp                                                     tp*vp                                                 
     1.0000     0.0000     0.0000     0.0000                1.0000     0.0000     0.0000     0.0000                1.0000     0.0000     0.0000     0.0000    
     0.0000     1.0000     0.0000     0.0000                0.0000     1.0000     0.0000     0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     0.0000     1.0000     0.0000                0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     -2.5000    1.0000                0.0000     0.0000     2.5000     1.0000                0.0000     0.0000     0.0000     1.0000    

     idx  458 lvid 107 cy[-254.001,-254.001, -2.500,254.001,254.001,  2.500]

    sn::getNodeTransformProduct idx 460 reverse 0 num_nds 3
     i 0 j 2 ii.idx 463 jj.idx 460 ixf N jxf Y

     (jxf.t)                                                (jxf.v)                                                (jxf.t)*(jxf.v)                                       
     1.3368     0.0000     0.0000     0.0000                0.7480     -0.0000    0.0000     -0.0000               1.0000     0.0000     0.0000     0.0000    
     0.0000     1.3368     0.0000     0.0000                -0.0000    0.7480     0.0000     0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     -0.0000    1.0000     -0.0000               0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     -5.0000    1.0000                -0.0000    0.0000     5.0000     1.0000                0.0000     0.0000     0.0000     1.0000    

     i 1 j 1 ii.idx 461 jj.idx 461 ixf N jxf N
     i 2 j 0 ii.idx 460 jj.idx 463 ixf Y jxf N

     (ixf.t)                                                (ixf.v)                                                (ixf.t)*(ixf.v)                                       
     1.3368     0.0000     0.0000     0.0000                0.7480     -0.0000    0.0000     -0.0000               1.0000     0.0000     0.0000     0.0000    
     0.0000     1.3368     0.0000     0.0000                -0.0000    0.7480     0.0000     0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     -0.0000    1.0000     -0.0000               0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     -5.0000    1.0000                -0.0000    0.0000     5.0000     1.0000                0.0000     0.0000     0.0000     1.0000    


     tp                                                     vp                                                     tp*vp                                                 
     1.3368     0.0000     0.0000     0.0000                0.7480     0.0000     0.0000     0.0000                1.0000     0.0000     0.0000     0.0000    
     0.0000     1.3368     0.0000     0.0000                0.0000     0.7480     0.0000     0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     0.0000     1.0000     0.0000                0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     -5.0000    1.0000                0.0000     0.0000     5.0000     1.0000                0.0000     0.0000     0.0000     1.0000    

     idx  460 lvid 107 zs[-190.001,-190.001,-168.226,190.001,190.001,  0.000]

    sn::getNodeTransformProduct idx 462 reverse 0 num_nds 2
     i 0 j 1 ii.idx 463 jj.idx 462 ixf N jxf Y

     (jxf.t)                                                (jxf.v)                                                (jxf.t)*(jxf.v)                                       
     1.0000     0.0000     0.0000     0.0000                1.0000     -0.0000    0.0000     -0.0000               1.0000     0.0000     0.0000     0.0000    
     0.0000     1.0000     0.0000     0.0000                -0.0000    1.0000     -0.0000    0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     -0.0000    1.0000     -0.0000               0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     -179.2157  1.0000                -0.0000    0.0000     179.2157   1.0000                0.0000     0.0000     0.0000     1.0000    

     i 1 j 0 ii.idx 462 jj.idx 463 ixf Y jxf N

     (ixf.t)                                                (ixf.v)                                                (ixf.t)*(ixf.v)                                       
     1.0000     0.0000     0.0000     0.0000                1.0000     -0.0000    0.0000     -0.0000               1.0000     0.0000     0.0000     0.0000    
     0.0000     1.0000     0.0000     0.0000                -0.0000    1.0000     -0.0000    0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     -0.0000    1.0000     -0.0000               0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     -179.2157  1.0000                -0.0000    0.0000     179.2157   1.0000                0.0000     0.0000     0.0000     1.0000    


     tp                                                     vp                                                     tp*vp                                                 
     1.0000     0.0000     0.0000     0.0000                1.0000     0.0000     0.0000     0.0000                1.0000     0.0000     0.0000     0.0000    
     0.0000     1.0000     0.0000     0.0000                0.0000     1.0000     0.0000     0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     0.0000     1.0000     0.0000                0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     -179.2157  1.0000                0.0000     0.0000     179.2157   1.0000                0.0000     0.0000     0.0000     1.0000    

     idx  462 lvid 107 co[-142.968,-142.968,  5.990,142.968,142.968, 17.170]

    ]U4Solid::init 107/0/Uni/463




Added sn::uncoincide : currently that forms the bbox and transforms with the CSG tree transform
------------------------------------------------------------------------------------------------

::

    sn::uncoincide sn__uncoincide_dump_lvid 107 (-ve for transform details)  lvid 107 num_prim 4
     idx  457 lvid 107 zs[-254.001,-254.001,  0.000,254.001,254.001,190.101]
     idx  458 lvid 107 cy[-254.001,-254.001, -5.000,254.001,254.001,  0.000]
     idx  460 lvid 107 zs[-254.001,-254.001,-173.226,254.001,254.001, -5.000]
     idx  462 lvid 107 co[-142.968,-142.968,-173.226,142.968,142.968,-162.045]

    ]U4Solid::init 107/0/Uni/463


HMM : where are the combined CSGPrim AABB set in A ? 
------------------------------------------------------

::

     512 CSGPrim* CSG_GGeo_Convert::convertPrim(const GParts* comp, unsigned primIdx )
     513 {
     ...
     560     CSGPrim* prim = foundry->addPrim(numParts, nodeOffset_ );
     561     prim->setMeshIdx(meshIdx);
     562     assert(prim) ;
     563 
     564     AABB bb = {} ;
     565 
     ...
     577     for(unsigned partIdxRel=0 ; partIdxRel < numParts ; partIdxRel++ )
     578     {
     579         CSGNode* n = convertNode(comp, primIdx, partIdxRel);
     580 
     581         if(root == nullptr) root = n ;   // first nodes becomes root 
     582 
     583         if(n->is_zero()) continue;
     584 
     585         bool negated = n->is_complemented_primitive();
     586 
     587         bool bbskip = negated ;
     ...
     600         float* naabb = n->AABB();
     601 
     602         if(!bbskip) bb.include_aabb( naabb );
     603     }
     ...
     652     if( has_xbb == false )
     653     {
     654         const float* bb_data = bb.data();
     655         LOG(debug)
     656             << " has_xbb " << has_xbb
     657             << " (using self defined BB for prim) "
     658             << " AABB \n" << AABB::Desc(bb_data)
     659             ;
     660         prim->setAABB( bb_data );
     661     }
     662     else
     663     {
     664         LOG(LEVEL)
     665             << " has_xbb " << has_xbb
     666             << " (USING EXTERNAL BB for prim, eg from G4VSolid) "
     667             << " xbb \n" << xbb.desc()
     668             ;
     669 
     670         prim->setAABB( xbb.min.x, xbb.min.y, xbb.min.z, xbb.max.x, xbb.max.y, xbb.max.z );
     671     }
     672 






DONE : enabled sn::uncoincide : gets all node param to match 
--------------------------------------------------------------

* initially used non ellipsoid transformed rperp 
  so the znudge upper/lower choice was matching old workflow by accident

* after using correct transform the rperp are equal : so added special casing for CYL-ZSH


::

    In [2]: w = np.where(ab.npa > 1e-2)[0] ;  w
    Out[2]: array([], dtype=int64)

    In [3]: ab.npa.max()
    Out[3]: 2.1e-44

    In [4]: ab.npa.shape
    Out[4]: (15968,)




::

    G4GDML: Reading '/Users/blyth/.opticks/GEOM/V1J010/origin.gdml' done!
    U4GDML::read                   yielded chars :  cout      0 cerr      0 : set VERBOSE to see them 
    sn::increase_zmax_ lvid 95 _zmax  -15.00 dz    1.00 new_zmax  -14.00
    sn::decrease_zmin_ lvid 95 _zmin -101.00 dz    1.00 new_zmin -102.00
    sn::increase_zmax_ lvid 95 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::decrease_zmin_ lvid 95 _zmin  -15.00 dz    1.00 new_zmin  -16.00
    sn::increase_zmax_cone lvid 96 z2 0.00 r2 450.00 dz 1.00 new_z2 1.00 new_r2 451.79
    sn::increase_zmax_ lvid 106 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::decrease_zmin_ lvid 106 _zmin    0.00 dz    1.00 new_zmin   -1.00
    sn::uncoincide sn__uncoincide_dump_lvid 106 lvid 106
    sn::uncoincide_ lvid 106 num_prim 4
    sn::uncoincide_zminmax lvid 106 (1,2)  lower_zmax -5 upper_zmin -5 lower_tag zs upper_tag cy can_znudge YES same_union YES z_minmax_coincide YES fixable_coincide YES enable YES
    sn::uncoincide_zminmax lvid 106 (1,2)  lower_rperp_at_zmax 185 upper_rperp_at_zmin 249
    sn::uncoincide_zminmax lvid 106 (1,2)  !(lower_rperp_at_zmax > upper_rperp_at_zmin) : lower->increase_zmax( dz )   : expand lower up into bigger upper 
    sn::uncoincide_zminmax lvid 106 (2,3)  lower_zmax 0 upper_zmin 0 lower_tag cy upper_tag zs can_znudge YES same_union YES z_minmax_coincide YES fixable_coincide YES enable YES
    sn::uncoincide_zminmax lvid 106 (2,3)  lower_rperp_at_zmax 249 upper_rperp_at_zmin 185
    sn::uncoincide_zminmax lvid 106 (2,3)  lower_rperp_at_zmax > upper_rperp_at_zmin : upper->decrease_zmin( dz )   : expand upper down into bigger lower 
    sn::uncoincide_ lvid 106 num_prim 4 coincide 2

    sn::postconvert lvid 106 coincide 2
     desc_prim_all before update AABB sn::DescPrim num_nd 4 DESC ORDER REVERSED 
     idx  450 lvid 106 zs[-249.000,-249.000,  0.000,249.000,249.000,185.100]
     idx  451 lvid 106 cy[-249.000,-249.000, -5.000,249.000,249.000,  0.000]
     idx  453 lvid 106 zs[-249.000,-249.000,-168.225,249.000,249.000, -5.000]
     idx  455 lvid 106 co[-139.625,-139.625,-168.225,139.625,139.625,-158.178]

    sn::postconvert lvid 106 coincide 2
     desc_prim_all after update AABB sn::DescPrim num_nd 4 DESC ORDER REVERSED 
     idx  450 lvid 106 zs[-249.000,-249.000, -1.000,249.000,249.000,185.100]
     idx  451 lvid 106 cy[-249.000,-249.000, -5.000,249.000,249.000,  0.000]
     idx  453 lvid 106 zs[-249.000,-249.000,-168.225,249.000,249.000, -4.000]
     idx  455 lvid 106 co[-139.625,-139.625,-168.225,139.625,139.625,-158.178]

    sn::increase_zmax_ lvid 107 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::decrease_zmin_ lvid 107 _zmin    0.00 dz    1.00 new_zmin   -1.00
    sn::postconvert lvid 107 coincide 2
     desc_prim_all before update AABB sn::DescPrim num_nd 4 DESC ORDER REVERSED 
     idx  457 lvid 107 zs[-254.001,-254.001,  0.000,254.001,254.001,190.101]
     idx  458 lvid 107 cy[-254.001,-254.001, -5.000,254.001,254.001,  0.000]
     idx  460 lvid 107 zs[-254.001,-254.001,-173.226,254.001,254.001, -5.000]
     idx  462 lvid 107 co[-142.968,-142.968,-173.226,142.968,142.968,-162.045]

    sn::postconvert lvid 107 coincide 2
     desc_prim_all after update AABB sn::DescPrim num_nd 4 DESC ORDER REVERSED 
     idx  457 lvid 107 zs[-254.001,-254.001, -1.000,254.001,254.001,190.101]
     idx  458 lvid 107 cy[-254.001,-254.001, -5.000,254.001,254.001,  0.000]
     idx  460 lvid 107 zs[-254.001,-254.001,-173.226,254.001,254.001, -4.000]
     idx  462 lvid 107 co[-142.968,-142.968,-173.226,142.968,142.968,-162.045]

    sn::increase_zmax_ lvid 108 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::increase_zmax_ lvid 108 _zmax  100.00 dz    1.00 new_zmax  101.00
    sn::increase_zmax_ lvid 117 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::increase_zmax_ lvid 117 _zmax   97.00 dz    1.00 new_zmax   98.00
    U4Tree::init U4Tree::desc
     st Y


::

    U4GDML::read                   yielded chars :  cout      0 cerr      0 : set VERBOSE to see them 
    sn::increase_zmax_ lvid 95 _zmax  -15.00 dz    1.00 new_zmax  -14.00
    sn::decrease_zmin_ lvid 95 _zmin -101.00 dz    1.00 new_zmin -102.00
    sn::increase_zmax_ lvid 95 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::decrease_zmin_ lvid 95 _zmin  -15.00 dz    1.00 new_zmin  -16.00
    sn::increase_zmax_cone lvid 96 z2 0.00 r2 450.00 dz 1.00 new_z2 1.00 new_r2 451.79
    sn::increase_zmax_ lvid 106 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::increase_zmax_ lvid 106 _zmax    2.50 dz    1.00 new_zmax    3.50
    sn::uncoincide sn__uncoincide_dump_lvid 106 lvid 106
    sn::uncoincide_ lvid 106 num_prim 4
    sn::uncoincide_zminmax lvid 106 (1,2)  lower_zmax -5 upper_zmin -5 lower_tag zs upper_tag cy can_znudge YES same_union YES z_minmax_coincide YES fixable_coincide YES enable YES
    sn::uncoincide_zminmax lvid 106 (1,2)  lower_rperp_at_zmax 185 upper_rperp_at_zmin 249 (leaf frame) 
     upper_pos    249.0000   249.0000    -5.0000 (tree frame) 
     lower_pos    249.0000   249.0000    -5.0000 (tree frame) 
     upper_rperp_smaller NO 
     !upper_rperp_smaller : lower->increase_zmax( dz ) : expand lower up into bigger upper 
    sn::uncoincide_zminmax lvid 106 (2,3)  lower_zmax 0 upper_zmin 0 lower_tag cy upper_tag zs can_znudge YES same_union YES z_minmax_coincide YES fixable_coincide YES enable YES
    sn::uncoincide_zminmax lvid 106 (2,3)  lower_rperp_at_zmax 249 upper_rperp_at_zmin 185 (leaf frame) 
     upper_pos    249.0000   249.0000     0.0000 (tree frame) 
     lower_pos    249.0000   249.0000     0.0000 (tree frame) 
     upper_rperp_smaller NO 
     !upper_rperp_smaller : lower->increase_zmax( dz ) : expand lower up into bigger upper 
    sn::uncoincide_ lvid 106 num_prim 4 coincide 2

    sn::postconvert lvid 106 coincide 2




DONE : Whats up with node bbox ? JUST SMALL DIFFS LESS THAN 1e-2 ON BIG VALUES float/double level diffs
-----------------------------------------------------------------------------------------------------------

::

    In [5]: a.node[:,2:].reshape(-1,8)[:,:6]
    Out[5]: 
    array([[-60000.  , -60000.  , -60000.  ,  60000.  ,  60000.  ,  60000.  ],
           [-28000.  , -27500.  ,  21750.  ,  34250.  ,  27500.  ,  51750.  ],
           [-28000.  , -27500.  ,  32750.  ,  34250.  ,  27500.  ,  51750.  ],
           [    -0.  ,     -0.  ,     -0.  ,      0.  ,      0.  ,      0.  ],
           [-28000.  , -29760.  ,  -7770.  ,  34250.  ,  29760.  ,  51750.  ],
           [-29000.  , -59520.  , -48290.  ,  35250.  ,  59520.  ,  32750.  ],
           [    -0.  ,     -0.  ,     -0.  ,      0.  ,      0.  ,      0.  ],
           [-25000.  , -26760.  ,  -4770.  ,  31250.  ,  26760.  ,  48750.  ],
           [-28000.  , -53520.  , -42290.  ,  34250.  ,  53520.  ,  32750.  ],
           [-28000.  , -27500.  ,  21750.  ,  34250.  ,  27500.  ,  32750.  ],
           ...,
           [ -3430.  ,    712.85,     -5.15,   3430.  ,    739.15,      5.15],
           [ -3430.  ,    713.  ,     -5.  ,   3430.  ,    739.  ,      5.  ],
           [ -3430.  ,    739.25,     -5.15,   3430.  ,    765.55,      5.15],
           [ -3430.  ,    739.4 ,     -5.  ,   3430.  ,    765.4 ,      5.  ],
           [ -3430.  ,    765.65,     -5.15,   3430.  ,    791.95,      5.15],
           [ -3430.  ,    765.8 ,     -5.  ,   3430.  ,    791.8 ,      5.  ],
           [ -3430.  ,    792.05,     -5.15,   3430.  ,    818.35,      5.15],
           [ -3430.  ,    792.2 ,     -5.  ,   3430.  ,    818.2 ,      5.  ],
           [ -3430.  ,    818.45,     -5.15,   3430.  ,    844.75,      5.15],
           [ -3430.  ,    818.6 ,     -5.  ,   3430.  ,    844.6 ,      5.  ]], dtype=float32)

    In [6]: b.node[:,2:].reshape(-1,8)[:,:6]
    Out[6]: 
    array([[-60000.  , -60000.  , -60000.  ,  60000.  ,  60000.  ,  60000.  ],
           [-28000.  , -27500.  ,  21750.  ,  34250.  ,  27500.  ,  51750.  ],
           [-28000.  , -27500.  ,  32750.  ,  34250.  ,  27500.  ,  51750.  ],
           [    -0.  ,     -0.  ,     -0.  ,      0.  ,      0.  ,      0.  ],
           [-28000.  , -29760.  ,  -7770.  ,  34250.  ,  29760.  ,  51750.  ],
           [-29000.  , -59520.  , -48290.  ,  35250.  ,  59520.  ,  32750.  ],
           [    -0.  ,     -0.  ,     -0.  ,      0.  ,      0.  ,      0.  ],
           [-25000.  , -26760.  ,  -4770.  ,  31250.  ,  26760.  ,  48750.  ],
           [-28000.  , -53520.  , -42290.  ,  34250.  ,  53520.  ,  32750.  ],
           [-28000.  , -27500.  ,  21750.  ,  34250.  ,  27500.  ,  32750.  ],
           ...,
           [ -3430.  ,    712.85,     -5.15,   3430.  ,    739.15,      5.15],
           [ -3430.  ,    713.  ,     -5.  ,   3430.  ,    739.  ,      5.  ],
           [ -3430.  ,    739.25,     -5.15,   3430.  ,    765.55,      5.15],
           [ -3430.  ,    739.4 ,     -5.  ,   3430.  ,    765.4 ,      5.  ],
           [ -3430.  ,    765.65,     -5.15,   3430.  ,    791.95,      5.15],
           [ -3430.  ,    765.8 ,     -5.  ,   3430.  ,    791.8 ,      5.  ],
           [ -3430.  ,    792.05,     -5.15,   3430.  ,    818.35,      5.15],
           [ -3430.  ,    792.2 ,     -5.  ,   3430.  ,    818.2 ,      5.  ],
           [ -3430.  ,    818.45,     -5.15,   3430.  ,    844.75,      5.15],
           [ -3430.  ,    818.6 ,     -5.  ,   3430.  ,    844.6 ,      5.  ]], dtype=float32)

    In [7]:                                                         

Small diffs node bb only::

    In [16]: w_bb = np.where( ab.nbb > 1e-2 ) ; w_bb
    Out[16]: (array([], dtype=int64),)




DONE  : Bring the CSGPrim bbox into B side ? Where is it done A side ?
------------------------------------------------------------------------

* done in `CSGImport::importPrim_` starting from the below::

     512 CSGPrim* CSG_GGeo_Convert::convertPrim(const GParts* comp, unsigned primIdx )
     513 {
     ...
     575     CSGNode* root = nullptr ;
     576 
     577     for(unsigned partIdxRel=0 ; partIdxRel < numParts ; partIdxRel++ )
     578     {   
     579         CSGNode* n = convertNode(comp, primIdx, partIdxRel);
     580         
     581         if(root == nullptr) root = n ;   // first nodes becomes root 
     582         
     583         if(n->is_zero()) continue;
     584         
     585         bool negated = n->is_complemented_primitive();
     586         
     587         bool bbskip = negated ;
     ... 
     600         float* naabb = n->AABB();
     601         
     602         if(!bbskip) bb.include_aabb( naabb );
     603     }


HMM: I think there is a bug in the above, it is including the 
all zero bbox from the operator nodes. 
Which means that some bbox are artificially enlarged to include the origin ? 


DONE  : Fixing s_bb::IncludeAABB to igore all zero other_aabb gets pbb close
-------------------------------------------------------------------------------

DIFFERENCE WITH TAIL AABB::

    w_pbb2 = np.where( ab.pbb > 1e-2 )[0]  ##
    w_pbb2.shape
    (2,)

    In [1]: w_pbb2
    Out[1]: array([2930, 2939])

    In [2]: a.pbb[2930]
    Out[2]: array([-264.   , -264.   , -183.225,  264.   ,  264.   ,    0.   ], dtype=float32)

    In [3]: b.pbb[2930]
    Out[3]: array([-264.   , -264.   , -183.225,  264.   ,  264.   ,  -39.   ], dtype=float32)

    In [4]: a.pbb[2939]
    Out[4]: array([-264.   , -264.   , -183.225,  264.   ,  264.   ,    0.   ], dtype=float32)

    In [5]: b.pbb[2939]
    Out[5]: array([-264.   , -264.   , -183.225,  264.   ,  264.   ,  -40.   ], dtype=float32)


    In [7]: b.plv[w_pbb2]
    Out[7]: array([110,  98], dtype=int32)

    In [8]: a.meshname[a.plv[w_pbb2]]
    Out[8]: array(['NNVTMCPPMTTail0x6176550', 'HamamatsuR12860Tail0x61673d0'], dtype=object)


Looking at the detailed dump, its clear that B is correct with the smaller AABB::

    In [11]: b.descLVDetail(110)    


* https://simoncblyth.bitbucket.io/env/presentation/opticks_20221117_mask_debug_and_tmm.html

Simtrace plots of the Tail in above,  can see that it is the fancy bowl shape at back of PMT.
The bbox clearly should extend up to -39. (not up to 0.). 


DONE : CONFIRMED and FIXED A SIDE saabb.h BUG 
-------------------------------------------------

Added the AllZero check to AABB::include_aabb. 
Maybe that will get A and B to match ? IT DOES::

    138 AABB_METHOD bool AABB::AllZero(const float* aabb) // static
    139 {   
    140     int count = 0 ; 
    141     for(int i=0 ; i < 6 ; i++) if(std::abs(aabb[i]) == 0.f) count += 1 ;
    142     return count == 6 ;
    143 }
    144 
    145 AABB_METHOD void AABB::include_aabb(const float* aabb)
    146 {
    147     if(AllZero(aabb)) return ;
    148 



::

    w_pbb3 = np.where( ab.pbb > 1e-3 )[0]  ##
    w_pbb3.shape
    (903,)
    w_pbb2 = np.where( ab.pbb > 1e-2 )[0]  ##
    w_pbb2.shape
    (0,)
    ----------------------------------------------------------------------------------------------------

    In [1]: a.pbb[2930]
    Out[1]: array([-264.   , -264.   , -183.225,  264.   ,  264.   ,  -39.   ], dtype=float32)

    In [2]: b.pbb[2930]
    Out[2]: array([-264.   , -264.   , -183.225,  264.   ,  264.   ,  -39.   ], dtype=float32)

    In [3]:


FIXED : CSGSolid B SIDE MISSES center_extent : THEY NOW MATCH
-----------------------------------------------------------------

::

    In [7]: a.solid[1]
    Out[7]:
    array([[      12658,           0,           0,           0],
           [          5,        2923,           0,           0],
           [          0,           0, -1047560320,  1114095814]], dtype=int32)

    In [8]: b.solid[1]
    Out[8]:
    array([[12658,     0,     0,     0],
           [    5,  2923,     0,     0],
           [    0,     0,     0,     0]], dtype=int32)


Missing center_extent from B::

    In [10]: a.solid.view(np.float32)[:,2]
    Out[10]:
    array([[    0.   ,     0.   ,     0.   , 60000.   ],
           [    0.   ,     0.   ,   -17.937,    57.938],
           [    0.   ,     0.   ,     5.438,   264.05 ],
           [    0.   ,     0.   ,     8.438,   264.05 ],
           [    0.   ,     0.   ,    84.525,   264.05 ],
           [    0.   ,     0.   ,     0.   ,    50.   ],
           [    0.   ,     0.   ,   -50.5  ,   195.   ],
           [    0.   ,     0.   ,   -67.15 ,   451.786],
           [    0.   ,     0.   ,     0.   ,  3430.6  ]], dtype=float32)

    In [11]: b.solid.view(np.float32)[:,2]
    Out[11]:
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], dtype=float32)




FIXED : last 299 nodes had node index deviation : looks like local prim vs global index difference
---------------------------------------------------------------------------------------------------

* NOPE : NOT SO SIMPLE : THE NIX IS THE PARTIDX WITHIN THE COMPOUND SOLID 
  (THAT IS KINDA HISTORICAL FROM THE OLD GGEO GMERGEDMESH SPLITS)

* FIXED THIS BY AUTOMATING THE SETTING OF THE CSGNode index in CSGFoundry::addNode


::

    w_nix = np.where(a.nix != b.nix)[0] ## 
    w_nix.shape
    (299,)

    In [5]: np.c_[a.nix[w_nix],b.nix[w_nix]]
    Out[5]:
    array([[    0, 15669],
           [    1, 15670],
           [    2, 15671],
           [    3, 15672],
           [    4, 15673],
           [    5, 15674],
           [    6, 15675],
           [    0, 15676],


NIX IS THE PARTIDX WITHIN THE COMPOUND SOLID, NOT WITHIN THE PRIM OR ABSOLUTE::

    In [4]: a.nix
    Out[4]: array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9, ..., 120, 121, 122, 123, 124, 125, 126, 127, 128, 129], dtype=int32)

    In [5]: np.where(a.nix == 0 )
    Out[5]: (array([    0, 15669, 15676, 15717, 15795, 15823, 15824, 15831, 15838]),)

    In [6]: np.where(a.nix == 1 )
    Out[6]: (array([    1, 15670, 15677, 15718, 15796, 15825, 15832, 15839]),)

    In [7]: np.where(a.nix == 2 )
    Out[7]: (array([    2, 15671, 15678, 15719, 15797, 15826, 15833, 15840]),)

    In [8]: np.where(a.nix == 3 )
    Out[8]: (array([    3, 15672, 15679, 15720, 15798, 15827, 15834, 15841]),)


::

    In [7]: a.spo        # prim offsets for each solid 
    Out[7]: array([   0, 2923, 2928, 2937, 2949, 2955, 2956, 2957, 2958], dtype=int32)

    In [8]: a.pno
    Out[8]: array([    0,     1,     2,     3,     6,     9,    10,    13,    16,    17, ..., 15958, 15959, 15960, 15961, 15962, 15963, 15964, 15965, 15966, 15967], dtype=int32)

    In [9]: a.pno.shape
    Out[9]: (3088,)

    In [10]: a.pno[a.spo]   # node offsets at the start of each solid 
    Out[10]: array([    0, 15669, 15676, 15717, 15795, 15823, 15824, 15831, 15838], dtype=int32)





::

    In [12]: a.nix[:15669]
    Out[12]: array([    0,     1,     2,     3,     4,     5,     6,     7,     8,     9, ..., 15659, 15660, 15661, 15662, 15663, 15664, 15665, 15666, 15667, 15668], dtype=int32)

    In [13]: a.nix[15669:15676]
    Out[13]: array([0, 1, 2, 3, 4, 5, 6], dtype=int32)

    In [14]: a.nix[15676:15717]
    Out[14]: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], dtype=int32)

    In [15]: a.nix[15717:15795]
    Out[15]:
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
           48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77], dtype=int32)

    In [16]: a.nix[15795:15823]
    Out[16]: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype=int32)

    In [17]: a.nix[15823:15824]
    Out[17]: array([0], dtype=int32)

    In [18]: a.nix[15824:15831]
    Out[18]: array([0, 1, 2, 3, 4, 5, 6], dtype=int32)

    In [19]: a.nix[15831:15838]
    Out[19]: array([0, 1, 2, 3, 4, 5, 6], dtype=int32)

    In [20]: a.nix[15838:]
    Out[20]:
    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,
            38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
            76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
           114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129], dtype=int32)

    In [21]:




::

    In [1]: a.nix
    Out[1]: array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9, ..., 120, 121, 122, 123, 124, 125, 126, 127, 128, 129], dtype=int32)

    In [2]: b.nix
    Out[2]: array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9, ..., 120, 121, 122, 123, 124, 125, 126, 127, 128, 129], dtype=int32)

    In [3]: np.all( a.nix == b.nix )
    Out[3]: True

    In [4]: np.where( a.nix == 0 )
    Out[4]: (array([    0, 15669, 15676, 15717, 15795, 15823, 15824, 15831, 15838]),)

    In [5]: np.where( b.nix == 0 )
    Out[5]: (array([    0, 15669, 15676, 15717, 15795, 15823, 15824, 15831, 15838]),)



WIP : A side has very tiny param 0 value for typecode 1/2
------------------------------------------------------------

Looks like a stray int32 in param for old workflow causing 1e-44 deviation between old/new. 

Try::

    681 CSGNode CSGNode::Make(unsigned typecode, const float* param6, const float* aabb ) // static
    682 {
    683     CSGNode nd = {} ;
    684     nd.setTypecode(typecode) ;
    685 
    686     // try avoiding CSG_UNION CSG_INTERSECT getting some stray int32 in param6[0]
    687     // by only setting param, aabb for primitives
    688     if(CSG::IsPrimitive(typecode))
    689     {
    690         if(param6) nd.setParam( param6 );
    691         if(aabb)   nd.setAABB( aabb );
    692     }
    693 
    694     return nd ;
    695 }







::

    In [25]: a.npa[w_npa]*1e44
    Out[25]: array([0.42 , 0.42 , 0.42 , 0.42 , 0.42 , 0.42 , 0.42 , 0.981, 0.981, 0.981, ..., 0.42 , 0.42 , 0.42 , 0.981, 0.42 , 0.42 , 0.981, 0.981, 0.981, 0.981])

    In [26]: b.npa[w_npa]*1e44
    Out[26]: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ..., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    In [27]: a.npa.shape
    Out[27]: (15968, 6)

    In [28]: a.ntc
    Out[28]: array([110, 110, 110,   2, 105, 110,   2, 105, 110, 110, ..., 110, 110, 110, 110, 110, 110, 110, 110, 110, 110], dtype=int32)


    In [30]: a.ntc[w_npa[0]]
    Out[30]: array([2, 2, 2, 2, 2, 2, 2, 1, 1, 1, ..., 2, 2, 2, 2, 1, 1, 2, 2, 2, 2], dtype=int32)

    In [31]: np.unique( a.ntc[w_npa[0]], return_counts=True )
    Out[31]: (array([1, 2], dtype=int32), array([2131,   27]))

    In [32]: np.c_[np.unique( a.ntc[w_npa[0]], return_counts=True )]
    Out[32]:
    array([[   1, 2131],
           [   2,   27]])


    In [30]: a.ntc[w_npa[0]]
    Out[30]: array([2, 2, 2, 2, 2, 2, 2, 1, 1, 1, ..., 2, 2, 2, 2, 1, 1, 2, 2, 2, 2], dtype=int32)

    In [31]: np.unique( a.ntc[w_npa[0]], return_counts=True )
    Out[31]: (array([1, 2], dtype=int32), array([2131,   27]))

    In [32]: np.c_[np.unique( a.ntc[w_npa[0]], return_counts=True )]
    Out[32]:
    array([[   1, 2131],
           [   2,   27]])

    In [33]: a.npa[w_npa]
    Out[33]: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ..., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

    In [34]: a.npa[w_npa].view(np.int32)
    Out[34]: array([3, 3, 3, 3, 3, 3, 3, 7, 7, 7, ..., 3, 3, 3, 7, 3, 3, 7, 7, 7, 7], dtype=int32)

    In [35]: np.unique( a.npa[w_npa].view(np.int32), return_counts=True )
    Out[35]: (array([ 3,  7, 15], dtype=int32), array([  25, 2129,    4]))

    In [36]: np.c_[np.unique( a.npa[w_npa].view(np.int32), return_counts=True )]
    Out[36]:
    array([[   3,   25],
           [   7, 2129],
           [  15,    4]])

    In [37]:



    In [38]: a.npa[a.ntc==1,0]
    Out[38]: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ..., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

    In [39]: a.npa[a.ntc==1,0]*1e44
    Out[39]: array([0.981, 0.   , 0.981, 0.   , 0.981, 0.   , 0.981, 0.   , 0.981, 0.   , ..., 2.102, 0.   , 0.   , 0.   , 0.42 , 0.42 , 0.   , 0.   , 0.   , 0.   ])

    In [40]: b.npa[b.ntc==1,0]*1e44
    Out[40]: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ..., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    In [41]: b.npa[b.ntc==1,0].view(np.int32)
    Out[41]: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)

    In [42]: a.npa[a.ntc==1,0].view(np.int32)
    Out[42]: array([ 7,  0,  7,  0,  7,  0,  7,  0,  7,  0, ..., 15,  0,  0,  0,  3,  3,  0,  0,  0,  0], dtype=int32)

    In [43]: a.npa[a.ntc==2,0].view(np.int32)
    Out[43]: array([ 3,  3,  3,  3,  3,  3,  3,  3,  3,  7,  0, 15,  0,  0,  3,  3,  3,  7,  0, 15,  0,  0,  3,  3,  3,  3,  3,  3,  7,  0,  7,  7,  7,  0,  7], dtype=int32)



HMM : THOSE STRAYS MAY ACTUALLY  BE subNum ?
------------------------------------------------

::


    203     // used for compound node types such as CSG_CONTIGUOUS, CSG_DISCONTIGUOUS 
            // and the rootnode of boolean trees CSG_UNION/CSG_INTERSECTION/CSG_DIFFERENCE    ...
    204     NODE_METHOD unsigned subNum()        const { return q0.u.x ; }
    205     NODE_METHOD unsigned subOffset()     const { return q0.u.y ; }
    206 
    207     NODE_METHOD void setSubNum(unsigned num){    q0.u.x = num ; }
    208     NODE_METHOD void setSubOffset(unsigned num){ q0.u.y = num ; }


::

    epsilon:tests blyth$ opticks-f setSubNum 
    ./CSG/CSGNode.cc:        nd.setSubNum(num_sub); 
    ./CSG/CSGNode.cc:        nd.setSubNum(num_sub); 
    ./CSG/CSGNode.h:    NODE_METHOD void setSubNum(unsigned num){    q0.u.x = num ; }
    ./CSG_GGeo/CSG_GGeo_Convert.cc:        root->setSubNum( root_subNum ); 
    ./npy/NNode.cpp:void nnode::setSubNum(unsigned num) 
    ./npy/NNode.cpp:        root->setSubNum( tree_nodes ); 
    ./npy/NPart.cpp:void npart::setSubNum(unsigned sub_num )
    ./npy/NMultiUnion.cpp:    n->setSubNum(sub_num); 
    ./npy/NPart.hpp:    void setSubNum(unsigned sub_num) ; 
    ./npy/NNode.hpp:    void     setSubNum(unsigned sub_num) ; 
    epsilon:opticks blyth$ 



CSG_GGeo_Convert::convertPrim::

     527     int root_typecode  = comp->getTypeCode(root_partIdx) ;
     528     int root_subNum    = comp->getSubNum(root_partIdx) ;
     529     int root_subOffset = comp->getSubOffset(root_partIdx) ;
     530     bool root_is_compound = CSG::IsCompound((int)root_typecode);

     // 314     static bool IsCompound(int type){      return  type < CSG_LEAF && type > CSG_ZERO ; }

     623     if(root_is_compound) // tc > CSG_ZERO && tc < CSG_LEAF
     624     {
     625         assert( numParts > 1 );
     626         bool tree = int(root_subNum) == int(numParts) ;
     627 
     628         if( tree == false )
     629         {
     630            LOG(error)
     631                << " non-tree nodes detected, eg with list-nodes "
     632                << " root_subNum " << root_subNum
     633                << " root_subOffset " << root_subOffset
     634                << " numParts " << numParts
     635                ;
     636         }
     637 
     638         root->setSubNum( root_subNum );
     639         root->setSubOffset( root_subOffset );
     640     }
     641     else
     642     {
     643         assert( numParts == 1 );
     644         assert( root_subNum == -1 );
     645         assert( root_subOffset == -1 );
     646     }


::

    In [27]: np.c_[np.unique(a.npa[a.pno[a.pnn>1],0].view(np.int32),return_counts=True)]
    Out[27]: 
    array([[   3,   25],
           [   7, 2129],
           [  15,    4]])






     22 typedef enum {
     23     CSG_ZERO=0,
     24     CSG_OFFSET_LIST=4,
     25     CSG_OFFSET_LEAF=7,
     26 
     27     CSG_TREE=1,
     28         CSG_UNION=1,
     29         CSG_INTERSECTION=2,
     30         CSG_DIFFERENCE=3,
     31 
     32     CSG_NODE=11,
     33     CSG_LIST=11,
     34         CSG_CONTIGUOUS=11,
     35         CSG_DISCONTIGUOUS=12,
     36         CSG_OVERLAP=13,
     37 
     38     CSG_LEAF=101,
     39         CSG_SPHERE=101,
     40         CSG_BOX=102,
     41         CSG_ZSPHERE=103,




    173 /**
    174 CSGNode::BooleanOperator
    175 -------------------------
    176 
    177 * num_sub is normally -1, for standard boolean trees
    178 * num_sub > 0 is used for compound "list" nodes : a more efficient approach 
    179   avoid tree overheads used for some complex solids 
    180 
    181 **/
    182 
    183 CSGNode CSGNode::BooleanOperator(unsigned op, int num_sub)   // static 
    184 {
    185     assert( CSG::IsOperator((OpticksCSG_t)op) );
    186     CSGNode nd = {} ;
    187     nd.setTypecode(op) ;
    188     if( num_sub > 0 )
    189     {
    190         nd.setSubNum(num_sub);
    191     }
    192     return nd ;
    193 }



::

     60 
     61     +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
     62     | q  |      x         |      y         |     z          |      w         |  notes                                          |
     63     +====+================+================+================+================+=================================================+
     64     |    | sp/zs/cy:cen_x | sp/zs/cy:cen_y | sp/zs/cy:cen_z | sp/zs/cy:radius|  eliminate center? as can be done by transform  |
     65     | q0 | cn:r1          | cn:z1          | cn:r2          | cn:z2          |  cn:z2 > z1                                     |
     66     |    | hy:r0 z=0 waist| hy:zf          | hy:z1          | hy:z2          |  hy:z2 > z1                                     |
     67     |    | b3:fx          | b3:fy          | b3:fz          |                |  b3: fullside dimensions, center always origin  |
     68     |    | pl/sl:nx       | pl/sl:ny       | pl/sl:nz       | pl:d           |  pl: NB Node plane distinct from plane array    |
     69     |    |                |                | ds:inner_r     | ds:radius      |                                                 |
     70     |    | co:subNum      | co:subOffset   |                | radius()       |                                                 |
     71     |    | cx:planeIdx    | cx:planeNum    |                |                |                                                 |
     72     +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
     73     |    | zs:zdelta_0    | zs:zdelta_1    | boundary       | index          |                                                 |
     74     |    | sl:a           | sl:b           |  (1,2)         | (within solid) |  sl:a,b offsets from origin                     |
     75     | q1 | cy:z1          | cy:z2          |                | (1,3)          |  cy:z2 > z1                                     |
     76     |    | ds:z1          | ds:z2          |                |                |                                                 |
     77     |    | z1()           | z2()           |                |                |                                                 |
     78     +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
     79     |    |                |                |                |                |  q2.w was previously typecode                   |
     80     |    |                |                |                |                |                                                 |
     81     | q2 |  BBMin_x       |  BBMin_y       |  BBMin_z       |  BBMax_x       |                                                 |
     82     |    |                |                |                |                |                                                 |
     83     |    |                |                |                |                |                                                 |
     84     +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
     85     |    |                |                |  typecode      | gtransformIdx  |  a.node[:,3,3].view(np.int32) & 0x7fffffff      |
     86     |    |                |                |  (3,2)         | complement     |                                                 |
     87     | q3 |  BBMax_y       |  BBMax_z       |                | (3,3)          |                                                 |
     88     |    |                |                |                |                |                                                 |
     89     |    |                |                |                |                |                                                 |
     90     |    |                |                |                |                |                                                 |
     91     |    |                |                |                |                |                                                 |
     92     +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
     93 





TODO : REMAINING DEVIATIONS TO CHASE : subNum on compound root nodes AND boundary index
-----------------------------------------------------------------------------------------


~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh ana::

    w_solid = np.where( a.solid != b.solid )[0]  ##
    w_solid.shape
    (0,)

    np.all( a.nbd == b.nbd )  # node boundary     : THIS NEEDS OSUR IMPLEMENTED IN OLD WORKFLOW TO MATCH NEW
    False                                         : TODO CHECK THAT THIS MATCHES WHEN DISABLE OSUR IN NEW WORKFLOW  

    np.all( a.nix == b.nix )  # nodeIdx local to the compound solid 
    True
    w_nix = np.where(a.nix != b.nix)[0] ## 
    w_nix.shape
    (0,)
    w_npa3 = np.where( np.abs(a.npa - b.npa) > 1e-3 )[0] ##  node param deviations
    w_npa3.shape
    (0,)
    w_nbb3 = np.where( np.abs(a.nbb - b.nbb) > 3e-2 )[0]  ## node bbox deviations
    w_nbb3.shape
    (0,)
    w_nbb2 = np.where( np.abs(a.nbb - b.nbb) > 1e-2 )[0]  ## node bbox deviations
    w_nbb2.shape
    (0,)
    np.all( a.ntc == b.ntc )  # node typecode
    True
    np.all( a.ncm == b.ncm )  # node complement 
    True
    np.all( a.ntr == b.ntr )  # node transform idx + 1 
    True
    np.all( a.pnn == b.pnn )  # prim numNode
    True
    np.all( a.pno == b.pno )  # prim nodeOffset
    True
    np.all( a.pto == b.pto )  # prim tranOffset
    True
    np.all( a.ppo == b.ppo )  # prim planOffset
    True
    np.all( a.psb == b.psb )  # prim sbtIndexOffset
    True
    np.all( a.plv == b.plv )  # prim lvid/meshIdx
    True
    np.all( a.prx == b.prx )  # prim ridx/repeatIdx
    True
    np.all( a.pix == b.pix )  # primIdx 
    True
    w_pbb3 = np.where( np.abs(a.pbb-b.pbb) > 1e-3 )[0]  ## prim bbox deviations
    w_pbb3.shape
    (1099,)
    w_pbb2 = np.where( np.abs(a.pbb-b.pbb) > 1e-2 )[0]  ## prim bbox deviations
    w_pbb2.shape
    (0,)
    np.all( a.snp == b.snp )  # solid numPrim 
    True
    np.all( a.spo == b.spo )  # solid primOffset
    True
    np.all( a.sce == b.sce )  # solid center_extent 
    True
    w_sce = np.where( np.abs( a.sce - b.sce ) > 1e-3 )[0]    # solid center_extent
    array([], dtype=int64)
    w_sce.shape 
    (0,)

    w_npa = np.where( a.npa != b.npa )[0]   ## int32 3,7,15 in first param slot 
    w_npa.shape   # these are subNum on compound root nodes : and new workflow omits it
    (2158,)
    tab = np.c_[np.unique(a.npa[a.pno[a.pnn>1],0].view(np.int32),return_counts=True)] ## subNum picked from node param 0 of compound root nodes
    tab 
    array([[   3,   25],
           [   7, 2129],
           [  15,    4]])

    In [1]:                                                                           



DONE : added subNum/subOffset to compound root nodes
------------------------------------------------------

::

    275     if(CSG::IsCompound(root->typecode()))
    276     {
    277         assert( numParts > 1 );
    278         root->setSubNum( numParts );
    279         root->setSubOffset( 0 );
    280         // THESE NEED REVISIT WHEN ADDING list-nodes SUPPORT
    281     }
    282 


WIP : boundary is only remaining deviant
-------------------------------------------

U4Tree.h the implicits are enabled by default::

     108     // disable the below with settings with by defining the below envvar
     109     static constexpr const char* __DISABLE_OSUR_IMPLICIT = "U4Tree__DISABLE_OSUR_IMPLICIT" ;
     110     static constexpr const char* __DISABLE_ISUR_IMPLICIT = "U4Tree__DISABLE_ISUR_IMPLICIT" ;
     111     bool                                        enable_osur ;
     112     bool                                        enable_isur ;
     113 


Confirmed that the below gets A/B match::

   export U4Tree__DISABLE_OSUR_IMPLICIT=1 

::

    np.all( a.nbd == b.nbd )  # node boundary 
    True

Of course cannot leave like that as need the OSUR implicits for Geant4 matching::

    In [12]: len( A.SSim.stree.standard.bnd_names )
    Out[12]: 124

    In [19]: len( A.SSim.extra.GGeo.bnd_names )
    Out[19]: 43

    In [13]: len( B.SSim.stree.standard.bnd_names )
    Out[13]: 43



TODO : BND NAME DIFF, COULD DUPLICATE SKIPPING POINTLESS ISUR IN A TO MATCH THIS ?
------------------------------------------------------------------------------------

B side skips pointless ISUR from NoRINDEX absorbers. There will be no photons
coming out of the metal so the ISUR are pointless. 

::

    In [3]: an = A.SSim.extra.GGeo.bnd_names

    In [4]: bn = B.SSim.stree.standard.bnd_names

    In [5]: w_bnd_names = np.where( an != bn )[0]

    In [6]: w_bnd_names
    Out[6]: array([21, 22, 26, 29, 30, 31, 33, 34, 38])

    In [7]: np.c_[an[w_bnd_names],bn[w_bnd_names]]
    Out[7]:
    array([['Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel', 'Water/StrutAcrylicOpSurface//StrutSteel'],
           ['Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutSteel', 'Water/Strut2AcrylicOpSurface//StrutSteel'],
           ['Water/HamamatsuMaskOpticalSurface/HamamatsuMaskOpticalSurface/CDReflectorSteel', 'Water/HamamatsuMaskOpticalSurface//CDReflectorSteel'],
           ['Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel', 'Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface//Steel'],
           ['Vacuum/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
            'Vacuum/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf//Steel'],
           ['Water/NNVTMaskOpticalSurface/NNVTMaskOpticalSurface/CDReflectorSteel', 'Water/NNVTMaskOpticalSurface//CDReflectorSteel'],
           ['Vacuum/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel', 'Vacuum/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf//Steel'],
           ['Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel', 'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface//Steel'],
           ['Water/Steel_surface/Steel_surface/Steel', 'Water/Steel_surface//Steel']], dtype='<U122')



TODO : Compare implicit in old and new, work out how to add osur implicits to old

::

    epsilon:opticks blyth$ opticks-fl RINDEX_NoRINDEX
    ./extg4/X4PhysicalVolume.cc
    ./sysrap/sstandard.h
    ./sysrap/S4.h
    ./sysrap/SBnd.h
    ./ggeo/GSurfaceLib.hh
    ./ggeo/GSurfaceLib.cc
    ./u4/U4TreeBorder.h


X4PhysicalVolume::convertImplicitSurfaces_r
--------------------------------------------


Enabling OSUR implicits in old GGeo workflow catching too many, and 
causing very slow convert. 

WIP : work out why so many ? New workflow doesnt get that many. 

Start by saving the names in X4PhysicalVolume::convertSurfaces::

     685     const G4VPhysicalVolume* pv = m_top ;
     686     int depth = 0 ;
     687     LOG(LEVEL) << "[convertImplicitSurfaces_r num_surf1 " << num_surf1 ;
     688     LOG(info) << "[convertImplicitSurfaces_r num_surf1 " << num_surf1 ;
     689     convertImplicitSurfaces_r(pv, depth);
     690     num_surf1 = m_slib->getNumSurfaces() ;
     691 
     692     bool surfname_debug = true ;
     693     if( surfname_debug )
     694     {
     695         std::vector<std::string> surfnames ;
     696         m_slib->collectBorderSurfaceNames(surfnames);
     697         NP::WriteNames("/tmp/X4PhysicalVolume__convertSurfaces_surfnames.txt", surfnames) ;
     698     }
     699 



Getting so many due to the 3inch PMT /tmp/X4PhysicalVolume__convertSurfaces_surfnames.txt::

    94041 PMT_3inch_log_phys0x72ff8e0
    94042 PMT_3inch_cntr_phys0x6900c30
    94043 PMT_3inch_log_phys0x72ff9e0
    94044 PMT_3inch_cntr_phys0x6900c30
    94045 PMT_3inch_log_phys0x72ffae0
    94046 PMT_3inch_cntr_phys0x6900c30
    94047 PMT_3inch_log_phys0x72ffbe0
    94048 PMT_3inch_cntr_phys0x6900c30
    94049 PMT_3inch_log_phys0x72ffce0
    94050 PMT_3inch_cntr_phys0x6900c30
    94051 PMT_3inch_log_phys0x72ffde0
    94052 PMT_3inch_cntr_phys0x6900c30
    94053 PMT_3inch_log_phys0x72ffee0
    94054 PMT_3inch_cntr_phys0x6900c30


More than half from PMT_3inch::

    epsilon:u4 blyth$ grep PMT_3inch /tmp/X4PhysicalVolume__convertSurfaces_surfnames.txt | wc -l 
       51208

    epsilon:u4 blyth$ cat /tmp/X4PhysicalVolume__convertSurfaces_surfnames.txt | wc -l 
       94276


The new workflow border surface identity is pointer based. 
But old workflow is pv name based. 


::

    In [6]: np.c_[B.SSim.stree.standard.bnd_names]                                            
       ...

       ['Vacuum/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf//Steel'],
       ['Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface//Steel'],
       ['Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum'],
       ['Pyrex//PMT_3inch_absorb_logsurf1/Vacuum'],
       ['Water/Implicit_RINDEX_NoRINDEX_PMT_3inch_log_phys_PMT_3inch_cntr_phys//Steel'],
       ['Water///LS'],
       ['Water/Steel_surface//Steel'],
       ['vetoWater///Water'],
       ['Pyrex///Pyrex'],
       ['Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum'],
       ['Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum']], dtype='<U122')




Investigate boundary discrepancy
-----------------------------------

A side (old workflow)  missing lots of OSUR implicits unexpectedly.::


    In [1]: an = A.SSim.extra.GGeo.bnd_names

    In [2]: bn = B.SSim.stree.standard.bnd_names


    In [7]: set(bn)-set(an)
    Out[7]:
    {'Air/Implicit_RINDEX_NoRINDEX_lUpperChimney_phys_pUpperChimneySteel//Steel',
     'Air/Implicit_RINDEX_NoRINDEX_lUpperChimney_phys_pUpperChimneyTyvek//Tyvek',
     'Air/Implicit_RINDEX_NoRINDEX_pExpHall_pPoolCover//Steel',
     'Air/Implicit_RINDEX_NoRINDEX_pPlane_0_ff__pPanel_0_f_//Aluminium',
     'Air/Implicit_RINDEX_NoRINDEX_pPlane_0_ff__pPanel_1_f_//Aluminium',
     'Air/Implicit_RINDEX_NoRINDEX_pPlane_0_ff__pPanel_2_f_//Aluminium',
     'Air/Implicit_RINDEX_NoRINDEX_pPlane_0_ff__pPanel_3_f_//Aluminium',
     'Air/Implicit_RINDEX_NoRINDEX_pPlane_1_ff__pPanel_0_f_//Aluminium',
     'Air/Implicit_RINDEX_NoRINDEX_pPlane_1_ff__pPanel_1_f_//Aluminium',
     'Air/Implicit_RINDEX_NoRINDEX_pPlane_1_ff__pPanel_2_f_//Aluminium',
     'Air/Implicit_RINDEX_NoRINDEX_pPlane_1_ff__pPanel_3_f_//Aluminium',
     'Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air',
     'Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air',
     'Tyvek//Implicit_RINDEX_NoRINDEX_pOuterWaterPool_pPoolLining/vetoWater',
     'Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface//Steel',
     'Vacuum/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf//Steel',
     'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface//Steel',
     'Vacuum/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf//Steel',
     'Water/HamamatsuMaskOpticalSurface//CDReflectorSteel',
     'Water/Implicit_RINDEX_NoRINDEX_PMT_3inch_log_phys_PMT_3inch_cntr_phys//Steel',
     'Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lSteel_phys//Steel',
     'Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lUpper_phys//Steel',
     'Water/NNVTMaskOpticalSurface//CDReflectorSteel',
     'Water/Steel_surface//Steel',
     'Water/Strut2AcrylicOpSurface//StrutSteel',
     'Water/StrutAcrylicOpSurface//StrutSteel',
     'vetoWater/Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLb1.bt02_HBeam_phys//LatticedShellSteel',
     'vetoWater/Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLb1.bt05_HBeam_phys//LatticedShellSteel',
     'vetoWater/Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLb1.bt06_HBeam_phys//LatticedShellSteel',
     'vetoWater/Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLb1.bt07_HBeam_phys//LatticedShellSteel',
     'vetoWater/Implicit_RINDEX_NoRINDEX_pOuterWaterPool_GLb1.bt08_HBeam_phys//LatticedShellSteel',



HMM: OLD WORKFLOW IS LOTS MORE INVOLVED THAN NEW::

     951 GItemList* GBndLib::createNames()
     952 {
     953     unsigned int ni = getNumBnd();
     954     GItemList* names = new GItemList(getType());
     955     for(unsigned int i=0 ; i < ni ; i++)      // over bnd
     956     {
     957         const guint4& bnd = m_bnd[i] ;
     958         names->add(shortname(bnd).c_str());
     959     }
     960     return names ;
     961 }


     622 std::string GBndLib::shortname(unsigned boundary) const
     623 {
     624     guint4 bnd = getBnd(boundary);
     625     return shortname(bnd);
     626 }
     627 
     628 
     629 std::string GBndLib::shortname(const guint4& bnd) const
     630 {
     631     std::stringstream ss ;
     632     ss
     633        << (bnd[OMAT] == UNSET ? "OMAT-unset-error" : m_mlib->getName(bnd[OMAT]))
     634        << "/"
     635        << (bnd[OSUR] == UNSET ? "" : m_slib->getName(bnd[OSUR]))
     636        << "/"
     637        << (bnd[ISUR] == UNSET ? "" : m_slib->getName(bnd[ISUR]))
     638        << "/"
     639        << (bnd[IMAT] == UNSET ? "IMAT-unset-error" : m_mlib->getName(bnd[IMAT]))
     640        ;
     641     return ss.str();
     642 }



Ingredients for adding old workflow implicits.. need to add surface and reference int 
by minting an index thats collected into GBndLib::


    927 GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_sele     ct )
    1928 {
    1929 #ifdef X4_PROFILE
    1930     float t00 = BTimeStamp::RealTime();
    1931 #endif
    1932   
    1933     // record copynumber in GVolume, as thats one way to handle pmtid
    1934     const G4PVPlacement* placement = dynamic_cast<const G4PVPlacement*>(pv);
    1935     assert(placement);
    1936     G4int copyNumber = placement->GetCopyNo() ;
    1937 
    1938     X4Nd* parent_nd = parent ? static_cast<X4Nd*>(parent->getParallelNode()) : NULL ;
    1939 
    1940     unsigned boundary = addBoundary( pv, pv_p );
    1941     std::string boundaryName = m_blib->shortname(boundary);
    1942     int materialIdx = m_blib->getInnerMaterial(boundary);




Making sure boundary surface uses X4::ShortName 
------------------------------------------------

::

    In [1]: an = A.SSim.extra.GGeo.bnd_names

    In [2]: bn = B.SSim.stree.standard.bnd_names

    In [3]: an.shape
    Out[3]: (132,)

    In [4]: bn.shape
    Out[4]: (124,)

    In [5]: np.c_[an]                                        

::

    In [7]: set(an)-set(bn)
    Out[7]: 
    {'Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Vacuum/Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_dynode_tube_phy/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Vacuum/Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_grid_phy/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Vacuum/Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_inner_edge_phy/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Vacuum/Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_inner_ring_phy/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Vacuum/Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_outer_edge_phy/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Vacuum/Implicit_RINDEX_NoRINDEX_HamamatsuR12860_PMT_20inch_inner_phys_HamamatsuR12860_PMT_20inch_shield_phy/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Vacuum/Implicit_RINDEX_NoRINDEX_NNVTMCPPMT_PMT_20inch_inner_phys_NNVTMCPPMT_PMT_20inch_edge_phy/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Vacuum/Implicit_RINDEX_NoRINDEX_NNVTMCPPMT_PMT_20inch_inner_phys_NNVTMCPPMT_PMT_20inch_mcp_phy/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Vacuum/Implicit_RINDEX_NoRINDEX_NNVTMCPPMT_PMT_20inch_inner_phys_NNVTMCPPMT_PMT_20inch_tube_phy/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel',
     'Water/Implicit_RINDEX_NoRINDEX_lLowerChimney_phys_pLowerChimneySteel/Steel_surface/Steel',
     'Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lSteel2_phys/Strut2AcrylicOpSurface/StrutSteel',
     'Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lSteel_phys/StrutAcrylicOpSurface/StrutSteel',
     'Water/Implicit_RINDEX_NoRINDEX_pLPMT_Hamamatsu_R12860_HamamatsuR12860pMaskTail//AcrylicMask',
     'Water/Implicit_RINDEX_NoRINDEX_pLPMT_Hamamatsu_R12860_HamamatsuR12860pMaskTail/HamamatsuMaskOpticalSurface/CDReflectorSteel',
     'Water/Implicit_RINDEX_NoRINDEX_pLPMT_NNVT_MCPPMT_NNVTMCPPMTpMaskTail//AcrylicMask',
     'Water/Implicit_RINDEX_NoRINDEX_pLPMT_NNVT_MCPPMT_NNVTMCPPMTpMaskTail/NNVTMaskOpticalSurface/CDReflectorSteel'}

    In [8]: set(bn)-set(an)
    Out[8]: 
    {'Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface//Steel',
     'Vacuum/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf//Steel',
     'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface//Steel',
     'Vacuum/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf//Steel',
     'Water///AcrylicMask',
     'Water/HamamatsuMaskOpticalSurface//CDReflectorSteel',
     'Water/NNVTMaskOpticalSurface//CDReflectorSteel',
     'Water/Steel_surface//Steel',
     'Water/Strut2AcrylicOpSurface//StrutSteel',
     'Water/StrutAcrylicOpSurface//StrutSteel'}


Looks like A is now adding implicits when there is prexisting surface. 

* X4LogicalBorderSurface::Convert was using X4::Name not X4::ShortName 


HMM: using an index that doesnt get into GBndLib


::

    IndexError                                Traceback (most recent call last)
    ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.py in <module>
         14 
         15     bn = cf.sim.stree.standard.bnd_names
    ---> 16     l_bnd = bn[u_bnd]
         17 
         18     abn = cf.sim.stree.standard.bnd_names

    IndexError: index 124 is out of bounds for axis 0 with size 124



CONCLUDED TOO MUCH EFFORT TO BRING OSUR IMPLICITS TO THE OLD WORKFLOW : LEAPING TO NEW WORKFLOW
