CSGFoundry_CreateFromSim_shakedown_now_with_flexible_sn
==========================================================


Context
----------

* prev :doc:`CSGFoundry_CreateFromSim_shakedown`


Overview
----------

Two geometry routes::

     A0             X4      CSG_GGeo
     OLD : Geant4 -----> GGeo ----->  CSGFoundry 

     ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh   ## A0 : Loads GDML, pass world to G4CXOpticks  

     BP=X4Solid::Balance ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh



     B0+B1        U4Tree        CSGImport
     NEW : Geant4 ----->  stree ------>  CSGFoundry 
                          snd/sn

     ~/opticks/u4/tests/U4TreeCreateSSimTest.sh             ## B0 : Loads GDML, Create SSim/stree using U4Tree.h 
     ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh    ## B1 : Loads SSim/stree, runs CSGImport creating CSGFoundry


     B0           U4Tree        
     NEW : Geant4 ----->  stree 
                          snd/sn

     ~/opticks/u4/tests/U4TreeCreateSSimTest.sh  ## B0 : Loads GDML, Create SSim/stree using U4Tree.h 



     B1                        CSGImport
     NEW :                stree ------>  CSGFoundry 
                          snd/sn

     ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh    ## B1 : Loads SSim/stree, runs CSGImport creating CSGFoundry


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



TODO : A/B CSGPrim prim content : ints match, B lacks bbox
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


::

    


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

* TODO: improve nudge logging, only lv 95 is showing up ? All the above six lv should appear. 

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




TODO : Disable uncoincidence shifts in A to check if that explains all the above CSGNode diffs
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



