CSGFoundry_CreateFromSim_shakedown_now_with_flexible_sn
==========================================================


Context
----------

* prev :doc:`CSGFoundry_CreateFromSim_shakedown`


Overview
----------

Two geometry routes::

     A              X4      CSG_GGeo
     OLD : Geant4 -----> GGeo ----->  CSGFoundry 

     B            U4Tree        CSGImport
     NEW : Geant4 ----->  stree ------>  CSGFoundry 
                          snd/sn



Workflow to create A and B CSGFoundry (and SSim/stree) 
-------------------------------------------------------

Create the A and B geometries::

    ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh   ## A
    ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh    ## B 

Breakpoint debug of A::

    BP=sn::Serialize ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh

Python comparison::

    ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh ana



HUH : BP=sn::Serialize for A shows all pool empty : probably this is an issue of the other stree
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


::

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


A/B comparison 
-----------------


~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh ana::

      ['sOuterWaterPool', '0x', '5a05390'],
       ['sPoolLining', '0x', '5a04be0'],
       ['sBottomRock', '0x', '5a00520'],
       ['sWorld', '0x', '59f2060']], dtype='<U44')

    In [38]: np.char.partition(a.meshname.astype("U"),"0x").shape
    Out[38]: (139, 3)

    In [42]: b_meshname = b.meshname.astype("U")
    In [43]: a_meshname = np.char.partition(a.meshname.astype("U"),"0x")[:,0]


::

    epsilon:tests blyth$  ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh info
             BASH_SOURCE : /Users/blyth/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh 
                     bin : CSGFoundry_CreateFromSimTest 
                    GEOM : V1J009 
                    BASE : /Users/blyth/.opticks/GEOM/V1J009 
                    FOLD : /tmp/blyth/opticks/CSGFoundry_CreateFromSimTest 
                   check : /Users/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/stree/nds.npy 
                A_CFBASE : /Users/blyth/.opticks/GEOM/V1J009 
                B_CFBASE : /tmp/blyth/opticks/CSGFoundry_CreateFromSimTest 
                  script : /Users/blyth/opticks/CSG/tests/CSGFoundryAB.py 

    In [24]: np.c_[ua[100:110],ub[100:110]]
    Out[24]: 
    array([['ZC2.A06_B06_FlangeI_Web_FlangeII', '30', 'ZC2.A05_B05_FlangeI_Web_FlangeII', '30'],
           ['ZC2.A06_B07_FlangeI_Web_FlangeII', '30', 'ZC2.A05_B06_FlangeI_Web_FlangeII', '30'],
           ['ZC2.B01_B01_FlangeI_Web_FlangeII', '30', 'ZC2.A06_B06_FlangeI_Web_FlangeII', '30'],
           ['ZC2.B03_B03_FlangeI_Web_FlangeII', '30', 'ZC2.A06_B07_FlangeI_Web_FlangeII', '30'],
           ['ZC2.B05_B05_FlangeI_Web_FlangeII', '30', 'ZC2.B01_B01_FlangeI_Web_FlangeII', '30'],
           ['base_steel', '1', 'ZC2.B03_B03_FlangeI_Web_FlangeII', '30'],
           ['mask_PMT_20inch_vetosMask', '1', 'ZC2.B05_B05_FlangeI_Web_FlangeII', '30'],
           ['sAcrylic', '1', 'base_steel', '1'],
           ['sAirTT', '1', 'mask_PMT_20inch_vetosMask', '1'],
           ['sBar', '128', 'mask_PMT_20inch_vetosMask_virtual', '1']], dtype='<U44')




FIXED : A.SSim.stree._csg.sn all references are -1 
------------------------------------------------------

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


After A:stree_standin FIX getting stree._csg perfect match as expected
------------------------------------------------------------------------

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






