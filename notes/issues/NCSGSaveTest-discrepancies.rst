NCSGSaveTest-discrepancies
=====================================

Test
------

::

    NCSGSaveTest 

    // loads NCSG from "$TMP/tboolean-box--/1" 
    // writes NCSG to "$TMP/tboolean-box--save/1"

::

    epsilon:1 blyth$ np.py /tmp/blyth/opticks/tboolean-box--save/1
    /tmp/blyth/opticks/tboolean-box--save/1
    /tmp/blyth/opticks/tboolean-box--save/1/transforms.npy : (1, 3, 4, 4) 
    /tmp/blyth/opticks/tboolean-box--save/1/nodes.npy : (1, 4, 4) 
    epsilon:1 blyth$ 
    epsilon:1 blyth$ np.py /tmp/blyth/opticks/tboolean-box--/1
    /tmp/blyth/opticks/tboolean-box--/1
    /tmp/blyth/opticks/tboolean-box--/1/transforms.npy : (1, 4, 4) 
    /tmp/blyth/opticks/tboolean-box--/1/nodes.npy : (1, 4, 4) 
    /tmp/blyth/opticks/tboolean-box--/1/idx.npy : (4,) 
    epsilon:1 blyth$ 



Major NCSG Refactoring 
--------------------------

Taking a bulldozer to the monolith. Split into NCSG into:

1. NCSGData
2. NPYList
3. NPYMeta 

Problems with the from nnode Adopt route should show up in standard tests, 
but also want to keep the from python route working. So remember to::



Failing to make a gtransforms buffer for this route
------------------------------------------------------

::

    tboolean-box -D

::

    (lldb) f 5
    frame #5: 0x00000001018ba62a libGGeo.dylib`GMaker::makeFromCSG(csg=0x000000010eec2900, (null)=0x000000010eebebe0, verbosity=0) at GMaker.cc:163
       160 	    volume->setLVName( strdup(lvn.c_str()) );
       161 	    volume->setCSGFlag( type );
       162 	
    -> 163 	    GParts* pts = GParts::make( csg, spec, verbosity );
       164 	
       165 	    volume->setParts( pts );
       166 	
    (lldb) p csg->m_csgdata->m_npy->desc()
    (std::__1::string) $11 = "NPYList srcnodes.npy 1,4,4   srcidx.npy 1,4   srctransforms.npy 1,4,4   nodes.npy 1,4,4   planes.npy 0,4   idx.npy 1,4   transforms.npy 1,3,4,4  "
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff74570b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7473b080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff744cc1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff744941ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000101877a52 libGGeo.dylib`GParts::make(tree=0x000000010eec2900, spec="Rock//perfectAbsorbSurface/Vacuum", verbosity=0) at GParts.cc:193
      * frame #5: 0x00000001018ba62a libGGeo.dylib`GMaker::makeFromCSG(csg=0x000000010eec2900, (null)=0x000000010eebebe0, verbosity=0) at GMaker.cc:163
        frame #6: 0x00000001018ba1a7 libGGeo.dylib`GMaker::makeFromCSG(this=0x000000010eebd7c0, csg=0x000000010eec2900, verbosity=0) at GMaker.cc:114
        frame #7: 0x00000001018b5a00 libGGeo.dylib`GGeoTest::importCSG(this=0x000000010eeb6130, volumes=size=1) at GGeoTest.cc:363
        frame #8: 0x00000001018b5032 libGGeo.dylib`GGeoTest::initCreateCSG(this=0x000000010eeb6130) at GGeoTest.cc:200
        frame #9: 0x00000001018b4a01 libGGeo.dylib`GGeoTest::init(this=0x000000010eeb6130) at GGeoTest.cc:137
        frame #10: 0x00000001018b48ad libGGeo.dylib`GGeoTest::GGeoTest(this=0x000000010eeb6130, ok=0x000000010af4b220, basis=0x000000010af57640) at GGeoTest.cc:128
        frame #11: 0x00000001018b4c65 libGGeo.dylib`GGeoTest::GGeoTest(this=0x000000010eeb6130, ok=0x000000010af4b220, basis=0x000000010af57640) at GGeoTest.cc:123
        frame #12: 0x00000001005f0bbf libOpticksGeo.dylib`OpticksHub::createTestGeometry(this=0x000000010af4ee20, basis=0x000000010af57640) at OpticksHub.cc:461
        frame #13: 0x00000001005ef5a2 libOpticksGeo.dylib`OpticksHub::loadGeometry(this=0x000000010af4ee20) at OpticksHub.cc:415
        frame #14: 0x00000001005ee2c2 libOpticksGeo.dylib`OpticksHub::init(this=0x000000010af4ee20) at OpticksHub.cc:176
        frame #15: 0x00000001005ee1ac libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010af4ee20, ok=0x000000010af4b220) at OpticksHub.cc:158
        frame #16: 0x00000001005ee3cd libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010af4ee20, ok=0x000000010af4b220) at OpticksHub.cc:157
        frame #17: 0x00000001000d3d7b libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfddc8, argc=28, argv=0x00007ffeefbfdea8, argforced=0x0000000000000000) at OKMgr.cc:44
        frame #18: 0x00000001000d41bb libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfddc8, argc=28, argv=0x00007ffeefbfdea8, argforced=0x0000000000000000) at OKMgr.cc:52
        frame #19: 0x000000010000b995 OKTest`main(argc=28, argv=0x00007ffeefbfdea8) at OKTest.cc:13
        frame #20: 0x00007fff74420015 libdyld.dylib`start + 1
        frame #21: 0x00007fff74420015 libdyld.dylib`start + 1
    (lldb) 


Seems issue from NCSGList loading, failing to somehow great the GTransforms buffer::

    TEST=NCSGListTest om-t


Argh : NCSG spills its guts in NCSGList.  Need to keep some distance between these.




Discreps
---------

1. transforms are tripletized by the save : need a format like "csg.py" option
   (are calling this the src format) 
2. missing idx.npy
3. nodes.py has gtransforms index off by one (its one based as written by python)
   (this was due to dual-purposing ... are now splitting the src format and the transport format)

analytic/csg.py::

    1026     def as_array(self, itransform=0, planeIdx=0, planeNum=0):
    1027         """
    1028         Both primitive and internal nodes:
    1029 
    1030         * q2.u.w : CSG type code eg CSG_UNION, CSG_DIFFERENCE, CSG_INTERSECTION, CSG_SPHERE, CSG_BOX, ... 
    1031         * q3.u.w : 1-based transform index, 0 for None
    1032 
    1033         Primitive nodes only:
    1034 
    1035         * q0 : 4*float parameters eg center and radius for sphere
    1036 
    1037         """
    1038         arr = np.zeros( (self.NJ, self.NK), dtype=np.float32 )
    ....   
    1052 
    1053         if self.transform is not None:
    1054             assert itransform > 0, itransform  # 1-based transform index
    1055             arr.view(np.uint32)[Q3,W] = itransform
    1056         pass
    1057 
    1058         if self.complement:
    1059             # view as float in order to set signbit 0x80000000
    1060             # do via float as this works even when the value is integer zero yielding negative zero
    1061             # AND with 0x7fffffff to recover the transform idx
    1062             np.copysign(arr.view(np.float32)[Q3,W:W+1], -1. , arr.view(np.float32)[Q3,W:W+1] )
    1063         pass
    1064 
    1065         if len(self.planes) > 0:
    1066             assert planeIdx > 0 and planeNum > 3, (planeIdx, planeNum)  # 1-based plane index
    1067             arr.view(np.uint32)[Q0,X] = planeIdx   # cf NNode::planeIdx
    1068             arr.view(np.uint32)[Q0,Y] = planeNum   # cf NNode::planeNum
    1069         pass
    1070 
    1071         arr.view(np.uint32)[Q2,W] = self.typ
    1072 
    1073         return arr


::

     503 npart nnode::part() const
     504 {  
     505     // this is invoked by NCSG::export_r to totally re-write the nodes buffer 
     506     // BUT: is it being used by partlist approach, am assuming not by not setting bbox
     507    
     508     npart pt ; 
     509     pt.zero();
     510     pt.setParam(  param );
     511     pt.setParam1( param1 );
     512     pt.setParam2( param2 );
     513     pt.setParam3( param3 );
     514 
     515     pt.setTypeCode( type ); 
     516     pt.setGTransform( gtransform_idx, complement );
     517   
     518     // gtransform_idx is index into a buffer of the distinct compound transforms for the tree
     519    
     520     if(npart::VERSION == 0u)
     521     {       
     522         nbbox bb = bbox();
     523         pt.setBBox( bb );   
     524     }       
     525             
     526     return pt ;
     527 } 


     36 void npart::setGTransform(unsigned gtransform_idx, bool complement)
     37 {
     38     assert(VERSION == 1u);
     39 
     40    assert( GTRANSFORM_J == 3 && GTRANSFORM_K == 3 );
     41 
     42    unsigned gpack = gtransform_idx & SSys::OTHERBIT32 ;
     43    if(complement) gpack |= SSys::SIGNBIT32 ; 
     44     
     45    LOG(debug) << "npart::setGTransform"
     46              << " gtransform_idx " << gtransform_idx
     47              << " complement " << complement
     48              << " gpack " << gpack
     49              << " gpack(hex) " << std::hex << gpack << std::dec
     50              ; 
     51 
     52    q3.u.w = gpack ; 
     53     
     54 }   

Hmm problem is that gtransform_idx is set on import.

Hmm not quite, there are two distinct formats of nodes.py:

1. input from python format : where [3,3] is the itransform pointing to local transform of shape (n,4,4)
2. saved from NCSG format : where [3,3] is the 1-based gtransform_idx pointing to global transform of shape (n,3,4,4)

How to de-scrambulate ? inputnodes.py vs nodes.py  



