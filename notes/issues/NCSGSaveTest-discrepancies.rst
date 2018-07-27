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




Discreps
---------

1. transforms are tripletized by the save : need a format like "csg.py" option
2. missing idx.npy
3. nodes.py has gtransforms index off by one (its one based as written by python)

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



