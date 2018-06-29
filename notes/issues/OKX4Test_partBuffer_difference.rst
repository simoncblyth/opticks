OKX4Test_partBuffer_difference
=================================


::

    epsilon:~ blyth$ ab-;ab-i
    import numpy as np

    a = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5/partBuffer.npy")
    ta = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5/tranBuffer.npy")
    pa = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5/primBuffer.npy")

    b = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5/partBuffer.npy")
    tb = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5/tranBuffer.npy")
    pb = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5/primBuffer.npy")

    def cf(a, b):
        assert len(a) == len(b)
        assert a.shape == b.shape
        for i in range(len(a)):
            tca = a[i].view(np.int32)[2][3]
            tcb = b[i].view(np.int32)[2][3]
            tc = tca 
            assert tca == tcb
            if tca != tcb:
                print " tc mismatch %d %d " % (tca, tcb)
            pass

            gta = a[i].view(np.int32)[3][3]
            gtb = b[i].view(np.int32)[3][3]
            #assert gta == gtb
            msg = " gt mismatch " if gta != gtb else ""

            mx = np.max(a[i]-b[i])
            print " i:%3d tc:%3d gta:%2d gtb:%2d mx:%10s %s  " % ( i, tc, gta, gtb, mx, msg  )
            #if mx > 0.:
            #    print (a[i]-b[i])/mx
            pass
        pass
    pass
    cf(a,b)

    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/i.py
     i:  0 tc:  1 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i:  1 tc:  2 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i:  2 tc: 12 gta: 4 gtb: 4 mx:       0.0   
     i:  3 tc:  2 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i:  4 tc:  5 gta: 3 gtb: 3 mx:       0.0   
     i:  5 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i:  6 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i:  7 tc:  5 gta: 1 gtb: 1 mx:       0.0   
     i:  8 tc:  5 gta: 2 gtb: 2 mx:       0.0   
     i:  9 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 10 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 11 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 12 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 13 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 14 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 15 tc:  1 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 16 tc:  2 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 17 tc: 12 gta: 4 gtb: 4 mx:       0.0   
     i: 18 tc:  2 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 19 tc:  5 gta: 3 gtb: 3 mx:       0.0   
     i: 20 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 21 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 22 tc:  5 gta: 1 gtb: 1 mx:       0.0   
     i: 23 tc:  5 gta: 2 gtb: 2 mx:       0.0   
     i: 24 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 25 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 26 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 27 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 28 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 29 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 30 tc:  1 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 31 tc:  3 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 32 tc:  3 gta: 0 gtb: 2 mx:       0.0  gt mismatch   
     i: 33 tc:  7 gta: 1 gtb: 1 mx:       0.0   
     i: 34 tc:  7 gta: 1 gtb: 1 mx:7.6293945e-06   
     i: 35 tc:  7 gta: 2 gtb: 1 mx:1.9073486e-06  gt mismatch   
     i: 36 tc:  7 gta: 2 gtb: 1 mx:1.9073486e-06  gt mismatch   
     i: 37 tc:  3 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 38 tc:  7 gta: 1 gtb: 1 mx:       0.0   
     i: 39 tc:  7 gta: 1 gtb: 1 mx:       0.0   
     i: 40 tc: 12 gta: 1 gtb: 1 mx:       0.0   

    In [1]: 

    old buffer gtransformIdx (gta) always zero for typecodes 0/1/2/3  CSG_ZERO/UNION/SUBTRACTION/INTERSECTION


gtransforms on operator nodes or not ?
-----------------------------------------

analytic/csg.py:serialize collects transforms from all nodes in preorder fashion.

* NB collecting node level transforms (not global gtransforms)

::

     675     def serialize(self, suppress_identity=False):
     676         """
     677         Array is sized for a complete tree, empty slots stay all zero
     678         """
     679         if not self.is_root: self.analyse()
     680         buf = np.zeros((self.totnodes,self.NJ,self.NK), dtype=np.float32 )
     681 
     682         transforms = []
     683         planes = []
     684 
     685         def serialize_r(node, idx):
     686             """
     687             :param node:
     688             :param idx: 0-based complete binary tree index, left:2*idx+1, right:2*idx+2 
     689             """
     690             trs = node.transform
     691             if trs is None and suppress_identity == False:
     692                 trs = np.eye(4, dtype=np.float32)
     693                 # make sure root node always has a transform, incase of global placement 
     694                 # hmm root node is just an op-node it doesnt matter, need transform slots for all primitives 
     695             pass
     696 
     697             if trs is None:
     698                 itransform = 0
     699             else:
     700                 itransform = len(transforms) + 1  # 1-based index pointing to the transform
     701                 transforms.append(trs)
     702             pass

     /////// the above trips over itself leading to node.transform of None ending up with 
     /////// itransform of 1 pointing at an identity matrix  

     703 
     704 
     705             node_planes = node.planes
     706             if len(node_planes) == 0:
     707                 planeIdx = 0
     708                 planeNum = 0
     709             else:
     710                 planeIdx = len(planes) + 1   # 1-based index pointing to the first plane for the node
     711                 planeNum = len(node_planes)
     712                 planes.extend(node_planes)
     713             pass
     714             log.debug("serialize_r idx %3d itransform %2d planeIdx %2d " % (idx, itransform, planeIdx))
     715 
     716             buf[idx] = node.as_array(itransform, planeIdx, planeNum)
     717 
     718             if node.left is not None and node.right is not None:
     719                 serialize_r( node.left,  2*idx+1)
     720                 serialize_r( node.right, 2*idx+2)
     721             pass
     722         pass
     723 
     724         serialize_r(self, 0)
     725 
     726         tbuf = np.vstack(transforms).reshape(-1,4,4) if len(transforms) > 0 else None
     727         pbuf = np.vstack(planes).reshape(-1,4) if len(planes) > 0 else None
     728 
     729         log.debug("serialized CSG of height %2d into buf with %3d nodes, %3d transforms, %3d planes, meta %r " % (self.height, len(buf), len(transforms), len(planes), self.meta ))



On import the gtransforms (**for primitives only**) are constructed by multiplication 
down the tree, and uniquely collected into m_gtransforms with the 1-based index being set 
on the node.


    1006         node = import_primitive( idx, typecode );
    1007 
    1008         node->parent = parent ;                // <-- parent hookup needed prior to gtransform collection 
    1009         node->idx = idx ;
    1010         node->complement = complement ;
    1011 
    1012         node->transform = import_transform_triple( transform_idx ) ;
    1013 
    1014         const nmat4triple* gtransform = node->global_transform();
    1015 
    1016         // see opticks/notes/issues/subtree_instances_missing_transform.rst
    1017         //if(gtransform == NULL && m_usedglobally)
    1018         if(gtransform == NULL )  // move to giving all primitives a gtransform 
    1019         {
    1020             gtransform = nmat4triple::make_identity() ;
    1021         }
    1022 
    1023         unsigned gtransform_idx = gtransform ? addUniqueTransform(gtransform) : 0 ;
    1024 
    1025         node->gtransform = gtransform ;
    1026         node->gtransform_idx = gtransform_idx ; // 1-based, 0 for None
    1027     }




primIdx 2 missing transform, when restrict to primitives

::

    epsilon:analytic blyth$ ab-prim
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5
    prim (5, 4) part (41, 4, 4) tran (12, 3, 4, 4) 

    primIdx 0 prim array([ 0, 15,  0,  0], dtype=int32) partOffset 0 numParts 15 tranOffset 0 planOffset 0  
        Part  1  0            union     0.0    
        Part  2  0     intersection     0.0    
        Part 12  4         cylinder     -84.5    
        Part  2  0     intersection     0.0    
        Part  5  3           sphere     69.0    
        Part  5  1           sphere     0.0    
        Part  5  2           sphere     43.0    

    primIdx 1 prim array([15, 15,  4,  0], dtype=int32) partOffset 15 numParts 15 tranOffset 4 planOffset 0  
        Part  1  0            union     0.0    
        Part  2  0     intersection     0.0    
        Part 12  4         cylinder     -81.5    
        Part  2  0     intersection     0.0    
        Part  5  3           sphere     69.0    
        Part  5  1           sphere     0.0    
        Part  5  2           sphere     43.0    

    primIdx 2 prim array([30,  7,  8,  0], dtype=int32) partOffset 30 numParts 7 tranOffset 8 planOffset 0  
        Part  1  0            union     0.0    
        Part  3  0       difference     0.0    
        Part  3  0       difference     0.0    
        Part  7  1          zsphere     0.0    
        Part  7  1          zsphere     0.0    
        Part  7  2          zsphere     43.0    
        Part  7  2          zsphere     43.0    

    primIdx 3 prim array([37,  3, 10,  0], dtype=int32) partOffset 37 numParts 3 tranOffset 10 planOffset 0  
        Part  3  0       difference     0.0    
        Part  7  1          zsphere     69.0    
        Part  7  1          zsphere     69.0    

    primIdx 4 prim array([40,  1, 11,  0], dtype=int32) partOffset 40 numParts 1 tranOffset 11 planOffset 0  
        Part 12  1         cylinder     -81.5    
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5
    prim (5, 4) part (41, 4, 4) tran (11, 3, 4, 4) 

    primIdx 0 prim array([ 0, 15,  0,  0], dtype=int32) partOffset 0 numParts 15 tranOffset 0 planOffset 0  
        Part  1  0            union     0.0    
        Part  2  0     intersection     0.0    
        Part 12  4         cylinder     -84.5    
        Part  2  0     intersection     0.0    
        Part  5  3           sphere     69.0    
        Part  5  1           sphere     0.0    
        Part  5  2           sphere     43.0    

    primIdx 1 prim array([15, 15,  4,  0], dtype=int32) partOffset 15 numParts 15 tranOffset 4 planOffset 0  
        Part  1  0            union     0.0    
        Part  2  0     intersection     0.0    
        Part 12  4         cylinder     -81.5    
        Part  2  0     intersection     0.0    
        Part  5  3           sphere     69.0    
        Part  5  1           sphere     0.0    
        Part  5  2           sphere     43.0    

    primIdx 2 prim array([30,  7,  8,  0], dtype=int32) partOffset 30 numParts 7 tranOffset 8 planOffset 0  
        Part  1  0            union     0.0    
        Part  3  0       difference     0.0    
        Part  3  0       difference     0.0    
        Part  7  1          zsphere     0.0    
        Part  7  1          zsphere     0.0    
        Part  7  1          zsphere     0.0    
        Part  7  1          zsphere     0.0    

    primIdx 3 prim array([37,  3,  9,  0], dtype=int32) partOffset 37 numParts 3 tranOffset 9 planOffset 0  
        Part  3  0       difference     0.0    
        Part  7  1          zsphere     69.0    
        Part  7  1          zsphere     69.0    

    primIdx 4 prim array([40,  1, 10,  0], dtype=int32) partOffset 40 numParts 1 tranOffset 10 planOffset 0  
        Part 12  1         cylinder     -81.5    
    epsilon:analytic blyth$ 







Still small differences 
----------------------------

* z1 for zsphere shows 1e-6 mm differences

* gtransform differences from whether to collect gtransforms 
  on operator nodes or just leaf node primitives ?

* on GPU there is no multiplying up the tree, the gtransforms 
  are only used for primitives 


::

    epsilon:~ blyth$ ab-;ab-i-partBuffer 
    import numpy as np
    a = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5/partBuffer.npy")
    ta = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5/tranBuffer.npy")
    b = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5/partBuffer.npy")
    tb = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5/tranBuffer.npy")

    def cf(a, b):
        assert len(a) == len(b)
        assert a.shape == b.shape
        for i in range(len(a)):
            tca = a[i].view(np.int32)[2][3]
            tcb = b[i].view(np.int32)[2][3]
            tc = tca 
            assert tca == tcb
            if tca != tcb:
                print " tc mismatch %d %d " % (tca, tcb)
            pass

            gta = a[i].view(np.int32)[3][3]
            gtb = b[i].view(np.int32)[3][3]
            #assert gta == gtb

            msg = " gt mismatch " if gta != gtb else ""

            mx = np.max(a[i]-b[i])
            print " i:%3d tc:%3d gta:%2d gtb:%2d mx:%10s %s  " % ( i, tc, gta, gtb, mx, msg  )
            if mx > 0.:
                print (a[i]-b[i])/mx
            pass
        pass
    pass
    cf(a,b)

    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/i.py
     i:  0 tc:  1 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i:  1 tc:  2 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i:  2 tc: 12 gta: 4 gtb: 4 mx:       0.0   
     i:  3 tc:  2 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i:  4 tc:  5 gta: 3 gtb: 3 mx:       0.0   
     i:  5 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i:  6 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i:  7 tc:  5 gta: 1 gtb: 1 mx:       0.0   
     i:  8 tc:  5 gta: 2 gtb: 2 mx:       0.0   
     i:  9 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 10 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 11 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 12 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 13 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 14 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 15 tc:  1 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 16 tc:  2 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 17 tc: 12 gta: 4 gtb: 4 mx:       0.0   
     i: 18 tc:  2 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 19 tc:  5 gta: 3 gtb: 3 mx:       0.0   
     i: 20 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 21 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 22 tc:  5 gta: 1 gtb: 1 mx:       0.0   
     i: 23 tc:  5 gta: 2 gtb: 2 mx:       0.0   
     i: 24 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 25 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 26 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 27 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 28 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 29 tc:  0 gta: 0 gtb: 0 mx:       0.0   
     i: 30 tc:  1 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 31 tc:  3 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 32 tc:  3 gta: 0 gtb: 2 mx:       0.0  gt mismatch   
     i: 33 tc:  7 gta: 1 gtb: 1 mx:       0.0   
     i: 34 tc:  7 gta: 1 gtb: 1 mx:7.6293945e-06   
    [[0. 0. 0. 0.]
     [1. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
     i: 35 tc:  7 gta: 2 gtb: 1 mx:1.9073486e-06  gt mismatch   
    [[0. 0. 0. 0.]
     [1. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
     i: 36 tc:  7 gta: 2 gtb: 1 mx:1.9073486e-06  gt mismatch   
    [[0. 0. 0. 0.]
     [1. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
     i: 37 tc:  3 gta: 0 gtb: 1 mx:       0.0  gt mismatch   
     i: 38 tc:  7 gta: 1 gtb: 1 mx:       0.0   
     i: 39 tc:  7 gta: 1 gtb: 1 mx:       0.0   
     i: 40 tc: 12 gta: 1 gtb: 1 mx:       0.0   






FIXED gibberish in partBuffer buffer via nzsphere
--------------------------------------------------------

::

    In [24]: exit
    epsilon:5 blyth$ ab-i partBuffer.npy 
    import numpy as np
    a = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5/partBuffer.npy")
    b = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5/partBuffer.npy")
    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/i.py

    In [1]: a[-1]
    Out[1]: 
    array([[  0. ,   0. ,   0. ,  27.5],
           [-83. ,  83. ,   0. ,   0. ],
           [  0. ,   0. ,   0. ,   0. ],
           [  0. ,   0. ,   0. ,   0. ]], dtype=float32)

    In [2]: b[-1]
    Out[2]: 
    array([[  0. ,   0. ,   0. ,  27.5],
           [-83. ,  83. ,   0. ,   0. ],
           [  0. ,   0. ,   0. ,   0. ],
           [  0. ,   0. ,   0. ,   0. ]], dtype=float32)

    In [3]: a[-2]
    Out[3]: 
    array([[  0.    ,   0.    ,   0.    ,  98.    ],
           [-98.    , -12.8687,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]], dtype=float32)

    In [4]: b[-2]
    Out[4]: 
    array([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  9.8000e+01],
           [-9.8000e+01, -1.2869e+01,  4.2039e-44,  5.4651e-44],
           [-1.3424e+22,  4.5915e-41, -1.3421e+22,  9.8091e-45],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.4013e-45]], dtype=float32)

    In [5]: 

::

    In [5]: b[-2].view(np.int32)
    Out[5]: 
    array([[          0,           0,           0,  1120141312],
           [-1027342336, -1051859421,          30,          39],
           [ -466227936,       32766,  -466230608,           7],
           [          0,           0,           0,           1]], dtype=int32)

    In [6]: a[-2].view(np.int32)
    Out[6]: 
    array([[          0,           0,           0,  1120141312],
           [-1027342336, -1051859420,          30,          39],
           [          3,           0,           0,           7],
           [          0,           0,           0,           1]], dtype=int32)

    In [7]: 


* getting some uninitialized crazies (?) in bbmin slots 



::

    1360 void NCSG::export_node(nnode* node, unsigned idx)
    1361 {
    1362     assert(idx < m_num_nodes);
    1363     LOG(trace) << "NCSG::export_node"
    1364               << " idx " << idx
    1365               << node->desc()
    1366               ;
    1367 
    1368     export_gtransform(node);
    1369     export_planes(node);
    1370  
    1371     // crucial 2-step here, where m_nodes gets totally rewritten
    1372     npart pt = node->part();
    1373     m_nodes->setPart( pt, idx);  // writes 4 quads to buffer
    1374 }

::

     461 npart nnode::part() const
     462 {
     463     // this is invoked by NCSG::export_r to totally re-write the nodes buffer 
     464     // BUT: is it being used by partlist approach, am assuming not by not setting bbox
     465 
     466     npart pt ;
     467     pt.zero();
     468     pt.setParam(  param );
     469     pt.setParam1( param1 );
     470     pt.setParam2( param2 );
     471     pt.setParam3( param3 );
     472 
     473     pt.setTypeCode( type );
     474     pt.setGTransform( gtransform_idx, complement );
     475 
     476     // gtransform_idx is index into a buffer of the distinct compound transforms for the tree
     477 
     478     if(npart::VERSION == 0u)
     479     {
     480         nbbox bb = bbox();
     481         pt.setBBox( bb );
     482     }
     483 
     484     return pt ;
     485 }


::

     10 struct NPY_API npart
     11 {
     12     nquad q0 ;  // x,y,z,w (float): param 
     13     nquad q1 ;  // x,y,z,w (uint) -/index/boundary/flags
     14     nquad q2 ;  // x,y,z (float):bbmin   w(uint):typecode  
     15     nquad q3 ;  // x,y,z (float):bbmax   
     16 
     17     nquad qx ;  // <- CPU only 
     18      
     19     static unsigned VERSION ;  // 0:with bbox, 1:without bbox and with GTransforms
     20 
     21     void zero();
     22     void dump(const char* msg);
     23     void setTypeCode(OpticksCSG_t typecode);
     24     void setGTransform(unsigned gtransform_idx, bool complement=false);
     25     void setBBox(const nbbox& bb);


::

     14 /*  
     15     
     16 
     17         0   1   2   3 
     18        
     19     0   .   .   .   .
     20 
     21     1   .   .   .   .
     22     
     23     2   .   .   .   tc
     24     
     25     3   .   .   .   gt 
     26 
     27 */  
     28     
     29     
     30 void npart::setTypeCode(OpticksCSG_t typecode)
     31 {
     32     assert( TYPECODE_J == 2 && TYPECODE_K == 3 );
     33     q2.u.w = typecode ;  
     34 }
     35 
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


Typecode 7 (CSG_ZSPHERE) always has 3 for endcap flags in a::

    In [18]: a[:,2].view(np.int32)
    Out[18]: 
    array([[ 0,  0,  0,  1],
           [ 0,  0,  0,  2],
           [ 0,  0,  0, 12],
           [ 0,  0,  0,  2],
           [ 0,  0,  0,  5],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  5],
           [ 0,  0,  0,  5],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  1],
           [ 0,  0,  0,  2],
           [ 0,  0,  0, 12],
           [ 0,  0,  0,  2],
           [ 0,  0,  0,  5],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  5],
           [ 0,  0,  0,  5],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  1],
           [ 0,  0,  0,  3],
           [ 0,  0,  0,  3],
           [ 3,  0,  0,  7],
           [ 3,  0,  0,  7],
           [ 3,  0,  0,  7],
           [ 3,  0,  0,  7],
           [ 0,  0,  0,  3],
           [ 3,  0,  0,  7],
           [ 3,  0,  0,  7],
           [ 0,  0,  0, 12]], dtype=int32)


::

    epsilon:npy blyth$ OpticksCSGTest
     type   0 name                 zero
     type   1 name                union
     type   2 name         intersection
     type   3 name           difference
     type   4 name             partlist
     type   5 name               sphere
     type   6 name                  box
     type   7 name              zsphere
     type   8 name                zlens
     type   9 name                  pmt
     type  10 name                prism
     type  11 name                 tubs
     type  12 name             cylinder
     type  13 name                 slab
     type  14 name                plane
     type  15 name                 cone
     type  16 name            multicone
     type  17 name                 box3
     type  18 name            trapezoid
     type  19 name     convexpolyhedron
     type  20 name                 disc
     type  21 name              segment
     type  22 name            ellipsoid
     type  23 name                torus
     type  24 name          hyperboloid
     type  25 name                cubic
     type  26 name            undefined
    epsilon:npy blyth$ 


::

     86 inline NPY_API unsigned nzsphere::flags() const { return param2.u.x ; }
     87 


