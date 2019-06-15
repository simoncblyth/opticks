GParts
========


The structure of persisted GParts buffers in the geocache foe DayaBay geometry 
is below examined.


TODO : compare with old GParts buffers
------------------------------------------

::

    epsilon:GPartsAnalytic blyth$ l
    total 0
    drwxr-xr-x  7 blyth  staff  - 238 Nov 29  2017 0
    drwxr-xr-x  6 blyth  staff  - 204 Nov 29  2017 2
    drwxr-xr-x  6 blyth  staff  - 204 Nov 29  2017 3
    drwxr-xr-x  6 blyth  staff  - 204 Nov 29  2017 4
    drwxr-xr-x  6 blyth  staff  - 204 Nov 29  2017 5
    epsilon:GPartsAnalytic blyth$ np.py 
    /Volumes/Delta/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GPartsAnalytic
          ./0/GParts.txt : 11984 
      ./0/partBuffer.npy : (11984, 4, 4) 
      ./0/planBuffer.npy : (672, 4) 
      ./0/primBuffer.npy : (3116, 4) 
      ./0/tranBuffer.npy : (5344, 3, 4, 4) 
          ./2/GParts.txt : 1 
      ./2/partBuffer.npy : (1, 4, 4) 
      ./2/primBuffer.npy : (1, 4) 
      ./2/tranBuffer.npy : (1, 3, 4, 4) 
          ./3/GParts.txt : 1 
      ./3/partBuffer.npy : (1, 4, 4) 
      ./3/primBuffer.npy : (1, 4) 
      ./3/tranBuffer.npy : (1, 3, 4, 4) 
          ./4/GParts.txt : 1 
      ./4/partBuffer.npy : (1, 4, 4) 
      ./4/primBuffer.npy : (1, 4) 
      ./4/tranBuffer.npy : (1, 3, 4, 4) 
          ./5/GParts.txt : 41 
      ./5/partBuffer.npy : (41, 4, 4) 
      ./5/primBuffer.npy : (5, 4) 
      ./5/tranBuffer.npy : (12, 3, 4, 4) 
    epsilon:GPartsAnalytic blyth$ l 0/
    total 4464
    -rw-r--r--  1 blyth  staff  -  419982 Nov 29  2017 GParts.txt
    -rw-r--r--  1 blyth  staff  -  767056 Nov 29  2017 partBuffer.npy
    -rw-r--r--  1 blyth  staff  -   10832 Nov 29  2017 planBuffer.npy
    -rw-r--r--  1 blyth  staff  -   49936 Nov 29  2017 primBuffer.npy
    -rw-r--r--  1 blyth  staff  - 1026128 Nov 29  2017 tranBuffer.npy
    epsilon:GPartsAnalytic blyth$ 




::

    epsilon:0 blyth$ np.py primBuffer.npy 
    (3116, 4)
    ...
    u32
    [[    0     1     0     0]
     [    1     1     1     0]
     [    2     1     2     0]
     ...
     [11981     1  5341   672]
     [11982     1  5342   672]
     [11983     1  5343   672]]    # lots of transforms, quite a few planes

::

    In [4]: a[:100]
    Out[4]: 
    array([[  0,   1,   0,   0],
           [  1,   1,   1,   0],
           [  2,   1,   2,   0],
           [  3,   7,   3,   0],
           [ 10,   7,   5,   0],
           [ 17,   7,   7,   0],
           [ 24,   7,   9,   0],
           [ 31,   7,  11,   0],
           [ 38,   3,  14,   0],
           [ 41,   3,  15,   0],
           [ 44,   3,  16,   0],
           [ 47,   1,  17,   0],
           [ 48,   7,  18,   0],
           [ 55,   3,  20,   0],

    In [7]: a[:100,1]
    Out[7]: 
    array([1, 1, 1, 7, 7, 7, 7, 7, 3, 3, 3, 1, 7, 3, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=int32)

    In [8]: a[:,1].max()
    Out[8]: 31

    In [10]: np.unique(a[:,1], return_counts=True)
    Out[10]: 
    (array([ 1,  3,  7, 15, 31], dtype=int32),
     array([ 638, 2140,  190,   62,   86]))


    In [1]: t = np.load("tranBuffer.npy")

    In [2]: t.shape
    Out[2]: (5344, 3, 4, 4)

    In [3]: t
    Out[3]: 
    array([[[[      0.5432,      -0.8396,       0.    ,       0.    ],
             [      0.8396,       0.5432,       0.    ,       0.    ],
             [      0.    ,       0.    ,       1.    ,       0.    ],
             [ -18079.453 , -799699.44  ,   -6605.    ,       1.    ]],

            [[      0.5432,       0.8396,       0.    ,       0.    ],
             [     -0.8396,       0.5432,       0.    ,       0.    ],
             [      0.    ,       0.    ,       1.    ,       0.    ],
             [-661623.25  ,  449556.2   ,    6605.    ,       1.    ]],

            [[      0.5432,      -0.8396,       0.    , -661623.25  ],
             [      0.8396,       0.5432,       0.    ,  449556.2   ],
             [      0.    ,       0.    ,       1.    ,    6605.    ],
             [      0.    ,       0.    ,       0.    ,       1.    ]]],







Structure of multi-complete tree buffers 
-------------------------------------------

* slot zero corresponds to the non-instanced global geometry 
* note the primBuffer.npy has starting part index and complete binary tree size
* binary tree sizes are always power of two minus 1 
* note some big complete binary trees in need of balancing 

/usr/local/opticks/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0::


    total 6168
    -rw-r--r--  1 blyth  staff  -  692248 Jun 26 13:18 GParts.txt
    -rw-r--r--  1 blyth  staff  -    5104 Jun 26 13:18 primBuffer.npy
    -rw-r--r--  1 blyth  staff  - 2452944 Jun 26 13:18 partBuffer.npy

    epsilon:0 blyth$ wc -l GParts.txt 
       38326 GParts.txt

    In [1]: a = np.load("primBuffer.npy")   ## start part and complete binary tree size, note some big ones in need of balancing 

    In [2]: a.shape
    Out[2]: (314, 4)

    In [8]: a[:100]
    Out[8]: 
    array([[    0,     1,     0,     0],
           [    1,     1,     0,     0],
           [    2,     1,     0,     0],
           [    3,     7,     0,     0],
           [   10,     7,     0,     0],
           [   17,     7,     0,     0],
           [   24,     7,     0,     0],
           [   31,     7,     0,     0],
           [   38,     3,     0,     0],
           [   41,     1,     0,     0],
           [   42,     1,     0,     0],
           [   43,     3,     0,     0],
           [   46,     1,     0,     0],
           [   47,     1,     0,     0],
           [   48,     3,     0,     0],
           [   51,     1,     0,     0],
           [   52,     1,     0,     0],
           [   53,     3,     0,     0],
           [   56,     1,     0,     0],
           [   57,     1,     0,     0],
           [   58,     3,     0,     0],
           [   61,     1,     0,     0],
           [   62,     1,     0,     0],
           [   63,     3,     0,     0],
           [   66,     1,     0,     0],
           [   67,     1,     0,     0],
           [   68,    63,     0,     0],
           [  131,    63,     0,     0],
           [  194,    63,     0,     0],
           [  257,    63,     0,     0],
           [  320,     1,     0,     0],
           [  321,     1,     0,     0],
           [  322,     3,     0,     0],
           [  325,     3,     0,     0],
           [  328,     3,     0,     0],
           [  331,     3,     0,     0],
           [  334,     3,     0,     0],
           [  337,     1,     0,     0],
           [  338,     1,     0,     0],
           [  339,     1,     0,     0],
           [  340,  2047,     0,     0],   
           [ 2387,     7,     0,     0],
           [ 2394,     7,     0,     0],
           [ 2401,     7,     0,     0],
           [ 2408,  2047,     0,     0],
           [ 4455,     7,     0,     0],
           [ 4462,     1,     0,     0],

    In [17]: a[-5:]
    Out[17]: 
    array([[38311,     3,     0,     0],
           [38314,     3,     0,     0],
           [38317,     3,     0,     0],
           [38320,     3,     0,     0],
           [38323,     3,     0,     0]], dtype=int32)


    In [3]: b = np.load("partBuffer.npy")

    In [4]: b.shape
    Out[4]: (38326, 4, 4)


    In [12]: a[:,0].min(), a[:,0].max()
    Out[12]: (0, 38323)

    In [14]: b = np.load("partBuffer.npy")

    In [15]: b.shape
    Out[15]: (38326, 4, 4)



z,w are tran and plan offsets::

     03 #include "quad.h"
      4 
      5 struct Prim
      6 {
      7     __device__ int partOffset() const { return  q0.i.x ; }
      8     __device__ int numParts()   const { return  q0.i.y < 0 ? -q0.i.y : q0.i.y ; }
      9     __device__ int tranOffset() const { return  q0.i.z ; }
     10     __device__ int planOffset() const { return  q0.i.w ; }
     11     __device__ int primFlag()   const { return  q0.i.y < 0 ? CSG_FLAGPARTLIST : CSG_FLAGNODETREE ; }
     12 
     13     quad q0 ;
     14 
     15 };



::

    #/usr/local/opticks/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/2

    -rw-r--r--  1 blyth  staff  -  724 Jun 26 13:18 GParts.txt
    -rw-r--r--  1 blyth  staff  -  160 Jun 26 13:18 primBuffer.npy
    -rw-r--r--  1 blyth  staff  - 2704 Jun 26 13:18 partBuffer.npy

    epsilon:2 blyth$ wc -l GParts.txt 
          41 GParts.txt

    epsilon:2 blyth$ np.py primBuffer.npy  ## primBuffer is "index" into the multiple complete binary trees in the partBuffer 
    (5, 4)
    f32
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    u32
    [[ 0 15  0  0]
     [15 15  0  0]
     [30  7  0  0]
     [37  3  0  0]
     [40  1  0  0]]

    epsilon:2 blyth$ np.py partBuffer.npy
    (41, 4, 4)
    f32




Size of the partBuffer for the global merge seems a bit alarming initially, 
but on reflection it is not such a big deal as the intersect CUDA programs 
have a primIdx argument that always focusses on a single one of those
complete binary trees.  

* The alarming thing is the size of some of the binary trees.

::

    249 RT_PROGRAM void intersect(int primIdx)
    250 {
    251     const Prim& prim    = primBuffer[primIdx];
    252 
    253     unsigned partOffset  = prim.partOffset() ;
    254     unsigned numParts    = prim.numParts() ;
    255     unsigned primFlag    = prim.primFlag() ;
    256 
    257     uint4 identity = identityBuffer[instance_index] ;
    258 
    259     if(primFlag == CSG_FLAGNODETREE)
    260     {
    261         Part pt0 = partBuffer[partOffset + 0] ;
    262 
    263         identity.z = pt0.boundary() ;        // replace placeholder zero with test analytic geometry root node boundary
    264 
    265         evaluative_csg( prim, identity );
    266         //intersect_csg( prim, identity );
    267 
    268     }
    269     else if(primFlag == CSG_FLAGINVISIBLE)
    270     {
    271         // do nothing : report no intersections for primitives marked with primFlag CSG_FLAGINVISIBLE 
    272     }




Current limit is height 7 corresponding to 255 nodes, which some 
of the prims exceed (probably the cause of missing geometry in the X4 ?)::

     544 static __device__
     545 void evaluative_csg( const Prim& prim, const uint4& identity )
     546 {
     547     unsigned partOffset = prim.partOffset() ;
     548     unsigned numParts   = prim.numParts() ;
     549     unsigned tranOffset = prim.tranOffset() ;
     550 
     551     unsigned height = TREE_HEIGHT(numParts) ; // 1->0, 3->1, 7->2, 15->3, 31->4 
     552 
     553 #ifdef USE_TWIDDLE_POSTORDER
     554     // bit-twiddle postorder limited to height 7, ie maximum of 0xff (255) nodes
     555     // (using 2-bytes with PACK2 would bump that to 0xffff (65535) nodes)
     556     // In any case 0xff nodes are far more than this is expected to be used with
     557     //
     558     if(height > 7)
     559     {
     560         rtPrintf("evaluative_csg tranOffset %u numParts %u perfect tree height %u exceeds current limit\n", tranOffset, numParts, height ) ;
     561         return ;
     562     }
     563 #else
     564     // pre-baked postorder limited to height 3 tree,  ie maximum of 0xf nodes
     565     // by needing to stuff the postorder sequence 0x137fe6dc25ba498ull into 64 bits 
     566     if(height > 3)
     567     {
     568         rtPrintf("evaluative_csg tranOffset %u numParts %u perfect tree height %u exceeds current limit\n", tranOffset, numParts, height ) ;
     569         return ;
     570     }
     571     const unsigned long long postorder_sequence[4] = { 0x1ull, 0x132ull, 0x1376254ull, 0x137fe6dc25ba498ull } ;
     572     unsigned long long postorder = postorder_sequence[height] ;
     573 #endif






Single Primitive
------------------


A single primitive part is shaped (4,4) containing 
float parameters, integer codes and bounding box info 




::

    2017-01-05 20:52:00.698 INFO  [748507] [GParts::makeSolidBuffer@328] GParts::solidify i   0 nodeIndex   0
    2017-01-05 20:52:00.698 INFO  [748507] [GParts::makeSolidBuffer@328] GParts::solidify i   1 nodeIndex   1
    2017-01-05 20:52:00.698 INFO  [748507] [GParts::makeSolidBuffer@328] GParts::solidify i   2 nodeIndex   2
    2017-01-05 20:52:00.698 INFO  [748507] [GParts::dump@569] GParts::dump OGeo::makeAnalyticGeometry pts
    2017-01-05 20:52:00.698 INFO  [748507] [GParts::dumpSolidInfo@385] OGeo::makeAnalyticGeometry pts (part_offset, parts_for_solid, solid_index, 0) numSolids:3
    2017-01-05 20:52:00.698 INFO  [748507] [GParts::dumpSolidInfo@390]  (  0,  1,  0,  0) 
    2017-01-05 20:52:00.698 INFO  [748507] [GParts::dumpSolidInfo@390]  (  1,  1,  1,  0) 
    2017-01-05 20:52:00.698 INFO  [748507] [GParts::dumpSolidInfo@390]  (  2,  1,  2,  0) 
    2017-01-05 20:52:00.698 INFO  [748507] [GParts::dump@581] GParts::dump ni 3
         0.0000      0.0000      0.0000   1200.0000 
         0.0000       0 id       123 bnd        0 flg   bn Rock//perfectAbsorbSurface/Vacuum 
     -1200.0100  -1200.0100  -1200.0100           3 (Box) 
      1200.0100   1200.0100   1200.0100           0 (nodeIndex) 

         0.0000      0.0000   -600.0000    641.2000 
         0.0000       1 id       124 bnd        0 flg   bn Vacuum///MainH2OHale 
      -226.1460   -226.1460     -0.0100           1 (Sphere) 
       226.1460    226.1460     41.2100           1 (nodeIndex) 

         0.0000      0.0000    600.0000    641.2000 
         0.0000       2 id       124 bnd        0 flg   bn Vacuum///MainH2OHale 
      -226.1460   -226.1460    -41.2100           1 (Sphere) 
       226.1460    226.1460      0.0100           2 (nodeIndex) 

    2017-01-05 20:52:00.699 INFO  [748507] [NPY<float>::dump@1088] OGeo::makeAnalyticGeometry partBuf (3,4,4) 

    (  0)       0.000       0.000       0.000    1200.000 
    (  0)       0.000       0.000       0.000       0.000 
    (  0)   -1200.010   -1200.010   -1200.010       0.000 
    (  0)    1200.010    1200.010    1200.010       0.000 
    (  1)       0.000       0.000    -600.000     641.200 
    (  1)       0.000       0.000       0.000       0.000 
    (  1)    -226.146    -226.146      -0.010       0.000 
    (  1)     226.146     226.146      41.210       0.000 
    (  2)       0.000       0.000     600.000     641.200 
    (  2)       0.000       0.000       0.000       0.000 
    (  2)    -226.146    -226.146     -41.210       0.000 
    (  2)     226.146     226.146       0.010       0.000 
    2017-01-05 20:52:00.699 INFO  [748507] [int>::dump@1088] OGeo::makeAnalyticGeometry solidBuf partOffset/numParts/solidIndex/0 (3,4) 

    (  0)           0           1           0           0 
    (  1)           1           1           1           0 
    (  2)           2           1           2           0 

