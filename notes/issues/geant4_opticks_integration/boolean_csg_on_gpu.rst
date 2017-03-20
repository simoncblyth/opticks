Boolean CSG on GPU
===================


TODO: numerical/chi2 history comparison with CFG4 booleans 
------------------------------------------------------------

TODO : python trees into opticks
----------------------------------

new way without container::

    2017-03-20 17:52:45.163 INFO  [525230] [GSolid::Dump@172] GMergedMesh::combine (source solids) numSolid 2
    2017-03-20 17:52:45.163 INFO  [525230] [GNode::dump@180] GNode::dump
    GNode::dump idx 0 nchild 0 
    2017-03-20 17:52:45.163 INFO  [525230] [GNode::dump@187] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000      0.000 
    2017-03-20 17:52:45.163 INFO  [525230] [GNode::dump@180] GNode::dump
    GNode::dump idx 1 nchild 0 
    2017-03-20 17:52:45.163 INFO  [525230] [GNode::dump@187] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000    186.383 
    2017-03-20 17:52:45.163 INFO  [525230] [GMesh::allocate@604] GMesh::allocate numVertices 2520 numFaces 840 numSolids 2
    2017-03-20 17:52:45.163 INFO  [525230] [GMesh::allocate@635] GMesh::allocate DONE 
    2017-03-20 17:52:45.163 INFO  [525230] [GMergedMesh::dumpSolids@606] GMergedMesh::combine (combined result)  ce0 gfloat4      0.000      0.000      0.000    186.383 
        0 ce             gfloat4      0.000      0.000      0.000    186.383  bb bb min   -186.383   -186.383   -186.383  max    186.383    186.383    186.383 
        1 ce             gfloat4      0.000      0.000      0.000    186.383  bb bb min   -186.383   -186.383   -186.383  max    186.383    186.383    186.383 
        0 ni[nf/nv/nidx/pidx] (  0,  0,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,  0,123,  0) 
        1 ni[nf/nv/nidx/pidx] (840,2520,  1,4294967295)  id[nidx,midx,bidx,sidx]  (  1,  1,124,  0) 
    2017-03-20 17:52:45.163 INFO  [525230] [GParts::Summary@580] GGeoTest::combineSolids --dbganalytic num_parts 4 num_prim 0


old way with container::


    2017-03-20 17:50:05.320 INFO  [524398] [GSolid::Dump@172] GMergedMesh::combine (source solids) numSolid 4
    2017-03-20 17:50:05.320 INFO  [524398] [GNode::dump@180] GNode::dump
    GNode::dump idx 0 nchild 0 
    2017-03-20 17:50:05.320 INFO  [524398] [GNode::dump@187] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000   1000.000 
    2017-03-20 17:50:05.320 INFO  [524398] [GNode::dump@180] GNode::dump
    GNode::dump idx 0 nchild 0 
    2017-03-20 17:50:05.320 INFO  [524398] [GNode::dump@187] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000    500.000 
    2017-03-20 17:50:05.320 INFO  [524398] [GNode::dump@180] GNode::dump
    GNode::dump idx 0 nchild 0 
    2017-03-20 17:50:05.320 INFO  [524398] [GNode::dump@187] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000    150.000 
    2017-03-20 17:50:05.320 INFO  [524398] [GNode::dump@180] GNode::dump
    GNode::dump idx 0 nchild 0 
    2017-03-20 17:50:05.320 INFO  [524398] [GNode::dump@187] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000      1.000 
    2017-03-20 17:50:05.320 INFO  [524398] [GMesh::allocate@604] GMesh::allocate numVertices 3912 numFaces 1316 numSolids 4
    2017-03-20 17:50:05.320 INFO  [524398] [GMesh::allocate@635] GMesh::allocate DONE 
    2017-03-20 17:50:05.320 FATAL [524398] [GMergedMesh::mergeSolid@460] GMergedMesh::mergeSolid mismatch  nodeIndex 0 m_cur_solid 1
    2017-03-20 17:50:05.320 FATAL [524398] [GMergedMesh::mergeSolid@460] GMergedMesh::mergeSolid mismatch  nodeIndex 0 m_cur_solid 2
    2017-03-20 17:50:05.320 FATAL [524398] [GMergedMesh::mergeSolid@460] GMergedMesh::mergeSolid mismatch  nodeIndex 0 m_cur_solid 3
    2017-03-20 17:50:05.321 INFO  [524398] [GMergedMesh::dumpSolids@606] GMergedMesh::combine (combined result)  ce0 gfloat4      0.000      0.000      0.000   1000.000 
        0 ce             gfloat4      0.000      0.000      0.000   1000.000  bb bb min  -1000.000  -1000.000  -1000.000  max   1000.000   1000.000   1000.000 
        1 ce             gfloat4      0.000      0.000      0.000    500.000  bb bb min   -500.000   -500.000   -500.000  max    500.000    500.000    500.000 
        2 ce             gfloat4      0.000      0.000      0.000    150.000  bb bb min   -150.000   -150.000   -150.000  max    150.000    150.000    150.000 
        3 ce             gfloat4      0.000      0.000      0.000    200.000  bb bb min   -200.000   -200.000   -200.000  max    200.000    200.000    200.000 
        0 ni[nf/nv/nidx/pidx] ( 12, 24,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,  0,  0,  0) 
        1 ni[nf/nv/nidx/pidx] ( 12, 24,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,  0,  0,  0) 
        2 ni[nf/nv/nidx/pidx] ( 12, 24,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,  0,  0,  0) 
        3 ni[nf/nv/nidx/pidx] (1280,3840,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,  0,  0,  0) 
    2017-03-20 17:50:05.321 INFO  [524398] [GParts::Summary@580] GGeoTest::combineSolids --dbganalytic num_parts 4 num_prim 0
     part  0 : node  0 type  6 boundary [123] Rock//perfectAbsorbSurface/Vacuum  
     part  1 : node  1 type  2 boundary [124] Vacuum///GlassSchottF2  
     part  2 : node  1 type  6 boundary [124] Vacuum///GlassSchottF2  
     part  3 : node  1 type  5 boundary [124] Vacuum///GlassSchottF2  






TODO : randbox photon source
------------------------------

* very handy python side for intersect testing, implement in CUDA too 
* also some option to load python generated rays into OptiX would allow
  same rays testing 


TODO : more natural test tree construction 
---------------------------------------------

* authoring CSG trees in levelorder is difficult, find more natural approach


TODO: support transforms in op nodes
-----------------------------------------

* G4 boolean transforms apply to right side only, follow that.


TODO: support empties, ie non-perfect trees
-----------------------------------------------

TODO : algorithm characteristic comparisons
---------------------------------------------

Extended period of boolean CSG debugging has resulted
in several CUDA/OptiX and python implementations 

Need to compare their characteristics

* maximum csg and tranche stack size
* performance


Boolean Debugging
-------------------

FIXED : Root Termination problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: ERROR_XOR_SIDE, ERROR_POP_EMPTY only for very overlapped height 3 trees

The problem was an bug in the translation of the recursive CSG algorithm into 
an iterative version for use on GPU.
The bug resulted in the root being visited twice for very overlapped geometries. 

Reiteration within another reiteration needs to terminate at the caller not at the root.


slavish.py
~~~~~~~~~~~

Using 10k rays generated in a random box, succeed to get a few errors python side, made reproducible by setting np.random.seed.

::

    slavish.py
    [2017-03-08 13:45:35,836] p62090 {/Users/blyth/opticks/dev/csg/intersectTest.py:180} INFO - np.random.seed 0 
    ierr: 0x00001008 iray:   785 
    ierr: 0x00001008 iray:  1972 
    ierr: 0x00001008 iray:  3546 
    ierr: 0x00001008 iray:  7119 
    ierr: 0x00001008 iray:  7325 
    ierr: 0x00001008 iray:  8894 
    [2017-03-08 13:45:53,221] p62090 {/Users/blyth/opticks/dev/csg/intersectTest.py:229} WARNING - $TMP/tboolean-csg-four-box-minus-sphere : compare : i_discrep {'d': IIS([ 785,  785,  785, 1972, 1972, 1972, 3546, 3546, 3546, 7119, 7119, 7119, 7325, 7325, 7325, 8894, 8894, 8894]), 'ipos': IIS([ 785,  785,  785, 1972, 1972, 1972, 3546, 3546, 3546, 7119, 7119, 7119, 7325, 7325, 7325, 8894, 8894, 8894]), 't': array([ 785, 1972, 3546, 7119, 7325, 8894]), 'o': IIS([ 785,  785,  785, 1972, 1972, 1972, 3546, 3546, 3546, 7119, 7119, 7119, 7325, 7325, 7325, 8894, 8894, 8894]), 'n': IIS([ 785,  785,  785, 1972, 1972, 1972, 3546, 3546, 3546, 7119, 7119, 7119, 7325, 7325, 7325, 8894, 8894, 8894])} r_discrep: {}  


Cause is same for all, iterative misses loopers that recursive does::

    [2017-03-08 14:34:06,284] p62256 {/Users/blyth/opticks/dev/csg/intersectTest.py:185} INFO - np.random.seed 0 
    [2017-03-08 14:34:06,286] p62256 {/Users/blyth/opticks/dev/csg/slavish.py:267} INFO -    785 I : tranche begin 0 end 7 
       785 I : nodeIdx  4 
       785 I : nodeIdx  5 
       785 I : nodeIdx  2 
       785 I : nodeIdx  6 
       785 I : nodeIdx  7 
       785 I : nodeIdx  3 
       785 I : nodeIdx  1 
    ierr: 0x00001008 tst.iray:   785 

       785 R : nodeIdx  4 
       785 R : nodeIdx  5 
       785 R : nodeIdx  2 
       785 R : nodeIdx  6 
       785 R : nodeIdx  7 
       785 R : nodeIdx  3 
       785 R : nodeIdx  6*   REPEAT RIGHT SUBTREE 
       785 R : nodeIdx  7* 
       785 R : nodeIdx  3* 
       785 R : nodeIdx  1 

    [2017-03-08 14:34:06,289] p62256 {/Users/blyth/opticks/dev/csg/slavish.py:267} INFO -   1972 I : tranche begin 0 end 7 
      1972 I : nodeIdx  4 
      1972 I : nodeIdx  5 
      1972 I : nodeIdx  2 
      1972 I : nodeIdx  6 
      1972 I : nodeIdx  7 
      1972 I : nodeIdx  3 
      1972 I : nodeIdx  1 
    ierr: 0x00001008 tst.iray:  1972 
      1972 R : nodeIdx  4 
      1972 R : nodeIdx  5 
      1972 R : nodeIdx  2 
      1972 R : nodeIdx  6 
      1972 R : nodeIdx  7 
      1972 R : nodeIdx  3 
      1972 R : nodeIdx  4*  REPEAT LEFT SUBTREE
      1972 R : nodeIdx  5* 
      1972 R : nodeIdx  2* 
      1972 R : nodeIdx  1 
    [2017-03-08 14:34:06,292] p62256 {/Users/blyth/opticks/dev/csg/slavish.py:267} INFO -   3546 I : tranche begin 0 end 7 
      3546 I : nodeIdx  4 
      3546 I : nodeIdx  5 
      3546 I : nodeIdx  2 
    ierr: 0x00001008 tst.iray:  3546 
      3546 R : nodeIdx  4 
      3546 R : nodeIdx  5 
      3546 R : nodeIdx  5*   REPEAT A BILEAF 
      3546 R : nodeIdx  2 
      3546 R : nodeIdx  6 
      3546 R : nodeIdx  7 
      3546 R : nodeIdx  3 
      3546 R : nodeIdx  1 
    [2017-03-08 14:34:06,295] p62256 {/Users/blyth/opticks/dev/csg/slavish.py:267} INFO -   7119 I : tranche begin 0 end 7 
      7119 I : nodeIdx  4 
      7119 I : nodeIdx  5    
      7119 I : nodeIdx  2 
      7119 I : nodeIdx  6 
      7119 I : nodeIdx  7 
      7119 I : nodeIdx  3 
      7119 I : nodeIdx  1 
    ierr: 0x00001008 tst.iray:  7119 
      7119 R : nodeIdx  4 
      7119 R : nodeIdx  5 
      7119 R : nodeIdx  2 
      7119 R : nodeIdx  6 
      7119 R : nodeIdx  7 
      7119 R : nodeIdx  3 
      7119 R : nodeIdx  4*
      7119 R : nodeIdx  5* 
      7119 R : nodeIdx  2* 
      7119 R : nodeIdx  1 





CSG Errors
~~~~~~~~~~~~~

Very overlapped geometry like : tboolean-csg-four-box-minus-sphere
gives errors, shown below. 
Dumping the launch_index and comparing between runs suggests the issue is reproducible.

Returning the improper 

::


     0x1008 -> 1008 -> ERROR_RHS_END_EMPTY 
     0x100c -> 100c -> ERROR_LHS_END_NONEMPTY ERROR_RHS_END_EMPTY 
           0x1 -> 1 -> ERROR_LHS_POP_EMPTY 


Origin shows not primary rays causing errors::

    2017-03-08 11:02:28.525 INFO  [457301] [OPropagator::prelaunch@149] 1 : (0;100000,1) prelaunch_times vali,comp,prel,lnch  0.0000 1.0982 0.1492 0.0000
    intersect_csg primIdx_ 1 ierr 1008 launch_index (  175,    0) li.x(26) 19 ray.direction (     0.865,    -0.354,    -0.354) ray.origin (   -50.111,   -37.211,    -4.933)   
    intersect_csg primIdx_ 1 ierr 100c launch_index (  249,    0) li.x(26) 15 ray.direction (    -0.000,     0.434,    -0.901) ray.origin (    35.866,   -53.215,    50.111)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index (  615,    0) li.x(26) 17 ray.direction (    -0.000,    -0.901,     0.434) ray.origin (    28.152,    50.111,     9.413)   
    intersect_csg primIdx_ 1 ierr 100c launch_index (   11,    0) li.x(26) 11 ray.direction (     0.434,    -0.000,    -0.901) ray.origin (    -6.774,    44.818,    50.111)   
    intersect_csg primIdx_ 1 ierr 100c launch_index (  323,    0) li.x(26) 11 ray.direction (     0.434,    -0.000,    -0.901) ray.origin (    -1.145,    31.434,    50.111)   
    intersect_csg primIdx_ 1 ierr 100c launch_index (  387,    0) li.x(26) 23 ray.direction (     0.354,    -0.865,     0.354) ray.origin (    42.450,    50.111,   -55.690)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index (  406,    0) li.x(26) 16 ray.direction (    -0.000,     0.901,     0.434) ray.origin (   -37.924,   -50.111,     0.866)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index (  397,    0) li.x(26)  7 ray.direction (     0.901,    -0.434,    -0.000) ray.origin (   -50.111,   -14.494,    17.463)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index ( 1286,    0) li.x(26) 12 ray.direction (     0.434,    -0.000,     0.901) ray.origin (  -158.749,   -45.161,   -50.111)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index (  207,    0) li.x(26) 25 ray.direction (     0.354,     0.354,     0.865) ray.origin (  -146.598,   -51.685,   -50.111)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index (  584,    0) li.x(26) 12 ray.direction (     0.901,    -0.000,     0.434) ray.origin (   -50.111,   -16.444,    17.319)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index (  662,    0) li.x(26) 12 ray.direction (     0.901,    -0.000,     0.434) ray.origin (   -50.111,   -17.234,    15.378)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index (  666,    0) li.x(26) 16 ray.direction (    -0.000,     0.901,     0.434) ray.origin (   -25.323,   -50.111,     1.325)   
    intersect_csg primIdx_ 1 ierr    1 launch_index ( 1325,    0) li.x(26) 25 ray.direction (     0.354,     0.865,     0.354) ray.origin (    31.793,   -50.111,   -10.657)   
    intersect_csg primIdx_ 1 ierr 100c launch_index ( 1519,    0) li.x(26) 11 ray.direction (     0.434,    -0.000,    -0.901) ray.origin (    10.308,    21.809,    50.111)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index (   99,    0) li.x(26) 21 ray.direction (    -0.354,    -0.865,     0.354) ray.origin (    52.533,   150.111,   -37.067)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index ( 1968,    0) li.x(26) 18 ray.direction (    -0.865,    -0.354,    -0.354) ray.origin (    50.111,   -41.536,    21.572)   
    intersect_csg primIdx_ 1 ierr 1008 launch_index (  967,    0) li.x(26)  5 ray.direction (    -0.000,    -0.000,     1.000) ray.origin (   -47.721,   -40.248,  -250.111)   
    intersect_csg primIdx_ 1 ierr 100c launch_index (  141,    0) li.x(26) 11 ray.direction (     0.434,    -0.000,    -0.901) ray.origin (    26.544,     3.120,    50.111)   
    intersect_csg primIdx_ 1 ierr 100c launch_index (  985,    0) li.x(26) 23 ray.direction (     0.779,    -0.007,     0.627) ray.origin (    38.651,    13.330,   -10.936)   

::

    intersect_csg primIdx_ 1 ierr 1008 tloop   0 launch_index ( 1005,  365) li.x(26) 17 ray.direction (    -0.990,    -0.111,     0.089) ray.origin (    80.850,   -27.053,   -58.984)   
    intersect_csg primIdx_ 1 ierr 1008 tloop   0 launch_index ( 1006,  365) li.x(26) 18 ray.direction (    -0.990,    -0.110,     0.089) ray.origin (    80.850,   -27.053,   -58.984)   
    intersect_csg primIdx_ 1 ierr 1008 tloop   0 launch_index ( 1007,  365) li.x(26) 19 ray.direction (    -0.990,    -0.109,     0.089) ray.origin (    80.850,   -27.053,   -58.984)   
    intersect_csg primIdx_ 1 ierr 1008 tloop   0 launch_index ( 1004,  367) li.x(26) 16 ray.direction (    -0.990,    -0.112,     0.091) ray.origin (    80.850,   -27.053,   -58.984)   
    intersect_csg primIdx_ 1 ierr 1008 tloop   0 launch_index ( 1005,  367) li.x(26) 17 ray.direction (    -0.990,    -0.111,     0.091) ray.origin (    80.850,   -27.053,   -58.984)   
    intersect_csg primIdx_ 1 ierr 1008 tloop   0 launch_index ( 1006,  367) li.x(26) 18 ray.direction (    -0.990,    -0.110,     0.091) ray.origin (    80.850,   -27.053,   -58.984)   

    PRINT BUFFER -1 OVERFLOW
    intersect_csg primIdx_ 1 ierr    1 tloop   2 launch_index (  920,  383) li.x(26) 10 ray.direction (    -0.978,    -0.184,     0.102) ray.origin (    82.681,   -27.666,   -60.320)   
    intersect_csg primIdx_ 1 ierr    1 tloop   2 launch_index (  921,  383) li.x(26) 11 ray.direction (    -0.978,    -0.183,     0.102) ray.origin (    82.681,   -27.666,   -60.320)   
    intersect_csg primIdx_ 1 ierr    1 tloop   2 launch_index (  922,  383) li.x(26) 12 ray.direction (    -0.978,    -0.182,     0.102) ray.origin (    82.681,   -27.666,   -60.320)   
    intersect_csg primIdx_ 1 ierr    1 tloop   2 launch_index (  923,  383) li.x(26) 13 ray.direction (    -0.978,    -0.182,     0.102) ray.origin (    82.681,   -27.666,   -60.320)   
    intersect_csg primIdx_ 1 ierr    1 tloop   2 launch_index (  924,  383) li.x(26) 14 ray.direction (    -0.978,    -0.181,     0.102) ray.origin (    82.681,   -27.666,   -60.320)   
    intersect_csg primIdx_ 1 ierr    1 tloop   2 launch_index (  925,  383) li.x(26) 15 ray.direction (    -0.978,    -0.180,     0.102) ray.origin (    82.681,   -27.666,   -60.320)   
    intersect_csg primIdx_ 1 ierr    1 tloop   2 launch_index (  926,  383) li.x(26) 16 ray.direction (    -0.978,    -0.179,     0.102) ray.origin (    82.681,   -27.666,   -60.320)   




DONE: boolean csg tree implementation
--------------------------------------


OptiX array
~~~~~~~~~~~~~


Hmm seems everything other than very simple things need to go into buffers.

* https://devtalk.nvidia.com/default/topic/966684/optix/array-program-variables/


C : Two meanings of static
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* static global variables and functions, scope limited to definining file
* static local variables, typically use compile time reserved data segment of memory 
  rather than transient call stack


CUDA guide : static local variables within function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
* http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#static-variables-function

Within the body of a __device__ or __global__ function, only __shared__
variables or variables without any device memory qualifiers may be declared
with static storage class. 

Within the body of a __device__ __host__ function, only unannotated 
static variables (i.e., without device memory qualifiers) may
be declared with static storage class. Unannotated function-scope static
variables have the same restrictions as __device__ variables defined in
namespace scope. They cannot have a non-empty constructor or a non-empty
destructor, if they are of class type (see Device Memory Qualifiers).

* hmm, this explains why I had to remove ctors/dtors in my simple structs

::

    struct S1_t { int x; }; 
    struct S2_t { int x; __device__ S2_t(void) { x = 10; } }; 
    struct S3_t { int x; __device__ S3_t(int p) : x(p) { } }; 
    __device__ void f1() { 
             static int i1; // OK 
             static int i2 = 11; // OK 
             static S1_t i3; // OK 
             static S1_t i4 = {22}; // OK 
             static __shared__ int i5; // OK 
             int x = 33; 
             static int i6 = x; // error: dynamic initialization is not allowed 
             static S1_t i7 = {x}; // error: dynamic initialization is not allowed 
             static S2_t i8; // error: dynamic initialization is not allowed 
             static S3_t i9(44); // error: dynamic initialization is not allowed
    }

* restriction to non-dynamic static local variables in device kernels
  makes sense, otherwise each of the millions of threads would need it own data segment

* With compile time defined restriction can just have one used for all threads


OptiX/CUDA static variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :google:`cuda static variable`

NB the below is an example of dynamic use of local static variables, so can only work host side.

/Developer/OptiX/SDK/optixTutorial/random.h:: 

     69 // Multiply with carry
     70 static __host__ __inline__ unsigned int mwc()
     71 {
     72   static unsigned long long r[4];
     73   static unsigned long long carry;
     74   static bool init = false;
     75   if( !init ) {
     76     init = true;
     77     unsigned int seed = 7654321u, seed0, seed1, seed2, seed3;
     78     r[0] = seed0 = lcg2(seed);
     79     r[1] = seed1 = lcg2(seed0);
     80     r[2] = seed2 = lcg2(seed1);
     81     r[3] = seed3 = lcg2(seed2);
     82     carry = lcg2(seed3);
     83   }
     84 
     85   unsigned long long sum = 2111111111ull * r[3] +
     86                            1492ull       * r[2] +
     87                            1776ull       * r[1] +
     88                            5115ull       * r[0] +
     89                            1ull          * carry;
     90   r[3]   = r[2];
     91   r[2]   = r[1];
     92   r[1]   = r[0];
     93   r[0]   = static_cast<unsigned int>(sum);        // lower half
     94   carry  = static_cast<unsigned int>(sum >> 32);  // upper half
     95   return static_cast<unsigned int>(r[0]);
     96 }





Adding node transforms
~~~~~~~~~~~~~~~~~~~~~~~~

Matrix manip, optixu_matrix_namespace.h


OptiX : const float3
~~~~~~~~~~~~~~~~~~~~~~~

::

    2112 OPTIXU_INLINE RT_HOSTDEVICE float luminanceCIE(const float3& rgb)
    2113 {
    2114   const float3 cie_luminance = { 0.2126f, 0.7152f, 0.0722f };
    2115   return  dot( rgb, cie_luminance );
    2116 }



OptiX float4 as a very short stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:optixu blyth$ grep ByIndex optixu_math_namespace.h
    OPTIXU_INLINE RT_HOSTDEVICE float getByIndex(const float1& v, int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(float1& v, int i, float x)
    OPTIXU_INLINE RT_HOSTDEVICE float getByIndex(const float2& v, int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(float2& v, int i, float x)
    OPTIXU_INLINE RT_HOSTDEVICE float getByIndex(const float3& v, int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(float3& v, int i, float x)
    OPTIXU_INLINE RT_HOSTDEVICE float getByIndex(const float4& v, int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(float4& v, int i, float x)
    OPTIXU_INLINE RT_HOSTDEVICE int getByIndex(const int1& v, int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(int1& v, int i, int x)
    OPTIXU_INLINE RT_HOSTDEVICE int getByIndex(const int2& v, int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(int2& v, int i, int x)
    OPTIXU_INLINE RT_HOSTDEVICE int getByIndex(const int3& v, int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(int3& v, int i, int x)
    OPTIXU_INLINE RT_HOSTDEVICE int getByIndex(const int4& v, int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(int4& v, int i, int x)
    OPTIXU_INLINE RT_HOSTDEVICE unsigned int getByIndex(const uint1& v, unsigned int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(uint1& v, int i, unsigned int x)
    OPTIXU_INLINE RT_HOSTDEVICE unsigned int getByIndex(const uint2& v, unsigned int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(uint2& v, int i, unsigned int x)
    OPTIXU_INLINE RT_HOSTDEVICE unsigned int getByIndex(const uint3& v, unsigned int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(uint3& v, int i, unsigned int x)
    OPTIXU_INLINE RT_HOSTDEVICE unsigned int getByIndex(const uint4& v, unsigned int i)
    OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(uint4& v, int i, unsigned int x)



Lookup tables in C
~~~~~~~~~~~~~~~~~~~~

* :google:`C lookup table`

Perfect tree traversal has lots of constants, also boolean_act and boolean_table 
decision logic has lots of if statements with a small 
range of input values. 

This kinda thing seems suited to small static lookup tables, to avoid computation
every time. Of course with CUDA its not at all sure there will be any benefit, as GPUs
favor computation over memory access.

* http://embeddedgurus.com/stack-overflow/2010/01/a-tutorial-on-lookup-tables-in-c/

* http://stackoverflow.com/questions/17088484/cuda-memory-for-lookup-tables

  This is talking about 4KB lookup tables, the ones I have in mind are miniscule

* http://www.marekfiser.com/Projects/Conways-Game-of-Life-on-GPU-using-CUDA/4-Advanced-lookup-table-implementation



Whats missing for opticks csg tree ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* postorder tree threading, leftmost operator starting point 
* stack of float4(quad) for tranches, holding tmin and begin/end tree indices
* stack of float4 holding normal and t 



Needs to be almost complete tree anyhow for easy serializing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* so postorder can be hardcoded for different tree depths


depth 1, triplet::


    In [21]: Node.postorder_r(root1, nodes=[])
    Out[21]: [s2.s, s3.s, I1.Intersection(s2.s,s3.s)]

    In [22]: root1.txt
    Out[22]: 
    root1            
         I1        
          o        
     s2      s3    
      o       o    



depth 2, septuplet::

    In [15]: Node.postorder_r(root2, nodes=[])
    Out[15]: 
    [s4.s,
     s5.s,
     I2.Intersection(s4.s,s5.s),
     s6.s,
     s7.s,
     I3.Intersection(s6.s,s7.s),
     U1.Union(I2.Intersection(s4.s,s5.s),I3.Intersection(s6.s,s7.s))]

    In [16]: root2.txt
    Out[16]: 
    root2                            
                 U1                
                  o                
         I2              I3        
          o               o        
     s4      s5      s6      s7    
      o       o       o       o    
                                   

depth 3, 15-tuplet::

    In [17]: Node.postorder_r(root3, nodes=[])
    Out[17]: 
    [s8.s,                            i  = 8
     s9.s,                            i+1 = 9                  add 1 to get to right sibling 
     I4.Intersection(s8.s,s9.s),      (i+1)/2 = 4              divide by 2, up to parent 
     s10.s,                           ( (i+1)/2) + 1)*2 = 10   add 1, multip by 2 
     s11.s,                           ((i/2) + 1)*2 + 1 = 11
     I5.Intersection(s10.s,s11.s),     
     U2.Union(I4.Intersection(s8.s,s9.s),I5.Intersection(s10.s,s11.s)),
     s12.s,
     s13.s,
     I6.Intersection(s12.s,s13.s),
     s14.s,
     s15.s,
     I7.Intersection(s14.s,s15.s),
     U3.Union(I6.Intersection(s12.s,s13.s),I7.Intersection(s14.s,s15.s)),
     U1.Union(U2.Union(I4.Intersection(s8.s,s9.s),I5.Intersection(s10.s,s11.s)),U3.Union(I6.Intersection(s12.s,s13.s),I7.Intersection(s14.s,s15.s)))]

    In [18]: root3.txt
    Out[18]: 
    root3                                                            
                                 U1                                
                                  o                                
                 U2                              U3                
                  o                               o                
         I4              I5              I6              I7        
          o               o               o               o        
     s8      s9     s10     s11     s12     s13     s14     s15    
      o       o       o       o       o       o       o       o    
                                                                   

*  4, 5, 2, 6, 7, 3, 1

* unsigned long long postorder_depth3 = 0x1376254    (64 bits) 


Simpler to fly above the leaves::

    In [26]: Node.postorder_r(root3, nodes=[], leaf=False)
    Out[26]: 
    [I4.Intersection(s8.s,s9.s),
     I5.Intersection(s10.s,s11.s),
     U2.Union(I4.Intersection(s8.s,s9.s),I5.Intersection(s10.s,s11.s)),
     I6.Intersection(s12.s,s13.s),
     I7.Intersection(s14.s,s15.s),
     U3.Union(I6.Intersection(s12.s,s13.s),I7.Intersection(s14.s,s15.s)),
     U1.Union(U2.Union(I4.Intersection(s8.s,s9.s),I5.Intersection(s10.s,s11.s)),U3.Union(I6.Intersection(s12.s,s13.s),I7.Intersection(s14.s,s15.s)))]






* If T has a total of N nodes, the number of internal nodes is I = (N – 1)/2 
* 
*        1 + 2 + 4 + 8 + ... + 2^d = tot_d
*  1 + ( 2 + 4 + 8 + 16 + ... + 2^d ) + 2^(d+1) = 1 + 2*tot_d 
*  tot_d + 2^(d+1) = 1 + 2*tot_d
*   tot_d = 2^(d+1) - 1


* internal nodes,  [( 2^(d+1) - 1 ) - 1] / 2  ->  2^d - 1


* better to base things from the depth, as might want to support gaps on the last row

*  depth   number of nodes    number of leaves
*  d = 0,  2^1 - 1 = 1              
*  d = 1,  2^2 - 1 = 3        
*  d = 2,  2^3 - 1 = 7
*  d = 3,  2^4 - 1 = 15
*  d = 4,  2^5 - 1 = 31





Tree Threading ?
~~~~~~~~~~~~~~~~~~

* GCSG (which should probably be renamed GCSGPmt) does something similar
  using a NPY buffer (created in python) as the input

* most methods require an item index

::

     32 #include "GGEO_API_EXPORT.hh"
     33 class GGEO_API GCSG {
     34     public:
     ..
     62     public:
     63         unsigned int getNumItems();
     64     public:
     65         float getX(unsigned int i);
     66         float getY(unsigned int i);
     67         float getZ(unsigned int i);
     68         float getOuterRadius(unsigned int i);
     69         float getInnerRadius(unsigned int i);
     70         float getSizeZ(unsigned int i);
     71         float getStartTheta(unsigned int i);
     72         float getDeltaTheta(unsigned int i);
     73     public:
     74         unsigned int getTypeCode(unsigned int i);
     75         bool isUnion(unsigned int i);
     76         bool isIntersection(unsigned int i);
     77         bool isSphere(unsigned int i);
     78         bool isTubs(unsigned int i);
     79 
     80         unsigned int getNodeIndex(unsigned int i);  // 1-based index, 0:unset
     81         unsigned int getParentIndex(unsigned int i);  // 1-based index, 0:unset
     82         unsigned int getSpare(unsigned int i);
     83 
     84         const char* getTypeName(unsigned int i);
     85     public:
     86         unsigned int getIndex(unsigned int i);
     87         unsigned int getNumChildren(unsigned int i);
     88         unsigned int getFirstChildIndex(unsigned int i);
     89         unsigned int getLastChildIndex(unsigned int i);
     90     private:
     91         float        getFloat(unsigned int i, unsigned int j, unsigned int k);
     92         unsigned int getUInt(unsigned int i, unsigned int j, unsigned int k);
     93 
     94     private:
     95         NPY<float>*        m_csg_buffer ;
     96         GItemList*         m_materials ;
     97         GItemList*         m_lvnames ;
     98         GItemList*         m_pvnames ;




CsgInBox test geometry
~~~~~~~~~~~~~~~~~~~~~~~

::

    152 tboolean-csg-notes(){ cat << EON
    153 
    154 * CSG tree is defined in breadth first order
    155 
    156 * parameters of boolean operations currently define adhoc box 
    157   intended to contain the geometry, TODO: calculate from bounds of the contained tree 
    158 
    159 * offsets arg identifies which nodes belong to which primitives by pointing 
    160   at the nodes that start each primitive
    161 
    162 EON
    163 }
    164 
    165 tboolean-csg()
    166 {
    167     local material=$(tboolean-material)
    168     local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    169     local radius=200
    170 
    171     local test_config=(
    172                       mode=CsgInBox
    173                       analytic=1
    174                       offsets=0,1     ## 
    175 
    176                       node=box          parameters=0,0,0,1000          boundary=Rock//perfectAbsorbSurface/Vacuum
    177 
    178                       node=union        parameters=0,0,0,400           boundary=Vacuum///$material
    179                       node=difference   parameters=0,0,100,300         boundary=Vacuum///$material
    180                       node=difference   parameters=0,0,-100,300        boundary=Vacuum///$material
    181                       node=box          parameters=0,0,100,$inscribe   boundary=Vacuum///$material
    182                       node=sphere       parameters=0,0,100,$radius     boundary=Vacuum///$material
    183                       node=box          parameters=0,0,-100,$inscribe  boundary=Vacuum///$material
    184                       node=sphere       parameters=0,0,-100,$radius    boundary=Vacuum///$material
    185 
    186                       )
    187 
    188     echo "$(join _ ${test_config[@]})" 
    189 }



Where is the tree ?
~~~~~~~~~~~~~~~~~~~~


::

    278 bool GGeoTestConfig::isStartOfPrimitive(unsigned nodeIdx )
    279 {
    280     return std::find(m_offsets.begin(), m_offsets.end(), nodeIdx) != m_offsets.end() ;
    281 }


    237 GMergedMesh* GGeoTest::createCsgInBox()
    238 {
    239     std::vector<GSolid*> solids ;
    240     unsigned int n = m_config->getNumElements();
    241 
    242     unsigned numPrim = m_config->getNumOffsets();
    243     LOG(info) << "GGeoTest::createCsgInBox"
    244               << " nodes " << n
    245               << " numPrim " << numPrim
    246              ;
    247 
    248     int primIdx(-1) ;
    249 
    250     for(unsigned int i=0 ; i < n ; i++)
    251     {
    252         bool primStart = m_config->isStartOfPrimitive(i); // as identified by configured offsets
    253         if(primStart)
    254         {
    255             primIdx++ ;
    256         }
    ...
    284         GParts* pts = solid->getParts();
    285 
    286         pts->setIndex(0u, i);
    287         pts->setNodeIndex(0u, primIdx );
    288         pts->setFlags(0u, flags);
    289         pts->setBndLib(m_bndlib);
    290 
    291         solids.push_back(solid);
    292     }


::

     86 char GMaker::NodeCode(const char* nodename)
     87 {
     88     char sc = 'U' ;
     89     if(     strcmp(nodename, BOX) == 0)     sc = 'B' ;
     90     else if(strcmp(nodename, SPHERE) == 0)  sc = 'S' ;
     91     else if(strcmp(nodename, ZSPHERE) == 0) sc = 'Z' ;
     92     else if(strcmp(nodename, ZLENS) == 0)   sc = 'L' ;
     93     else if(strcmp(nodename, PMT) == 0)     sc = 'P' ;  // not operational
     94     else if(strcmp(nodename, PRISM) == 0)   sc = 'M' ;
     95     else if(strcmp(nodename, INTERSECTION) == 0)   sc = 'I' ;
     96     else if(strcmp(nodename, UNION) == 0)          sc = 'J' ;
     97     else if(strcmp(nodename, DIFFERENCE) == 0)     sc = 'K' ;
     98     return sc ;
     99 }


Tree serialization
~~~~~~~~~~~~~~~~~~~

::

    2017-03-01 15:31:06.796 INFO  [6205604] [GParts::dumpPrimInfo@530] OGeo::makeAnalyticGeometry pts (part_offset, parts_for_prim, prim_index, prim_flags) numPrim:2
    2017-03-01 15:31:06.796 INFO  [6205604] [GParts::dumpPrimInfo@535]  (  0,  1,  0, 16) 
    2017-03-01 15:31:06.796 INFO  [6205604] [GParts::dumpPrimInfo@535]  (  1,  7,  1,  4) 
    2017-03-01 15:31:06.796 INFO  [6205604] [GParts::dump@731] GParts::dump ni 8
         0.0000      0.0000      0.0000   1000.0000 
         0.0000       0 <-id       123 <-bnd       16 <-flg  SHAPE_PRIMITIVE   bn Rock//perfectAbsorbSurface/Vacuum 
     -1000.0100  -1000.0100  -1000.0100           3 (PART_BOX) 
      1000.0100   1000.0100   1000.0100           0 (nodeIndex) 

         0.0000      0.0000      0.0000    400.0000 
         0.0000       1 <-id       124 <-bnd        4 <-flg  SHAPE_UNION   bn Vacuum///GlassSchottF2 
      -400.0100   -400.0100   -400.0100           3 (PART_BOX) 
       400.0100    400.0100    400.0100           1 (nodeIndex) 

         0.0000      0.0000    100.0000    300.0000 
         0.0000       2 <-id       124 <-bnd        8 <-flg  SHAPE_DIFFERENCE   bn Vacuum///GlassSchottF2 
      -300.0100   -300.0100   -300.0100           3 (PART_BOX) 
       300.0100    300.0100    300.0100           1 (nodeIndex) 

         0.0000      0.0000   -100.0000    300.0000 
         0.0000       3 <-id       124 <-bnd        8 <-flg  SHAPE_DIFFERENCE   bn Vacuum///GlassSchottF2 
      -300.0100   -300.0100   -300.0100           3 (PART_BOX) 
       300.0100    300.0100    300.0100           1 (nodeIndex) 

         0.0000      0.0000    100.0000    150.1111 
         0.0000       4 <-id       124 <-bnd       16 <-flg  SHAPE_PRIMITIVE   bn Vacuum///GlassSchottF2 
      -150.1211   -150.1211   -150.1211           3 (PART_BOX) 
       150.1211    150.1211    150.1211           1 (nodeIndex) 

         0.0000      0.0000    100.0000    200.0000 
         0.0000       5 <-id       124 <-bnd       16 <-flg  SHAPE_PRIMITIVE   bn Vacuum///GlassSchottF2 
      -200.0100   -200.0100   -200.0100           1 (PART_SPHERE) 
       200.0100    200.0100    200.0100           1 (nodeIndex) 

         0.0000      0.0000   -100.0000    150.1111 
         0.0000       6 <-id       124 <-bnd       16 <-flg  SHAPE_PRIMITIVE   bn Vacuum///GlassSchottF2 
      -150.1211   -150.1211   -150.1211           3 (PART_BOX) 
       150.1211    150.1211    150.1211           1 (nodeIndex) 

         0.0000      0.0000   -100.0000    200.0000 
         0.0000       7 <-id       124 <-bnd       16 <-flg  SHAPE_PRIMITIVE   bn Vacuum///GlassSchottF2 
      -200.0100   -200.0100   -200.0100           1 (PART_SPHERE) 
       200.0100    200.0100    200.0100           1 (nodeIndex) 





FIXED Issue : ray trace "near/tmin" clipping fails to see inside booleans
---------------------------------------------------------------------------

* **FIXED BY STARTING boolean tA_min and tB_min at ray.tmin**

The usual behavior of near clipping enabling to see inside things is not working
with booleans when the viewpoint is outside the boolean.

As approach a boolean solid the near point preceeds you... when it reaches 
the solid a circular-ish black hole forms, this gets bigger as proceed 
onwards the black filling most of the frame until the viewpoint 
gets into the boolean primitive bbox(?) and suddenly the blackness changes into
a view of the insides. Once inside changing the near point works 
to clip how much of insides can see.


Tempted to use scene_epsilon in the below, but its not correct (or currently possible) 
for general intersection code to depend on a rendering only thing like scene_epsilon.

Begs the question how does non-boolean geometry manage to get near clipped ? 

* rays are shot with t_min set to scene_epsilon 


Exploring optix_device.h find ray.tmin, this might provide a solution::
    
    simon:include blyth$ grep tmin *.h
    optix_device.h:  optix::rt_trace(*(unsigned int*)&topNode, ray.origin, ray.direction, ray.ray_type, ray.tmin, ray.tmax, &prd, sizeof(T));
    optix_device.h:  * @param[in] tmin  t value of the ray to be checked
    optix_device.h:static inline __device__ bool rtPotentialIntersection( float tmin )
    optix_device.h:  return optix::rt_potential_intersection( tmin );
    optix_device.h:              "  ray tmin      : %f\n"
    simon:include blyth$ 

    1811 template<class T>
    1812 static inline __device__ void rtTrace( rtObject topNode, optix::Ray ray, T& prd )
    1813 {
    1814   optix::rt_trace(*(unsigned int*)&topNode, ray.origin, ray.direction, ray.ray_type, ray.tmin, ray.tmax, &prd, sizeof(T));
    1815 }

YEP IT WORKS::

     33 static __device__
     34 void intersect_boolean( const uint4& prim, const uint4& identity )
     ..
     61     //float tA_min = propagate_epsilon ;  
     62     //float tB_min = propagate_epsilon ;
     63     float tA_min = ray.tmin ;
     64     float tB_min = ray.tmin ;
     65     float tA     = 0.f ;
     66     float tB     = 0.f ;
        


::

     33 static __device__
     34 void intersect_boolean( const uint4& prim, const uint4& identity )
     35 {          
     ..
     57     // _min 0.f rather than propagate_epsilon 
     58     // leads to missed boundaries when start photons on a boundary, 
     59     // see boolean_csg_on_gpu.rst
     60 
     61     float tA_min = propagate_epsilon ;   
     62     float tB_min = propagate_epsilon ;
     63     float tA     = 0.f ;
     64     float tB     = 0.f ;




scene_epsilon
~~~~~~~~~~~~~~~~

scene_epsilon is how the near clipping feeds into the rays::

     45 RT_PROGRAM void pinhole_camera()
     46 {
     47 
     48   PerRayData_radiance prd;
     49   prd.flag = 0u ;
     50   prd.result = bad_color ;
     51 
     52   float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;
     53 
     54   optix::Ray ray = parallel == 0 ?
     55                        optix::make_Ray( eye                 , normalize(d.x*U + d.y*V + W), radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX)
     56                      :
     57                        optix::make_Ray( eye + d.x*U + d.y*V , normalize(W)                , radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX)
     58                      ;
     59 

::

    simon:geant4_opticks_integration blyth$ opticks-find scene_epsilon
    ./optixrap/cu/pinhole_camera.cu:rtDeclareVariable(float,         scene_epsilon, , );
    ...
    ./optixrap/cu/pinhole_camera.cu:  // scene_epsilon is "t_min" but ray_direction is normalized, 
    ./optixrap/cu/pinhole_camera.cu:  // scene_epsilon is the distance along the ray at which to start 
    ./optixrap/OTracer.cc:    m_context[ "scene_epsilon"]->setFloat(m_composition->getNear());
    ./optixrap/OTracer.cc:    float scene_epsilon = m_composition->getNear();
    ./optixrap/OTracer.cc:    m_context[ "scene_epsilon"]->setFloat(scene_epsilon); 
    ./ana/debug/genstep_sequence_material_mismatch.py:     328     m_context[ "scene_epsilon"]->setFloat(m_composition->getNear());



FIXED Issue : boolean insides invisible from outside
-------------------------------------------------------------

**Not sure why, but fixed by using "absolute loop ctrl" instead of relative in intersect_boolean**

::

    159         else if(
    160                      (action & AdvanceAAndLoop)
    161                   || 
    162                      ((action & AdvanceAAndLoopIfCloser) && tA <= tB )
    163                 )
    164         {
    165 
    166 #ifdef BOOLEAN_DEBUG
    167             if( (action & AdvanceAAndLoop) )                     debugA = 2 ;
    168             if( (action & AdvanceAAndLoopIfCloser) && tA <= tB ) debugA = 3 ;
    169 #endif
    170 
    171             //ctrl = ctrl & ~LIVE_B  ;   // CAUSES INVISIBLE INSIDES 
    172             ctrl = LIVE_A  ;
    173             tA_min = tA ;
    174         }
    175         else if(     
    176                      (action & AdvanceBAndLoop)
    177                   ||  
    178                      ((action & AdvanceBAndLoopIfCloser) && tB <= tA )
    179                 )
    180         {   
    181             //ctrl = ctrl & ~LIVE_A  ;   // CAUSES INVISIBLE INSIDES
    182             ctrl = LIVE_B ;
    183             tB_min = tB ;
    184         }
    185      
    186      }     // while loop 
    187 }



tboolean-box-dented shows a hole where expect to see surface of concave 
hemi-spherical dent.

Using BOOLEAN_DEBUG to color the A and B intersects makes the 
problem clearer.  Can only see innards when the viewpoint is inside.

tboolean-box-minus-sphere shows no insides::

    106     local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    107     local test_config_1=(
    108                  mode=BoxInBox
    109                  analytic=1
    110                  
    111                  shape=box          parameters=0,0,0,1000          boundary=Rock//perfectAbsorbSurface/Vacuum
    112                  
    113                  shape=difference   parameters=0,0,0,300           boundary=Vacuum///$material
    114                  shape=box          parameters=0,0,0,$inscribe     boundary=Vacuum///$material
    115                  shape=sphere       parameters=0,0,0,200           boundary=Vacuum///$material
    116                  
    117                )




FIXED : Issue : cannot see booleans from inside 
------------------------------------------------

* formerly saw that when navigating inside the union, 
  see only container box not the union shape insides

Fixed by moving from::

   if( valid_intersect ) 
   {
       float tint = tmin > 0.f ? tmin : tmax ;  // pick the intersect
       tt = tint > tt_min ? tint : tt_min ;   
       ...

To::

   if( valid_intersect ) 
   {
       //  just because the ray intersects the box doesnt 
       //  mean want to see it, there are 3 possibilities
       //
       //                t_near       t_far   
       //
       //                  |           |
       //        -----1----|----2------|------3---------->
       //                  |           |
       //
       tt =  tt_min < t_near ?  
                              t_near 
                           :
                              ( tt_min < t_far ? t_far : tt_min )


FIXED : Issue : ray trace of box shows slab intersects extending behind the box
--------------------------------------------------------------------------------

**Was due to intersect validity not handling axis aligned photons**

* checked the non-boolean box, thats working fine with no artifacts.

* Using discaxial torch type to shoot photons from 26 positions 
  and directions, so can feel the geometry in a numerical manner.

* when on target, things look correct, the same as the non-boolen box
  when off target the invalid intersects manifest 


::

    local discaxial_hit=0,0,0
    local discaxial_miss=0,0,300
    local torch_config_discaxial=(
                 type=discaxial
                 photons=$photons
                 frame=-1
                 transform=$identity
                 source=$discaxial_hit
                 target=0,0,0
                 time=0.1
                 radius=110
                 distance=200
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$wavelength
               )


Axis aligned photon directions appear to be part of the problem at least::

    421       else if( ts.type == T_DISCAXIAL )
    422       {
    423           unsigned long long photon_id = launch_index.x ;
    424 
    425           //float3 dir = get_direction_26( photon_id % 26 );
    426           //float3 dir = get_direction_6( photon_id % 6 );
    427           //float3 dir = get_direction_6( photon_id % 4, -0.00001f );  // 1st 4: +X,-X,+Y,-Y   SPURIOUS INTERSECTS GONE
    428           //float3 dir = get_direction_6( photon_id % 4, -0.f );       // 1st 4: +X,-X,+Y,-Y   SPURIOUS INTERSECTS GONE
    429           float3 dir = get_direction_6( photon_id % 4, 0.f );          // 1st 4: +X,-X,+Y,-Y   SPURIOUS INTERSECTS BACK AGAIN
    430           
    431           float r = radius*sqrtf(u1) ; // sqrt avoids pole bunchung  
    432           float3 discPosition = make_float3( r*cosPhi, r*sinPhi, 0.f );
    433           rotateUz(discPosition, dir);
    434           
    435           // ts.x0 should be placed inside the target when hits are desired
    436           // wih DISCAXIAL mode
    437           p.position = ts.x0 + distance*dir + discPosition ;
    438           p.direction = -dir ;
    439           


Curious the direction zeros are all negative 0 resulting in -inf for both -X and +X directions::

  ray.origin 200.000000 -11.247929 307.520966 ray.direction -1.000000 -0.000000 -0.000000 idir -1.000000 -inf -inf 
  ray.origin 200.000000 44.386002 262.619629 ray.direction -1.000000 -0.000000 -0.000000 idir -1.000000 -inf -inf 
  ray.origin 200.000000 -88.033470 321.681213 ray.direction -1.000000 -0.000000 -0.000000 idir -1.000000 -inf -inf 
  ray.origin 200.000000 -39.863480 244.735748 ray.direction -1.000000 -0.000000 -0.000000 idir -1.000000 -inf -inf 
  ray.origin -200.000000 97.620598 274.010651 ray.direction 1.000000 -0.000000 -0.000000 idir 1.000000 -inf -inf 
  ray.origin 200.000000 8.609403 199.297638 ray.direction -1.000000 -0.000000 -0.000000 idir -1.000000 -inf -inf 
  ray.origin -200.000000 -67.498100 266.557739 ray.direction 1.000000 -0.000000 -0.000000 idir 1.000000 -inf -inf 
  ray.origin -200.000000 78.251770 366.333496 ray.direction 1.000000 -0.000000 -0.000000 idir 1.000000 -inf -inf 
  ray.origin -200.000000 47.188507 215.060699 ray.direction 1.000000 -0.000000 -0.000000 idir 1.000000 -inf -inf 

Using a delta 0.00001f get -1/delta and spurious interects remain::

  ray.origin 200.000778 9.482430 213.216736 ray.direction -1.000000 -0.000010 -0.000010 idir -1.000000 -100000.000000 -100000.000000 
  ray.origin -199.999054 48.094410 346.568787 ray.direction 1.000000 -0.000010 -0.000010 idir 1.000000 -100000.000000 -100000.000000 

Bizarrely switching to delta -0.00001f get 1/delta and the spurious intersects are gone::

  ray.origin 199.999344 -88.035469 321.679199 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 
  ray.origin 199.999222 9.478431 213.212708 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 
  ray.origin 200.000000 49.761848 249.952194 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 
  ray.origin 200.000748 39.745564 334.747955 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 
  ray.origin -199.999298 -8.694067 238.793365 ray.direction 1.000000 0.000010 0.000010 idir 1.000000 100000.000000 100000.000000 
  ray.origin 199.999878 -76.475029 363.946503 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 
  ray.origin 200.000290 44.076099 285.449768 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 

Same when using -0.f::

    425           //float3 dir = get_direction_26( photon_id % 26 );
    426           //float3 dir = get_direction_6( photon_id % 6 );
    427           //float3 dir = get_direction_6( photon_id % 4, -0.00001f );     // 1st 4: +X,-X,+Y,-Y 
    428           float3 dir = get_direction_6( photon_id % 4, -0.f );     // 1st 4: +X,-X,+Y,-Y 
    429           
    430           float r = radius*sqrtf(u1) ; // sqrt avoids pole bunchung  
    431           float3 discPosition = make_float3( r*cosPhi, r*sinPhi, 0.f );
    432           rotateUz(discPosition, dir);
    433           
    434           // ts.x0 should be placed inside the target when hits are desired
    435           // wih DISCAXIAL mode
    436           p.position = ts.x0 + distance*dir + discPosition ;
    437           p.direction = -dir ;

::

  ray.origin 200.000000 14.684715 244.904205 ray.direction -1.000000 0.000000 0.000000 idir -1.000000 inf inf 
  ray.origin 200.000000 -68.328766 251.635269 ray.direction -1.000000 0.000000 0.000000 idir -1.000000 inf inf 
  ray.origin -200.000000 102.468193 335.907471 ray.direction 1.000000 0.000000 0.000000 idir 1.000000 inf inf 
  ray.origin 200.000000 -26.478765 307.570923 ray.direction -1.000000 0.000000 0.000000 idir -1.000000 inf inf 
  ray.origin 200.000000 -15.085106 304.063721 ray.direction -1.000000 0.000000 0.000000 idir -1.000000 inf inf 


::

     42    float3 idir = make_float3(1.f)/ray.direction ;
     43    float3 t0 = (bmin - ray.origin)*idir;
     44    float3 t1 = (bmax - ray.origin)*idir;


::

     idir -1.000000 -inf -inf t0 300.000000 inf inf t1 100.000000 -inf inf 
     idir -1.000000 -inf -inf t0 300.000000 inf inf t1 100.000000 -inf inf 
     idir -1.000000 -inf -inf t0 300.000000 inf inf t1 100.000000 -inf inf 
     idir -1.000000 -inf -inf t0 300.000000 inf inf t1 100.000000 -inf inf 
     idir 1.000000  -inf -inf t0 100.000000 inf inf t1 300.000000 -inf inf 
     idir 1.000000  -inf -inf t0 100.000000 inf inf t1 300.000000 -inf inf 
     idir 1.000000  -inf -inf t0 100.000000 inf inf t1 300.000000 -inf inf 
     idir 1.000000  -inf -inf t0 100.000000 inf inf t1 300.000000 -inf inf 





CUDA fminf/fmaxf/max infinity/nan handling ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

::

    simon:include blyth$ grep fminf *.*
    device_functions.h:__DEVICE_FUNCTIONS_STATIC_DECL__ float fminf(float x, float y);
    device_functions.hpp:__DEVICE_FUNCTIONS_STATIC_DECL__ float fminf(float x, float y)
    device_functions.hpp:  return __nv_fminf(x, y);
    device_functions_decls.h:__DEVICE_FUNCTIONS_DECLS__ float __nv_fminf(float x, float y);
    math_functions.h:extern __host__ __device__ __device_builtin__ float                  fminf(float x, float y) __THROW;
    math_functions.h:extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl fminf(float x, float y);
    math_functions.h:__func__(float fminf(float a, float b));
    math_functions.hpp:  return fminf(a, b);
    math_functions.hpp:  return fminf(a, b);
    math_functions.hpp:__func__(float fminf(float a, float b))
    nppi_color_conversion.h: *  This code uses the fmaxf() and fminf() 32 bit floating point math functions.
    nppi_color_conversion.h: *  Npp32f nMin = fminf(nNormalizedR, nNormalizedG);
    nppi_color_conversion.h: *         nMin = fminf(nMin, nNormalizedB);
    nppi_color_conversion.h: *  This code uses the fmaxf() and fminf() 32 bit floating point math functions.
    nppi_color_conversion.h: *  Npp32f nTemp = fminf(nNormalizedR, nNormalizedG);
    nppi_color_conversion.h: *         nTemp = fminf(nTemp, nNormalizedB);
    simon:include blyth$ 
    simon:include blyth$ 
    simon:include blyth$ pwd
    /Developer/NVIDIA/CUDA-7.0/include





FIXED Issue : boolean intersection "lens" : boundary disappears from inside
------------------------------------------------------------------------------

**FIXED by starting tmin from propagate_epsilon, as during propagation photons start on boundaries**


Using boolean sphere-sphere intersection to construct a lens.::

     72 tboolean-testconfig()
     73 {
     74     local material=GlassSchottF2
     75     #local material=MainH2OHale
     76 
     77     local test_config=(
     78                  mode=BoxInBox
     79                  analytic=1
     80 
     81                  shape=box      parameters=0,0,0,1200               boundary=Rock//perfectAbsorbSurface/Vacuum
     82 
     83                  shape=intersection parameters=0,0,0,400            boundary=Vacuum///$material
     84                  shape=sphere       parameters=0,0,-600,641.2          boundary=Vacuum///$material
     85                  shape=sphere       parameters=0,0,600,641.2           boundary=Vacuum///$material
     86 
     87                )
     91      echo "$(join _ ${test_config[@]})" 
     92 }

Observe that photons reflecting inside the lens off the 2nd boundary do 
not intersect with the 1st boundary on their way back yielding "TO BT BR SA"

Similarly, and more directly, also have "TO BT SA" not seeing the 2nd boundary. 

Initially thought the raytrace confirmed this as 
it looked OK from outside but when go inside the boundary disappears, but
that turns out to be just near clipping.

::

    tboolean-;tboolean--




FIXED Issue : lens not bending light 
--------------------------------------

Fixed by passing the boundary index 
via the instanceIdentity attribute from intersection 
to closest hit progs.






approach
-----------


ggeo/GPmt.hh
ggeo/GCSG.hh
    Brings python prepared CSG tree for DYB PMT into GPmt member

    Looks like GCSG is currently being translated into into 
    partBuffer/solidBuffer representation prior to GPU ? 




hemi-pmt.cu::

    /// flag needed in solidBuffer
    ///
    ///   0:primitive
    ///   1:boolean-intersect
    ///   2:boolean-union
    ///   3:boolean-difference
    ///
    /// presumably the numParts will be 2 for booleans
    /// thence can do the sub-intersects and boolean logic
    /// 
    /// ...
    /// need to elide the sub-solids from OptiX just passing booleans
    /// in as a single solidBuffer entry with numParts = 2 ?
    ///
    /// maybe change name solidBuffer->primBuffer
    /// as booleans handled as OptiX primitives composed of two parts
    ///   

    1243 RT_PROGRAM void intersect(int primIdx)
    1244 {
    1245   const uint4& solid    = solidBuffer[primIdx];
    1246   unsigned int numParts = solid.y ;
    ....
    1252   uint4 identity = identityBuffer[instance_index] ;
    1254 
    1255   for(unsigned int p=0 ; p < numParts ; p++)
    1256   {
    1257       unsigned int partIdx = solid.x + p ;
    1258 
    1259       quad q0, q1, q2, q3 ;
    1260 
    1261       q0.f = partBuffer[4*partIdx+0];
    1262       q1.f = partBuffer[4*partIdx+1];
    1263       q2.f = partBuffer[4*partIdx+2] ;
    1264       q3.f = partBuffer[4*partIdx+3];
    1265 
    1266       identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)
    1267 
    1268       int partType = q2.i.w ;
    1269 
    1270       // TODO: use enum      
    ////     this is the NPart.hpp enum 
    ////
    1271       switch(partType)
    1272       {
    1273           case 0:
    1274                 intersect_aabb(q2, q3, identity);
    1275                 break ;
    1276           case 1:
    1277                 intersect_zsphere<false>(q0,q1,q2,q3,identity);
    1278                 break ;



