RPN_convert_shakedown
======================

* importPrim does not invoke importNode for listnode - instead they are routed to importListnode
* importPrim_RPN has no such special handling of listnode


How to simplify ? To avoid omission of tesselation config ?
-------------------------------------------------------------

* write geometry specific environment file into the geometry dir ?
* yes - but do not always run from persisted geometry dir



Looks like "Waterdistributor* WaterDistributor* uni1*" ARE NOT BEING AUTO-TESSELATED - THIS COULD EXPLAIN POOR TRACE PERFORMANCE - AND MESS WITH SER improvement
------------------------------------------------------------------------------------------------------------------------------------------------------------------

::

    ~/o/CSG/tests/CSGFoundry_CreateFromSimTest.sh


    ]rt.render
    ]CSGImport::importPrim_RPN.dump_LVID:1 node.lvid 119 LVID 119 name s_EMFcoil_holder_ring17_seg1 soname s_EMFcoil_holder_ring17_seg1
    2026-11-17 15:22:53.847 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type discontiguous nd.lvid 270 node.lvid 270 soname uni1_1 name uni1_1 write_rpn YES
    2026-11-17 15:22:53.847 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type discontiguous nd.lvid 269 node.lvid 269 soname uni1_0 name uni1_0 write_rpn YES
    2026-11-17 15:22:53.862 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type contiguous nd.lvid 35 node.lvid 35 soname Waterdistributor_1 name Waterdistributor_1 write_rpn YES
    2026-11-17 15:22:53.862 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type contiguous nd.lvid 34 node.lvid 34 soname Waterdistributor_0 name Waterdistributor_0 write_rpn YES
    2026-11-17 15:22:53.862 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type contiguous nd.lvid 35 node.lvid 35 soname Waterdistributor_1 name Waterdistributor_1 write_rpn YES
    2026-11-17 15:22:53.862 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type contiguous nd.lvid 34 node.lvid 34 soname Waterdistributor_0 name Waterdistributor_0 write_rpn YES
    2026-11-17 15:22:53.862 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type contiguous nd.lvid 42 node.lvid 42 soname WaterDistributorPartIIIUnion_Seg1 name WaterDistributorPartIIIUnion_Seg1 write_rpn YES
    2026-11-17 15:22:53.876 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type contiguous nd.lvid 247 node.lvid 247 soname Waterdistributor_3 name Waterdistributor_3 write_rpn YES
    2026-11-17 15:22:53.877 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type contiguous nd.lvid 246 node.lvid 246 soname Waterdistributor_2 name Waterdistributor_2 write_rpn YES
    2026-11-17 15:22:53.878 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type contiguous nd.lvid 247 node.lvid 247 soname Waterdistributor_3 name Waterdistributor_3 write_rpn YES
    2026-11-17 15:22:53.878 FATAL [1258300] [CSGImport::importNode@642]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type contiguous nd.lvid 246 node.lvid 246 soname Waterdistributor_2 name Waterdistributor_2 write_rpn YES
    2026-11-17 15:22:53.977 INFO  [1258300] [CSGFoundry::CreateFromSim@3069] ] creating alt CSGFoundry
    2026-11-17 15:22:53.977 INFO  [1258300] [CSGFoundry::CreateFromSim@3074] ]
    2026-11-17 15:22:53.977 INFO  [1258300] [CSGFoundry::save_@2614] /tmp/blyth/opticks/CSGFoundry_CreateFromSimTest/CSGFoundry
    2026-11-17 15:22:54.095 INFO  [1258300] [CSGFoundry::save_@2638]  SSim::save /tmp/blyth/opticks/CSGFoundry_CreateFromSimTest/CSGFoundry



Full geometry RPN worked - modulo the fail to auto tesselate issue
---------------------------------------------------------------------

::

    15439  RPN nodes    ## almost half the nodes using RPN compared to complete binary tree
    29173  CBT nodes


::

    A[blyth@localhost junosw]$ cd /tmp/blyth/opticks/CSGFoundry_CreateFromSimTest/CSGFoundry
    A[blyth@localhost CSGFoundry]$ l
    total 6448
    3076 -rw-r--r--. 1 blyth blyth 3148032 Nov 17 15:22 inst.npy
     600 -rw-r--r--. 1 blyth blyth  613696 Nov 17 15:22 itra.npy
     600 -rw-r--r--. 1 blyth blyth  613696 Nov 17 15:22 tran.npy
    1824 -rw-r--r--. 1 blyth blyth 1867200 Nov 17 15:22 node.npy
     232 -rw-r--r--. 1 blyth blyth  233728 Nov 17 15:22 prim.npy
       4 -rw-r--r--. 1 blyth blyth    1217 Nov 17 15:22 meta.txt
       4 -rw-r--r--. 1 blyth blyth     269 Nov 17 15:22 mmlabel.txt
      92 -rw-r--r--. 1 blyth blyth   94091 Nov 17 15:22 primname.txt
       4 -rw-r--r--. 1 blyth blyth     752 Nov 17 15:22 solid.npy
      12 -rw-r--r--. 1 blyth blyth   11152 Nov 17 15:22 meshname.txt
       0 drwxr-xr-x. 4 blyth blyth      46 Nov 17 15:16 ..
       0 drwxr-xr-x. 5 blyth blyth      93 Nov 13 16:05 SSim
       0 drwxr-xr-x. 3 blyth blyth     190 Nov 13 16:05 .
    A[blyth@localhost CSGFoundry]$ date
    Tue Nov 17 03:45:24 PM CST 2026
    A[blyth@localhost CSGFoundry]$ cd ..
    A[blyth@localhost CSGFoundry_CreateFromSimTest]$ l
    total 8
    4 drwxr-xr-x.  4 blyth blyth 4096 Nov 17 15:16 CSGFoundry_RPN
    0 drwxr-xr-x.  4 blyth blyth   46 Nov 17 15:16 .
    4 drwxr-xr-x. 26 blyth blyth 4096 Nov 16 19:56 ..
    0 drwxr-xr-x.  3 blyth blyth  190 Nov 13 16:05 CSGFoundry
    A[blyth@localhost CSGFoundry_CreateFromSimTest]$ cd CSGFoundry_RPN/
    A[blyth@localhost CSGFoundry_RPN]$ l
    total 5596
    3076 -rw-r--r--. 1 blyth blyth 3148032 Nov 17 15:22 inst.npy
     600 -rw-r--r--. 1 blyth blyth  613696 Nov 17 15:22 itra.npy
     600 -rw-r--r--. 1 blyth blyth  613696 Nov 17 15:22 tran.npy
     968 -rw-r--r--. 1 blyth blyth  988224 Nov 17 15:22 node.npy
     232 -rw-r--r--. 1 blyth blyth  233728 Nov 17 15:22 prim.npy
       4 -rw-r--r--. 1 blyth blyth    1217 Nov 17 15:22 meta.txt
       4 -rw-r--r--. 1 blyth blyth     269 Nov 17 15:22 mmlabel.txt
      92 -rw-r--r--. 1 blyth blyth   94091 Nov 17 15:22 primname.txt
       4 -rw-r--r--. 1 blyth blyth     752 Nov 17 15:22 solid.npy
      12 -rw-r--r--. 1 blyth blyth   11152 Nov 17 15:22 meshname.txt
       0 drwxr-xr-x. 5 blyth blyth      93 Nov 17 15:16 SSim
       4 drwxr-xr-x. 4 blyth blyth    4096 Nov 17 15:16 .
       0 drwxr-xr-x. 2 blyth blyth     102 Nov 17 15:16 rpn
       0 drwxr-xr-x. 4 blyth blyth      46 Nov 17 15:16 ..
    A[blyth@localhost CSGFoundry_RPN]$ cd rpn
    A[blyth@localhost rpn]$ l
    total 972
    600 -rw-r--r--. 1 blyth blyth 613696 Nov 17 15:22 itra.npy
      4 -rw-r--r--. 1 blyth blyth     27 Nov 17 15:22 NPFold_index.txt
      0 -rw-r--r--. 1 blyth blyth      0 Nov 17 15:22 NPFold_names.txt
    300 -rw-r--r--. 1 blyth blyth 306912 Nov 17 15:22 parm.npy
     64 -rw-r--r--. 1 blyth blyth  61884 Nov 17 15:22 node.npy
      0 drwxr-xr-x. 2 blyth blyth    102 Nov 17 15:16 .
      4 drwxr-xr-x. 4 blyth blyth   4096 Nov 17 15:16 ..
    A[blyth@localhost rpn]$ xxd -l128 node.npy
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 7534 272c 2027 666f 7274  r': '<u4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2831 3534  e, 'shape': (154
    00000040: 3339 2c29 2c20 7d20 2020 2020 2020 2020  39,), }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    A[blyth@localhost rpn]$ xxd -l128 ../node.npy
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6634 272c 2027 666f 7274  r': '<f4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2831 3534  e, 'shape': (154
    00000040: 3339 2c20 342c 2034 292c 207d 2020 2020  39, 4, 4), }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    A[blyth@localhost rpn]$


::

    A[blyth@localhost rpn]$ f
    Python 3.13.2 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:02) [GCC 11.2.0]
    Type 'copyright', 'credits' or 'license' for more information
    IPython 9.1.0 -- An enhanced Interactive Python. Type '?' for help.
    Tip: Use `ipython --help-all | less` to view all the IPython configuration options.
    f

    CMDLINE:/home/blyth/np/fold.py
    f.base:.

      : f.node                                             :             (15439,) : 0:38:16.093992
      : f.itra                                             :         (9587, 4, 4) : 0:38:16.092992
      : f.parm                                             :         (9587, 2, 4) : 0:38:16.092992
      : f.NPFold_index                                     :                 (3,) : 0:38:16.092992
      : f.NPFold_names                                     :                 (0,) : 0:38:16.092992

     min_stamp : 2026-11-17 15:22:57.600451
     max_stamp : 2026-11-17 15:22:57.601451
     dif_stamp : 0:00:00.001000
     age_stamp : 0:38:16.092992

    In [1]:


::

    In [9]: np.c_[np.unique( f.node & 0x7f, return_counts=True )]
    Out[9]:
    array([[   1, 5409],
           [   2,  432],
           [  11,    9],
           [  12,    2],
           [ 101,   13],
           [ 103,   25],
           [ 105, 1627],
           [ 108,  112],
           [ 110, 7554],
           [ 116,  158],
           [ 121,   68],
           [ 124,    6],
           [ 126,   24]])


Is boundary needed ?
---------------------

NO: for trace, YES: for propagate

* however not enough bits in srpn.h for this so try carrying it in the Binding.h

::

     21 struct CustomPrim
     22 {
     23     int numNode ;
     24     int nodeOffset ;
     25     int globalPrimIdx ;
     26 };
     27
     28 struct TriMesh
     29 {
     30     unsigned boundary ;
     31     uint3*  indice ;  // indices of triangle vertices and normal
     32     float3* vertex ;
     33     float3* normal ;
     34     int globalPrimIdx ;
     35 };
     36
     37
     38 struct HitGroupData
     39 {
     40     union
     41     {
     42         CustomPrim prim ;
     43         TriMesh    mesh ;
     44     };
     45 };





::

     890 extern "C" __global__ void __closesthit__ch()
     891 {
     892     unsigned iindex = optixGetInstanceIndex() ;
     893     unsigned identity = optixGetInstanceId() ;
     894     unsigned iindex_identity = (( iindex & 0xffffu ) << 16 ) | ( identity & 0xffffu ) ;
     895
     896     OptixPrimitiveType type = optixGetPrimitiveType(); // HUH: getting type 0, when expect OPTIX_PRIMITIVE_TYPE_TRIANGLE
     897     const HitGroupData* hg = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
     898
     899 #if defined(DEBUG_PIDX)
     900     //const uint3 idx = optixGetLaunchIndex();
     901     //if(idx.x == 10 && idx.y == 10) printf("//__closesthit__ch idx(%u,%u,%u) type %d \n", idx.x, idx.y, idx.z, type);
     902     //if(identity == 52264 || identity == 52265 || identity == 52266) printf("//__closesthit__ch iindex %u type %d identity %d \n", iindex, type, identity );
     903 #endif
     904
     905     if(type == OPTIX_PRIMITIVE_TYPE_TRIANGLE || type == 0)  // WHY GETTING ZERO HERE ?
     906     {
     907         const TriMesh& mesh = hg->mesh ;
     908         unsigned globalPrimIdx_boundary = (( mesh.globalPrimIdx & 0xffffu ) << 16 ) | ( mesh.boundary & 0xffffu ) ;
     909         const unsigned prim_idx = optixGetPrimitiveIndex();
     910         const float2   barys    = optixGetTriangleBarycentrics();













::

    In [4]: f.node[:,1,2].view(np.uint32)
    Out[4]: array([  0,   1,   2,   1,   1, ..., 439, 440, 441, 442, 443], shape=(15439,), dtype=uint32)

    In [5]: f.node[:,1,2].view(np.uint32).min()
    Out[5]: np.uint32(0)

    In [6]: f.node[:,1,2].view(np.uint32).max()
    Out[6]: np.uint32(449)


Hmm - lots of single node boundary, burning thru bits::

    n [9]: np.c_[np.unique(f.node[:,1,2].view(np.uint32), return_counts=True)]
    Out[9]:
    array([[   0,    1],
           [   1,    9],
           [   2,    1],
           [   3,    3],
           [   4,    3],
           [   5,    7],
           [   6,    3],
           [   7,  195],
           [   8,    1],
           [   9,    1],
           [  10,    3],
           [  11,    1],
           [  12,    1],
           [  13,   64],
           [  14,   64],
           [  34,    1],
           [  35,    1],
           [  36,    3],
           [  37,    3],
           [  38,    3],
           [  39,    3],
           [  40,    3],
           [  41,   24],
           [  42,   24],
           [  43,    9],
           [  44,    3],
           [  45,    4],
           [  46,    5],
           [  47,    5],
           [  48,    5],
           [  49,    5],
           [  50,    5],
           [  51,    5],
           [  52,    5],
           [  53,    5],
           [  54,    5],
           [  55,    5],
           [  56,    5],
           [  57,    5],
           [  58,    5],
           [  59,    5],
           [  60,    5],
           [  61,    5],
           [  62,    5],
           [  63,    5],
           [  64,    5],
           [  65,    5],
           [  66,    5],
           [  67,    5],
           [  68,    5],
           [  69,   25],
           [  70,   45],
           [  71,   30],
           [  72,   30],
           [  73,   30],
           [  74,   30],
           [  75,   30],
           [  76,   90],
           [  77,   30],
           [  78,   30],
           [  79,   30],
           [  80,   30],
           [  81,   15],
           [  82,   25],
           [  83,    5],
           [  84,    5],
           [  85,    5],
           [  86,    5],
           [  87,    5],
           [  88,    5],
           [  89,    5],
           [  90,    7],
           [  91,    7],
           [  92,    7],
           [  93,    7],
           [  94,    7],
           [  95,    7],
           [  96,    7],
           [  97,    7],
           [  98,    7],
           [  99,    7],
           [ 100,    7],
           [ 101,    7],
           [ 102,    7],
           [ 103,    7],
           [ 104,    7],
           [ 105,    7],
           [ 106,    7],
           [ 107,    7],
           [ 108,    7],
           [ 109,    7],
           [ 110,    7],
           [ 111,    7],
           [ 112,    7],
           [ 113,    7],
           [ 114,    7],
           [ 115,    7],
           [ 116,    7],
           [ 117,    5],
           [ 118,    7],
           [ 119,    7],
           [ 120,    7],
           [ 121,    7],
           [ 122,    5],
           [ 123,    7],
           [ 124,    7],
           [ 125,    7],
           [ 126,    7],
           [ 127,    7],
           [ 128,    7],
           [ 129,    7],
           [ 130,    7],
           [ 131,    7],
           [ 132,    7],
           [ 133,    7],
           [ 134,    7],
           [ 135,    7],
           [ 136,    7],
           [ 137,    7],
           [ 138,    7],
           [ 139,    7],
           [ 140,    7],
           [ 141,    7],
           [ 142,    7],
           [ 143,    7],
           [ 144,    7],
           [ 145,    7],
           [ 146,    7],
           [ 147,    7],
           [ 148,    7],
           [ 149,    7],
           [ 150,    5],
           [ 151,    5],
           [ 152,    7],
           [ 153,    7],
           [ 154,    7],
           [ 155,    7],
           [ 156,    7],
           [ 157,    7],
           [ 158,    7],
           [ 159,    7],
           [ 160,    7],
           [ 161,    7],
           [ 162,  150],
           [ 163,  150],
           [ 164,  150],
           [ 165,  150],
           [ 166,  150],
           [ 167,  150],
           [ 168,  150],
           [ 169,  150],
           [ 170,  150],
           [ 171,  150],
           [ 172,  150],
           [ 173,  150],
           [ 174,  150],
           [ 175,  150],
           [ 176,  150],
           [ 177,  150],
           [ 178,  150],
           [ 179,  150],
           [ 180,  150],
           [ 181,  150],
           [ 182,  150],
           [ 183,  150],
           [ 184,  150],
           [ 185,  150],
           [ 186,  150],
           [ 187,  150],
           [ 188,   25],
           [ 189,   25],
           [ 190,   13],
           [ 191,    1],
           [ 192,    3],
           [ 193,    3],
           [ 194,    3],
           [ 195,    1],
           [ 196,    1],
           [ 197,    3],
           [ 198,    3],
           [ 199,    3],
           [ 200,    3],
           [ 201,    1],
           [ 202,   50],
           [ 203,  150],
           [ 204,  150],
           [ 205,  150],
           [ 206,  150],
           [ 207,  150],
           [ 208,  150],
           [ 209,  150],
           [ 210,  150],
           [ 211,  150],
           [ 212,  150],
           [ 213,  150],
           [ 214,  150],
           [ 215,  150],
           [ 216,  150],
           [ 217,  150],
           [ 218,  150],
           [ 219,  150],
           [ 220,  150],
           [ 221,  150],
           [ 222,  150],
           [ 223,   50],
           [ 224,  150],
           [ 225,  150],
           [ 226,  150],
           [ 227,  150],
           [ 228,  150],
           [ 229,  150],
           [ 230,  150],
           [ 231,  150],
           [ 232,  150],
           [ 233,  150],
           [ 234,  150],
           [ 235,  150],
           [ 236,  150],
           [ 237,  150],
           [ 238,  150],
           [ 239,  150],
           [ 240,  150],
           [ 241,  150],
           [ 242,  150],
           [ 243,  150],
           [ 244,  150],
           [ 245,  150],
           [ 246,  150],
           [ 247,    3],
           [ 248,    3],
           [ 249,   24],
           [ 250,   24],
           [ 251,    3],
           [ 252,    3],
           [ 253,    5],
           [ 254,  189],
           [ 255,    5],
           [ 256,  138],
           [ 257,  120],
           [ 258,   16],
           [ 259,    5],
           [ 260,   11],
           [ 261,    5],
           [ 262,    7],
           [ 263, 1540],
           [ 264,    1],
           [ 265,    7],
           [ 266,   13],
           [ 267,  952],
           [ 268,   13],
           [ 269,   14],
           [ 270,   11],
           [ 271,    7],
           [ 272,    3],
           [ 273,    3],
           [ 274,    3],
           [ 275,    3],
           [ 276,    3],
           [ 277,    1],
           [ 278,    3],
           [ 279,   11],
           [ 280,    1],
           [ 281,    3],
           [ 282,    3],
           [ 283,    3],
           [ 284,    1],
           [ 285,    1],
           [ 286,    1],
           [ 287,    1],
           [ 288,    1],
           [ 289,    1],
           [ 290,    1],
           [ 291,    1],
           [ 292,    1],
           [ 293,    1],
           [ 294,    1],
           [ 295,    1],
           [ 296,    1],
           [ 297,    1],
           [ 298,    1],
           [ 299,    1],
           [ 300,    1],
           [ 301,    1],
           [ 302,    1],
           [ 303,    1],
           [ 304,    1],
           [ 305,    1],
           [ 306,    1],
           [ 307,    1],
           [ 308,    1],
           [ 309,    1],
           [ 310,    1],
           [ 311,    1],
           [ 312,    1],
           [ 313,    1],
           [ 314,    1],
           [ 315,    1],
           [ 316,    1],
           [ 317,    1],
           [ 318,    1],
           [ 319,    1],
           [ 320,    1],
           [ 321,    1],
           [ 322,    1],
           [ 323,    1],
           [ 324,    1],
           [ 325,    1],
           [ 326,    1],
           [ 327,    1],
           [ 328,    1],
           [ 329,    1],
           [ 330,    1],
           [ 331,    1],
           [ 332,    1],
           [ 333,    1],
           [ 334,    1],
           [ 335,    1],
           [ 336,    1],
           [ 337,    1],
           [ 338,    1],
           [ 339,    1],
           [ 340,    1],
           [ 341,    1],
           [ 342,    1],
           [ 343,    1],
           [ 344,    1],
           [ 345,    1],
           [ 346,    1],
           [ 347,    1],
           [ 348,    1],
           [ 349,    1],
           [ 350,    1],
           [ 351,    1],
           [ 352,    1],
           [ 353,    1],
           [ 354,    1],
           [ 355,    1],
           [ 356,    1],
           [ 357,    1],
           [ 358,    1],
           [ 359,    1],
           [ 360,    1],
           [ 361,    1],
           [ 362,    1],
           [ 363,    1],
           [ 364,    1],
           [ 365,    1],
           [ 366,    1],
           [ 367,    1],
           [ 368,    1],
           [ 369,    1],
           [ 370,    1],
           [ 371,    1],
           [ 372,    1],
           [ 373,    1],
           [ 374,    1],
           [ 375,    1],
           [ 376,    1],
           [ 377,    1],
           [ 378,    1],
           [ 379,    1],
           [ 380,    1],
           [ 381,    1],
           [ 382,    1],
           [ 383,    1],
           [ 384,    1],
           [ 385,    1],
           [ 386,    1],
           [ 387,    1],
           [ 388,    1],
           [ 389,    1],
           [ 390,    1],
           [ 391,    1],
           [ 392,    1],
           [ 393,    1],
           [ 394,    1],
           [ 395,    1],
           [ 396,    1],
           [ 397,    1],
           [ 398,    1],
           [ 399,    1],
           [ 400,    1],
           [ 401,    1],
           [ 402,    1],
           [ 403,    1],
           [ 404,    1],
           [ 405,    1],
           [ 406,    1],
           [ 407,    1],
           [ 408,    1],
           [ 409,    1],
           [ 410,    1],
           [ 411,    1],
           [ 412,    1],
           [ 413,    1],
           [ 414,    1],
           [ 415,    1],
           [ 416,    1],
           [ 417,    1],
           [ 418,    1],
           [ 419,    1],
           [ 420,    1],
           [ 421,    1],
           [ 422,    1],
           [ 423,    1],
           [ 424,    1],
           [ 425,    1],
           [ 426,    1],
           [ 427,    1],
           [ 428,    1],
           [ 429,    1],
           [ 430,    1],
           [ 431,    1],
           [ 432,    1],
           [ 433,    1],
           [ 434,    1],
           [ 435,    1],
           [ 436,    1],
           [ 437,    1],
           [ 438,    1],
           [ 439,    1],
           [ 440,    1],
           [ 441,    1],
           [ 442,    1],
           [ 443,    1],
           [ 444,    1],
           [ 445,    3],
           [ 446,    5],
           [ 447,    5],
           [ 448,    7],
           [ 449,    7]])

    In [10]:







Are they are all standardly tesselated ?
--------------------------------------

::

     20 oj_geomsetup()
     21 {
     22    : opticks_juno.sh
     23    : extra env settings needed by Opticks-JUNOSW geometry conversion
     24    : NB this is sourced from envset.sh so needs to be appropriate to every usage
     25    : not just the .gitlab-ci.yml one off first runs
     26
     27    if [ -n "$OJ_VERBOSE" ]; then type $FUNCNAME ; fi
     28
     29    export GEOM=$OJ_SLUG
     30    export OJ_PREFIX=$(dirname $(dirname $(realpath $BASH_SOURCE))) # one above bin dir
     31    export OJ_GEOMDIR=$OJ_PREFIX/.opticks/GEOM/${GEOM}
     32
     33
     34    export ${GEOM}_CFBaseFromGEOM=${OJ_GEOMDIR}  ## for viz, eg CSGOptiXRenderInteractiveTest
     35    export CSGOptiXRenderInteractiveTest__FRAME_HOP=1
     36
     37    export stree__FREQ_CUT=300   # less than 348 in order to instance WP_ATM_LPMT : requires Opticks-v0.5.3 or later to take effect
     38    export stree__force_triangulate_solid="filepath:${OJ_GEOMDIR}/CSGFoundry/tessname.txt"
     39    export SScene__initFromTree_addFrames=${OJ_GEOMDIR}/CSGFoundry/framespec.txt
     40
     41    local vv="FUNCNAME OJ_SLUG GEOM BASH_SOURCE OJ_PREFIX OJ_GEOMDIR ${GEOM}_CFBaseFromGEOM stree__force_triangulate_solid SScene__initFromTree_addFrames"
     42    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
     43
     44


::

    A[blyth@localhost CSGFoundry]$ pwd
    /cvmfs/opticks.ihep.ac.cn/oj/releases/el9_amd64_gcc15/J26.4.1_Opticks-v0.7.3/Latest/.opticks/GEOM/J26_4_1_Opticks_v0_7_3/CSGFoundry

    A[blyth@localhost CSGFoundry]$ grep -v \# tessname.txt
    solidSJCLSanchor
    solidSJFixture
    solidSJReceiver
    solidSJReceiverFastern
    solidXJanchor
    solidXJfixture
    A[blyth@localhost CSGFoundry]$



When does tessname.txt get written ?
----------------------------------------

::

     51 ##############################################################################
     52 # Opticks Build Job Template
     53 ##############################################################################
     54 .opticks_build_job_template:
     55   extends: .job_template
     56   stage: build
     57   image: simoncblyth/junosw:el9.4-cuda13.1-opticks
     58   variables:
     59     JUNO_CLANG_PREFIX:
     60     JUNO_OPTICKS_PREFIX: /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-vLatest
     61     EXTRA_BUILD_FLAGS: JUNO_CMAKE_BUILD_TYPE=Debug
     62
     63   before_script:
     64     - git config --global --add safe.directory $CI_PROJECT_DIR
     65
     66   script:
     67     - printf "[ ================ BUILD PHASE OF BUILD JOB ========================\n"
     68     - for repo in juno.ihep.ac.cn opticks.ihep.ac.cn; do if [ ! -d /cvmfs/$repo ]; then sudo mkdir /cvmfs/$repo; fi; sudo mount -t cvmfs $repo /cvmfs/$repo; done
     69     - export OPTICKS_SETUP_VERBOSE=1   # ~/j/gitlab-ci/mockbuild.sh shows this protects against "set -eo pipefail"
     70     - export JUNO_OFFLINE_OFF=1        # Disable sourcing JUNOTOP/junosw/InstallArea/setup.sh prior to generation
     71     - source $JUNOTOP/setup.sh
     72     - if [ -n "$JUNO_CLANG_PREFIX" ]; then source $JUNO_CLANG_PREFIX/bashrc; fi
     73     - source $JUNO_OPTICKS_PREFIX/bashrc   # config Opticks including OPTICKS_CONFIG
     74     - export JUNO_OPTICKS_CONFIG=$OPTICKS_CONFIG
     75     - printf "================= AFTER SETUP PROCEED WITH BUILD === JUNO_OPTICKS_CONFIG $JUNO_OPTICKS_CONFIG =================\n"
     76     - env $EXTRA_BUILD_FLAGS ./build.sh
     77     - printf "] ================ BUILD PHASE OF BUILD JOB =========================\n"
     78     - printf "[ ================ INIT PHASE OF BUILD JOB =========================\n"
     79     - source InstallArea/setup.sh        ## runtime requires to source the just created InstallArea/setup.sh
     80     - ./oj_initialize.sh InstallArea     ## writes and sources envset.sh, init geomdir, makes tarball, scp to stratum-zero
     81     - printf "] ================ INIT PHASE OF BUILD JOB =========================\n"
     82
     83   rules:
     84     - if: $CI_PIPELINE_SOURCE == "merge_request_event"
     85       changes:
     86       - .gitlab-ci.yml
     87       - .gitlab-ci/oj_helper.sh
     88       - Simulation/**/*
     89     - if: $CI_PIPELINE_SOURCE == "schedule"
     90       when: always
     91


oj_initialize.sh
-------------------

::

    075
     76
     77 idir=${1:-InstallArea}
     78
     79 source .gitlab-ci/oj_helper.sh BLESS_INSTALL $idir
     80
     81 envset=$idir/envset.sh
     82 source $envset
     83 [ $? -ne 0 ] && echo "$BASH_SOURCE - ERROR from envset[$envset]"
     84
     85 oj_initialize_dump()
     86 {
     87     local vv="BASH_SOURCE PWD idir envset OPTICKS_PREFIX TUTORIALROOT OJ_DEBUG OJ_NOINIT"
     88     for v in $vv ; do printf "%30s : %s\n " "$v" "${!v}" ; done
     89 }
     90
     91 oj_initialize_dump
     92
     93 if [ -n "$OJ_DEBUG" ]; then
     94
     95     echo $BASH_SOURCE - OJ_DEBUG is defined
     96     oj_initialize_debug
     97     [ $? -ne 0 ] && echo $BASH_SOURCE - ERROR FROM oj_initialize_debug
     98
     99 elif [ -n "$OJ_NOINIT" ]; then
    100
    101     echo $BASH_SOURCE - SKIP geometry initialization as OJ_NOINIT is defined
    102
    103 else
    104     oj_initialize_framespec
    105     oj_initialize_meshname_tessname
    106     oj_initialize_geomdir
    107
    108     source .gitlab-ci/oj_helper.sh MAKE_TAR_SCP_TAR $idir
    109 fi
    110 oj_initialize_dump
    111



::

    108 oj_tessname_from_meshname()
    109 {
    110    : opticks_juno.sh
    111    : can only create tessname after meshname exists so this needs to be done after the first run
    112    : solids that can be auto detected as needing triangulation, eg those including G4Torus or G4CutTubs,
    113    : need not and should not be included in the regexp.
    114    : It would be better to not have need any manual triangulation config
    115    :
    116    : TODO: add way to auto determine that the remaining solids named below need triangulation
    117    :
    118    local oneline="import sys,re;ptn=re.compile('^(solidSJ|solidXJ).*');print('\n'.join([('%s' if ptn.match(line) else '#%s') % line.rstrip() for line in sys.stdin.readlines()]))"
    119    python -c "$oneline" -
    120    return 0
    121 }







Workaround while work out how to handle listnode bbox in RPN
-------------------------------------------------------------

* simply skip the assert for now

::

    623 CSGNode* CSGImport::importNode(int nodeOffset, int partIdx, const snode& node, const sn* nd)
    624 {
    625     bool write_rpn = fd->isRPN() && fd->rpn && nd ;
    626
    627     // Althout nodeOffset and partIdx are not used below - they are handy for debug certainty
    628     if(nd) assert( node.lvid == nd->lvid );
    629
    630     int  typecode = nd ? nd->typecode : CSG_ZERO ;
    631     bool leaf = CSG::IsLeaf(typecode) ;  // primitive (ie not operator or listnode)
    632
    633     bool external_bbox_is_expected = CSG::ExpectExternalBBox(typecode);
    634     // CSG_CONVEXPOLYHEDRON, CSG_CONTIGUOUS, CSG_DISCONTIGUOUS, CSG_OVERLAP
    635
    636     bool expect = external_bbox_is_expected == false ;
    637     LOG_IF(fatal, !expect)
    638         << " NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED "
    639         << " for node of type " << CSG::Name(typecode)
    640         << " nd.lvid " << ( nd ? nd->lvid : -1 )
    641         << " write_rpn " << ( write_rpn ? "YES" : "NO " )
    642         ;
    643
    644     if(!write_rpn)
    645     {
    646         assert(expect);
    647         if(!expect) std::raise(SIGINT);
    648     }


::

    ~/o/CSG/tests/CSGFoundry_CreateFromSimTest.sh
    ...
    ]CSGImport::importPrim_RPN.dump_LVID:1 node.lvid 119 LVID 119 name s_EMFcoil_holder_ring17_seg1 soname s_EMFcoil_holder_ring17_seg1
    2026-11-17 15:00:47.120 FATAL [1254437] [CSGImport::importNode@634]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type discontiguous nd.lvid 270
    CSGFoundry_CreateFromSimTest: /home/blyth/opticks/CSG/CSGImport.cc:639: CSGNode* CSGImport::importNode(int, int, const snode&, const sn*): Assertion `expect' failed.
    /home/blyth/o/CSG/tests/CSGFoundry_CreateFromSimTest.sh: line 80: 1254437 Aborted                 (core dumped) $bin
    /home/blyth/o/CSG/tests/CSGFoundry_CreateFromSimTest.sh run error
    [lo] A[blyth@localhost tests]$


::


    ~/o/CSG/tests/CSGFoundry_CreateFromSimTest.sh dbg
    ...
    sn::desc pid  536 idx  536 typecode   2 num_node   5 num_leaf   3 maxdepth  2 is_positive_form Y lvid 119 xform Y tag in
    sn::render mode 8 IDX
       536

    537      538

          539   540





    ]rt.render
    ]CSGImport::importPrim_RPN.dump_LVID:1 node.lvid 119 LVID 119 name s_EMFcoil_holder_ring17_seg1 soname s_EMFcoil_holder_ring17_seg1
    2026-11-17 15:03:57.175 FATAL [1254848] [CSGImport::importNode@634]  NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED  for node of type discontiguous nd.lvid 270
    CSGFoundry_CreateFromSimTest: /home/blyth/opticks/CSG/CSGImport.cc:639: CSGNode* CSGImport::importNode(int, int, const snode&, const sn*): Assertion `expect' failed.

    Thread 1 "CSGFoundry_Crea" received signal SIGABRT, Aborted.
    0x00007ffff608bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 nvidia-driver-cuda-libs-610.43.02-1.el9.x86_64 openssl-libs-3.5.1-7.el9_7.x86_64
    (gdb) bt
    #0  0x00007ffff608bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff603eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff6028833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff602875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff6037886 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007ffff7d3f9a3 in CSGImport::importNode (this=0x16396690, nodeOffset=13145, partIdx=3, node=..., nd=0xfdc91f0) at /home/blyth/opticks/CSG/CSGImport.cc:639
    #6  0x00007ffff7d3f51e in CSGImport::importPrim_RPN (this=0x16396690, primIdx=1, node=...) at /home/blyth/opticks/CSG/CSGImport.cc:533
    #7  0x00007ffff7d3e169 in CSGImport::importSolidFactor (this=0x16396690, ridx=8, ridx_type=70 'F') at /home/blyth/opticks/CSG/CSGImport.cc:263
    #8  0x00007ffff7d3d5c9 in CSGImport::importSolid (this=0x16396690) at /home/blyth/opticks/CSG/CSGImport.cc:104
    #9  0x00007ffff7d3d3ae in CSGImport::import_ (this=0x16396690) at /home/blyth/opticks/CSG/CSGImport.cc:67
    #10 0x00007ffff7c95027 in CSGFoundry::CreateFromSim () at /home/blyth/opticks/CSG/CSGFoundry.cc:3067
    #11 0x0000000000407b49 in main (argc=1, argv=0x7fffffffb748) at /home/blyth/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.cc:29
    (gdb)







