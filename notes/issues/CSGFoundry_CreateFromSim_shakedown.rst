CSGFoundry_CreateFromSim_shakedown
====================================

Compare geometries from X4/GGeo and stree routes::

    ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh
    ~/opticks/CSG/tests/CSGFoundry_CreateFromSimTest.sh ana



A/B comparison of CSGFoundry geometries
------------------------------------------

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


+---------------------------------------------------------------+
| A:X4/GGeo  by ntds3_noxj and pulled back with "GEOM get"      |
+---------------------------------------------------------------+
| B:SSim/stree                                                  |
+---------------------------------------------------------------+




::


    In [17]: print(ab.descLVDetail(0))

    CSGFoundryAB.descLVDetail
    descLV lvid:0 meshname:sTopRock_domeAir pidxs:[4]
    pidx    4 lv   0 pxl    4 :                                   sTopRock_domeAir : no     6 nn    3 tcn 2:intersection 105:cylinder 110:!box3 tcs [  2 105 110] : bnd 3 : Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air 
    a.node[6:6+3].reshape(-1,16)[:,:6] # descNodeParam 
    [[     0.      0.      0.      0.      0.      0.]
     [     0.      0.      0.  26760. -28125.  28125.]
     [ 75040. 107040.  62250.      0.      0.      0.]]
    a.node[6:6+3].reshape(-1,16)[:,8:14] # descNodeBB 
    [[    -0.     -0.     -0.      0.      0.      0.]
     [-25000. -26760.  -4770.  31250.  26760.  48750.]
     [-28000. -53520. -42290.  34250.  53520.  32750.]]
    a.node[6:6+3].reshape(-1,16).view(np.int32)[:,6:8] # descNodeBoundaryIndex 
    [[3 6]
     [3 7]
     [3 8]]
    a.node[6:6+3].reshape(-1,16).view(np.int32)[:,14:16] # descNodeTCTran 
    [[          2           0]
     [        105           6]
     [        110 -2147483641]]
    a.node[6:6+3].reshape(-1,16).view(np.int32)[:,14:16] & 0x7ffffff  # descNodeTCTran 
    [[  2   0]
     [105   6]
     [110   7]]

    descLV lvid:0 meshname:sTopRock_domeAir pidxs:[4]
    pidx    4 lv   0 pxl    4 :                                   sTopRock_domeAir : no     6 nn    3 tcn 3:difference 105:cylinder 110:box3 tcs [  3 105 110] : bnd 3 : Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air 
    b.node[6:6+3].reshape(-1,16)[:,:6] # descNodeParam 
    [[     0.      0.      0.      0.      0.      0.]
     [     0.      0.      0.  26760. -28125.  28125.]
     [ 75040. 107040.  62250.      0.      0.      0.]]
    b.node[6:6+3].reshape(-1,16)[:,8:14] # descNodeBB 
    [[     0.      0.      0.      0.      0.      0.]
     [-26760. -26760. -28125.  26760.  26760.  28125.]
     [-37520. -53520. -31125.  37520.  53520.  31125.]]
    b.node[6:6+3].reshape(-1,16).view(np.int32)[:,6:8] # descNodeBoundaryIndex 
    [[3 6]
     [3 7]
     [3 8]]
    b.node[6:6+3].reshape(-1,16).view(np.int32)[:,14:16] # descNodeTCTran 
    [[  3   0]
     [105   6]
     [110   7]]
    b.node[6:6+3].reshape(-1,16).view(np.int32)[:,14:16] & 0x7ffffff  # descNodeTCTran 
    [[  3   0]
     [105   6]
     [110   7]]



typecode difference, complements : where to positivize ?
------------------------------------------------------------

* B typecode is difference, A is intersection with complement in the leaf
* HMM: where to positivize ? Where does X4/GGeo do that ?

::

    epsilon:opticks blyth$ opticks-fl positivize
    ./integration/tests/tboolean.bash
    ./sysrap/tests/sn_test.cc
    ./sysrap/sn.h
    ./CSG_GGeo/CSG_GGeo_Convert.cc
    ./analytic/csg.py
    ./analytic/sc.py
    ./analytic/treebuilder.py
    ./npy/NTreePositive.hpp
    ./npy/tests/NTreeBalanceTest.cc
    ./npy/tests/NTreePositiveTest.cc
    ./npy/NTreeProcess.cpp
    ./npy/NTreeBalance.cpp
    ./npy/NTreePositive.cpp
    epsilon:opticks blyth$ 

    epsilon:opticks blyth$ opticks-fl NTreePositive
    ./sysrap/sn.h
    ./om.bash
    ./CSG_GGeo/CSG_GGeo_Convert.cc
    ./npy/NTreePositive.hpp
    ./npy/CMakeLists.txt
    ./npy/tests/CMakeLists.txt
    ./npy/tests/NTreeBalanceTest.cc
    ./npy/tests/NTreePositiveTest.cc
    ./npy/NTreeProcess.hpp
    ./npy/NTreeProcess.cpp
    ./npy/NTreePositive.cpp
    epsilon:opticks blyth$ 


::

    153 template <typename T>
    154 void NTreeProcess<T>::init()
    155 {
    ...
    162     positiver = new NTreePositive<T>(root) ;  // inplace changes operator types and sets complements on primitives


    1205 GMesh* X4PhysicalVolume::ConvertSolid_FromRawNode( const Opticks* ok, int lvIdx, int soIdx, const G4VSolid* const solid, const char* son     ame, const char* lvname, bool balance_deep_tree,
    1206      nnode* raw)
    1207 {
    1208     bool is_x4balanceskip = ok->isX4BalanceSkip(lvIdx) ;
    1209     bool is_x4polyskip = ok->isX4PolySkip(lvIdx);   // --x4polyskip 211,232
    1210     bool is_x4nudgeskip = ok->isX4NudgeSkip(lvIdx) ;
    1211     bool is_x4pointskip = ok->isX4PointSkip(lvIdx) ;
    1212     bool do_balance = balance_deep_tree && !is_x4balanceskip ;
    1213 
    1214     nnode* root = do_balance ? NTreeProcess<nnode>::Process(raw


::

    272 inline void U4Solid::init_Sphere()
    273 {
    274     int outer = init_Sphere_('O');  assert( outer > -1 );
    275     int inner = init_Sphere_('I');
    276     root = inner == -1 ? outer : snd::Boolean( CSG_DIFFERENCE, outer, inner ) ;
    277 }



need to decide : sn vs snd vs sn+snd ?
-------------------------------------------

* need sn for flexible handling 
* what does snd have that sn doesnt ? 



a nidx
--------

nidx increments from 0 to 15926 then takes a dive
repeatedly incrementing from 0. This is presumably the repeated unbalanced
in the GGeo geometry.

::

    In [12]: nidx = a.node[:,1,3].view(np.int32)   # increment from zero up to 15926 then start

    In [31]: nidx[15900:15930]
    Out[31]:
    array([15900, 15901, 15902, 15903, 15904, 15905, 15906, 15907, 15908, 15909, 15910, 15911, 15912, 15913, 15914, 15915, 15916, 15917, 15918, 15919, 15920, 15921, 15922, 15923, 15924, 15925, 15926,
               0,     1,     2], dtype=int32)


    In [35]: nidx[15927:]
    Out[35]:
    array([  0,   1,   2,   3,   4,   5,   6,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
            31,  32,  33,  34,  35,  36,  37,  38,  39,  40,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
            28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
            66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
            26,  27,   0,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,   0,   1,   2,   3,
             4,   5,   6,   0,   1,   2,   3,   4,   5,   6,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
            28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
            66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
           104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129], dtype=int32)



tran diverges in the last 800 or so
-------------------------------------

::

    In [3]: (a.tran[:6672]-b.tran[:6672]).max()
    Out[3]: 0.001953125

    In [4]: a.tran.shape
    Out[4]: (7557, 4, 4)

    In [5]: b.tran.shape
    Out[5]: (7557, 4, 4)


where are the current bbox coming from
-----------------------------------------

Need to follow CSG_GGeo_Convert::convertNode for defining bbox
and sometimes transforming it.


snd has no complement, sn does
---------------------------------




