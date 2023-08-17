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


After adding sn.h features to bring it up to snd.hh making some checks of equivalence
----------------------------------------------------------------------------------------

::

      67 struct _sn
      68 {
      69     int type ;         // 0
      70     int complement ;   // 1 
      71     int lvid ;         // 2
      72     int tv ;           // 3
      73     int pa ;           // 4
      74     int bb ;           // 5 
      75     int parent ;       // 6 
      76 
      77 #ifdef WITH_CHILD
      78     int sibdex ;       // 7     0-based sibling index 
      79     int num_child ;    // 8
      80     int first_child ;  // 9
      81     int next_sibling ; // 10  
      82     static constexpr const int NV = 11 ;
      83 #else
      84     int left ;         // 7
      85     int right ;        // 8
      86     static constexpr const int NV = 9 ;
      87 #endif
      88     std::string desc() const ;
      89     bool is_root_importable() const ;
      90 };



::

    ~/opticks/u4/tests/U4TreeCreateTest.sh ana

    In [3]: f._csg
    Out[3]: 
    _csg

    CMDLINE:/Users/blyth/opticks/u4/tests/U4TreeCreateTest.py
    _csg.base:/tmp/blyth/opticks/U4TreeCreateTest/stree/_csg

      : _csg.s_bb                                          :             (346, 6) : 0:05:55.896806 
      : _csg.sn                                            :             (551, 9) : 0:05:55.896988 
      : _csg.s_pa                                          :             (346, 6) : 0:05:55.896655 
      : _csg.NPFold_index                                  :                 (4,) : 0:05:55.897159 
      : _csg.s_tv                                          :            (205, 32) : 0:05:55.896499 

     min_stamp : 2023-08-16 10:40:59.006289 
     max_stamp : 2023-08-16 10:40:59.006949 
     dif_stamp : 0:00:00.000660 
     age_stamp : 0:05:55.896499 





Single parent node from virtual mask 
----------------------------------------

::

    In [18]: c.uparent[np.where(c.nparent == 1)]
    Out[18]: array([467, 501], dtype=int32)

    In [19]: c.sn[467]
    Out[19]: array([  1,   0, 108,  -1,  -1,  -1,  -1, 466,  -1], dtype=int32)

    In [20]: c.lvn[108]
    Out[20]: 'HamamatsuR12860sMask_virtual0x6163af0'

    In [8]: c.sn[501] 
    Out[8]: array([  1,   0, 117,  -1,  -1,  -1,  -1, 500,  -1], dtype=int32)

    In [9]: c.lv[501]
    Out[9]: 117

    In [10]: c.lvn[117]
    Out[10]: 'NNVTMCPPMTsMask_virtual0x61737a0'

    In [15]: np.c_[c.sn[wlv],wlv] 
    Out[15]: 
    array([[105,   0, 117,  -1, 308, 308, 500,  -1,  -1, 498],
           [105,   0, 117,  -1, 309, 309, 500,  -1,  -1, 499],
           [  1,   0, 117,  -1,  -1,  -1, 501, 498, 499, 500],
           [  1,   0, 117,  -1,  -1,  -1,  -1, 500,  -1, 501]])       ## FUNNY : LEFT ONLY 


    In [23]: wlv = np.where(c.lv == 108)[0]

    In [24]: np.c_[c.sn[wlv],wlv]
    Out[24]: 
    array([[105,   0, 108,  -1, 286, 286, 466,  -1,  -1, 464],
           [105,   0, 108,  -1, 287, 287, 466,  -1,  -1, 465],
           [  1,   0, 108,  -1,  -1,  -1, 467, 464, 465, 466],
           [  1,   0, 108,  -1,  -1,  -1,  -1, 466,  -1, 467]])        ## ANOTHER LEFT ONLY ?




Difference of 2 nodes between sn and snd
-------------------------------------------

::


    In [36]: w = np.where(sn[:551,2] != snd[:551,2] )[0]

    In [37]: w
    Out[37]: 
    array([468, 475, 486, 489, 492, 495, 496, 497, 498, 502, 503, 504, 505, 506, 507, 509, 510, 512, 513, 514, 516, 517, 518, 520, 521, 523, 524, 530, 531, 535, 536, 540, 541, 543, 544, 546, 547, 548,
           549, 550])

    In [38]: w.min()
    Out[38]: 468

    In [54]: np.c_[sn[:551,2],snd[:551,2],sn[:551,2]-snd[:551,2]][460:480]
    Out[54]: 
    array([[107, 107,   0],
           [107, 107,   0],
           [107, 107,   0],
           [107, 107,   0],
           [108, 108,   0],
           [108, 108,   0],
           [108, 108,   0],
           [108, 108,   0],
           [109, 108,   1],
           [109, 109,   0],
           [109, 109,   0],
           [109, 109,   0],
           [109, 109,   0],
           [109, 109,   0],
           [109, 109,   0],
           [110, 109,   1],
           [110, 110,   0],
           [110, 110,   0],
           [110, 110,   0],
           [110, 110,   0]], dtype=int32)



Missing CSG_CONE node in sn::

    In [49]: snd[np.where(snd[:,2]==108)]
    Out[49]: 
    array([[105,  -1, 108,  -1, 286, 286, 467,   0,   0,  -1, 465, 464,   2,   0,   0,   0,   0],
           [105,  -1, 108,  -1, 287, 287, 467,   1,   0,  -1,  -1, 465,   2,   0,   0,   0,   0],
           [108,  -1, 108,  -1, 288, 288, 468,   1,   0,  -1,  -1, 466,   1,   0,   0,   0,   0],
           [  1,  -1, 108,  -1,  -1,  -1, 468,   0,   2, 464, 466, 467,   1,   0,   0,   0,   0],
           [  1,  -1, 108,  -1,  -1,  -1,  -1,  -1,   2, 467,  -1, 468,   0,   0,   0,   0,   0]], dtype=int32)

    In [50]: sn[np.where(sn[:,2]==108)]
    Out[50]: 
    array([[105,   0, 108,  -1, 286, 286, 466,  -1,  -1],
           [105,   0, 108,  -1, 287, 287, 466,  -1,  -1],
           [  1,   0, 108,  -1,  -1,  -1, 467, 464, 465],
           [  1,   0, 108,  -1,  -1,  -1,  -1, 466,  -1]], dtype=int32)


Again a missing CSG_CONE node in sn::

    In [56]: sn[np.where(sn[:,2]==117)],1,snd[np.where(snd[:,2]==117)]
    Out[56]: 
    (array([[105,   0, 117,  -1, 308, 308, 500,  -1,  -1],
            [105,   0, 117,  -1, 309, 309, 500,  -1,  -1],
            [  1,   0, 117,  -1,  -1,  -1, 501, 498, 499],
            [  1,   0, 117,  -1,  -1,  -1,  -1, 500,  -1]], dtype=int32),
     1,
     array([[105,  -1, 117,  -1, 308, 308, 502,   0,   0,  -1, 500, 499,   2,   0,   0,   0,   0],
            [105,  -1, 117,  -1, 309, 309, 502,   1,   0,  -1,  -1, 500,   2,   0,   0,   0,   0],
            [108,  -1, 117,  -1, 310, 310, 503,   1,   0,  -1,  -1, 501,   1,   0,   0,   0,   0],
            [  1,  -1, 117,  -1,  -1,  -1, 503,   0,   2, 499, 501, 502,   1,   0,   0,   0,   0],
            [  1,  -1, 117,  -1,  -1,  -1,  -1,  -1,   2, 502,  -1, 503,   0,   0,   0,   0,   0]], dtype=int32))




Most Likely source of issue is sn::UnionTree vs snd::UnionTree
-------------------------------------------------------------------

::

    2102 inline sn* sn::Collection(std::vector<sn*>& prims ) // static
    2103 {
    2104     sn* n = nullptr ;
    2105     switch(VERSION)
    2106     {
    2107         case 0: n = UnionTree(prims)  ; break ;
    2108         case 1: n = Contiguous(prims) ; break ;
    2109     }
    2110     return n ;
    2111 }
    2112 
    2113 inline sn* sn::UnionTree(std::vector<sn*>& prims )
    2114 {
    2115     sn* n = CommonOperatorTree( prims, CSG_UNION );
    2116     return n ;
    2117 }


    1747 int snd::UnionTree(const std::vector<int>& prims )
    1748 {
    1749     int idx = sndtree::CommonTree_PlaceLeaves( prims, CSG_UNION );
    1750     return idx ;
    1751 }




WITH_SND debug
---------------

::

    U4Tree::initSolid U4Tree__IsFlaggedSolid_NAME [HamamatsuR12860sMask_virtual] flagged YES solid_level 1 name HamamatsuR12860sMask_virtual0x6163af0 lvid 108
    U4Polycone::collectPrims outside YES idx 464 is_cylinder YES
    U4Polycone::collectPrims outside YES idx 465 is_cylinder YES
    U4Polycone::collectPrims outside YES idx 466 is_cylinder NO 
    U4Polycone::init.WITH_SND outer_prims.size 3
    U4Polycone::init has_inner NO 
    U4Polycone::U4Polycone WITH_SND
    U4Polycone::desc level 1 num 4 rz 4
     num_R_inner   1 R_inner_min          0 R_inner_max          0
     num_R_outer   2 R_outer_min    132.025 R_outer_max     264.05
     num_Z         4 Z_min         -183.225 Z_max           200.05
     has_inner NO root 468 label WITH_SND
      0 RZ      0.000    264.050   -183.225
      1 RZ      0.000    264.050      0.000
      2 RZ      0.000    264.050    100.000
      3 RZ      0.000    132.025    200.050

    U4Solid::init_Polycone level 1
    U4Solid::desc level 1 solid Y lvid 108 depth   0 type   6 root  468 U4Solid::Tag(type) Pol name HamamatsuR12860sMask_virtual0x6163af0
    U4Solid::init SUCCEEDED desc: U4Solid::desc level 1 solid Y lvid 108 depth   0 type   6 root  468 U4Solid::Tag(type) Pol name HamamatsuR12860sMask_virtual0x6163af0



    U4Polycone::Convert

    sn::desc pid  479 idx  467 type   1 num_node   5 num_leaf   3 maxdepth  2 is_positive_form Y
    sn::render mode 0 MINIMAL
             o        
                      
       o        o     
                      
    o     o           
                      
                      
                      

    preorder  sn::desc_order [479 475 470 471 472 ]
    inorder   sn::desc_order [470 475 471 479 472 ]
    postorder sn::desc_order [470 471 475 472 479 ]
     ops = operators(0) 2
     CSG::MaskDesc(ops) : union 
     is_positive_form() : YES


    sn::desc pid  479 idx  467 type   1 num_node   5 num_leaf   3 maxdepth  2 is_positive_form Y
    sn::render mode 1 TYPECODE
             1        
                      
       1        108   
                      
    105   105         
                      
                      
                      



    sn::desc pid  479 idx  467 type   1 num_node   5 num_leaf   3 maxdepth  2 is_positive_form Y
    sn::render mode 2 DEPTH
             0        
                      
       0        0     
                      
    0     0           
                      
                      
                      



    sn::desc pid  479 idx  467 type   1 num_node   5 num_leaf   3 maxdepth  2 is_positive_form Y
    sn::render mode 3 SUBDEPTH
             0        
                      
       0        0     
                      
    0     0           
                      
                      
                      



    sn::desc pid  479 idx  467 type   1 num_node   5 num_leaf   3 maxdepth  2 is_positive_form Y
    sn::render mode 4 TYPETAG
             un       
                      
       un       co    
                      
    cy    cy          
                      
                      
                      



    sn::desc pid  479 idx  467 type   1 num_node   5 num_leaf   3 maxdepth  2 is_positive_form Y
    sn::render mode 5 PID
             479      
                      
       475      472   
                      
    470   471         
                      
                      
                      

    preorder  sn::desc_order [479 475 470 471 472 ]
    inorder   sn::desc_order [470 475 471 479 472 ]
    postorder sn::desc_order [470 471 475 472 479 ]
     ops = operators(0) 2
     CSG::MaskDesc(ops) : union 
     is_positive_form() : YES

    U4Solid::init_Polycone level 1
    U4Solid::desc level 1 solid Y lvid 108 depth   0 type   6 root  467 U4Solid::Tag(type) Pol name HamamatsuR12860sMask_virtual0x6163af0
    U4Solid::init SUCCEEDED desc: U4Solid::desc level 1 solid Y lvid 108 depth   0 type   6 root  467 U4Solid::Tag(type) Pol name HamamatsuR12860sMask_virtual0x6163af0
    U4Tree::init U4Tree::desc






    U4Tree::initSolid U4Tree__IsFlaggedSolid_NAME [HamamatsuR12860sMask_virtual] flagged YES solid_level 1 name HamamatsuR12860sMask_virtual0x6163af0 lvid 108
    U4Polycone::collectPrims outside YES idx 464 is_cylinder YES
    U4Polycone::collectPrims outside YES idx 465 is_cylinder YES
    U4Polycone::collectPrims outside YES idx 466 is_cylinder NO 
    U4Polycone::init.NOT-WITH_SND outer_prims.size 3
    U4Polycone::init has_inner NO 
    U4Polycone::U4Polycone NOT-WITH_SND
    U4Polycone::desc level 1 num 4 rz 4
     num_R_inner   1 R_inner_min          0 R_inner_max          0
     num_R_outer   2 R_outer_min    132.025 R_outer_max     264.05
     num_Z         4 Z_min         -183.225 Z_max           200.05
     has_inner NO root 467 label NOT-WITH_SND
      0 RZ      0.000    264.050   -183.225
      1 RZ      0.000    264.050      0.000
      2 RZ      0.000    264.050    100.000
      3 RZ      0.000    132.025    200.050

    U4Solid::init_Polycone level 1
    U4Solid::desc level 1 solid Y lvid 108 depth   0 type   6 root  467 U4Solid::Tag(type) Pol name HamamatsuR12860sMask_virtual0x6163af0
    U4Solid::init SUCCEEDED desc: U4Solid::desc level 1 solid Y lvid 108 depth   0 type   6 root  467 U4Solid::Tag(type) Pol name HamamatsuR12860sMask_virtual0x6163af0



::

    In [3]: w=np.where(sn[:,2]==108)[0]; np.c_[w,sn[w]]
    Out[3]: 
    array([[464, 105,   0, 108,  -1, 286, 286, 466,  -1,  -1],
           [465, 105,   0, 108,  -1, 287, 287, 466,  -1,  -1],
           [466,   1,   0, 108,  -1,  -1,  -1, 467, 464, 465],
           [467,   1,   0, 108,  -1,  -1,  -1,  -1, 466,  -1]])

    In [4]: w=np.where(snd[:,2]==108)[0]; np.c_[w,snd[w]]
    Out[4]: 
    array([[464, 105,  -1, 108,  -1, 286, 286, 467,   0,   0,  -1, 465, 464,   2,   0,   0,   0,   0],
           [465, 105,  -1, 108,  -1, 287, 287, 467,   1,   0,  -1,  -1, 465,   2,   0,   0,   0,   0],
           [466, 108,  -1, 108,  -1, 288, 288, 468,   1,   0,  -1,  -1, 466,   1,   0,   0,   0,   0],
           [467,   1,  -1, 108,  -1,  -1,  -1, 468,   0,   2, 464, 466, 467,   1,   0,   0,   0,   0],
           [468,   1,  -1, 108,  -1,  -1,  -1,  -1,  -1,   2, 467,  -1, 468,   0,   0,   0,   0,   0]])
                  tc   cmp  lv   xf   pa   bb parent sib  nc  fc  nexsib idx  depth   
    In [5]:                                                                                                         



Succeed to reproduce the issue in U4Polycone_test.sh 
--------------------------------------------------------


::

    epsilon:tests blyth$ ./U4Polycone_test.sh ana
    f

    CMDLINE:/Users/blyth/opticks/u4/tests/U4Polycone_test.py
    f.base:/tmp/U4Polycone_test

      : f.csg                                              :                 None : 0:07:43.592101 
      : f._csg                                             :                 None : 0:11:43.324385 

     min_stamp : 2023-08-16 16:01:01.997569 
     max_stamp : 2023-08-16 16:05:01.729853 
     dif_stamp : 0:03:59.732284 
     age_stamp : 0:07:43.592101 

    In [1]: f.csg
    Out[1]: 
    csg

    CMDLINE:/Users/blyth/opticks/u4/tests/U4Polycone_test.py
    csg.base:/tmp/U4Polycone_test/csg

      : csg.node                                           :              (5, 17) : 0:01:22.315148 
      : csg.aabb                                           :               (3, 6) : 0:01:22.314793 
      : csg.xform                                          :         (0, 2, 4, 4) : 0:01:22.314622 
      : csg.NPFold_index                                   :                 (4,) : 0:01:22.315369 
      : csg.param                                          :               (3, 6) : 0:01:22.314960 

     min_stamp : 2023-08-16 16:11:26.457238 
     max_stamp : 2023-08-16 16:11:26.457985 
     dif_stamp : 0:00:00.000747 
     age_stamp : 0:01:22.314622 

    In [2]: f.csg.node
    Out[2]: 
    array([[105,  -1,  -1,  -1,   0,   0,   3,   0,   0,  -1,   1,   0,  -1,   0,   0,   0,   0],
           [105,  -1,  -1,  -1,   1,   1,   3,   1,   0,  -1,  -1,   1,  -1,   0,   0,   0,   0],
           [108,  -1,  -1,  -1,   2,   2,   4,   1,   0,  -1,  -1,   2,  -1,   0,   0,   0,   0],
           [  1,  -1,  -1,  -1,  -1,  -1,   4,   0,   2,   0,   2,   3,  -1,   0,   0,   0,   0],
           [  1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   2,   3,  -1,   4,  -1,   0,   0,   0,   0]], dtype=int32)

    In [3]: f._csg.sn
    Out[3]: 
    array([[105,   0,  -1,  -1,   0,   0,   2,  -1,  -1],
           [105,   0,  -1,  -1,   1,   1,   2,  -1,  -1],
           [  1,   0,  -1,  -1,  -1,  -1,   3,   0,   1],
           [  1,   0,  -1,  -1,  -1,  -1,  -1,   2,  -1]], dtype=int32)

    In [4]:                                




::

    s_csg::brief total_size 10
     pa : s_pool::brief - count 3 pool.size 3 num_root 3
     bb : s_pool::brief - count 3 pool.size 3 num_root 3
     tv : s_pool::brief - count 0 pool.size 0 num_root 0
     n : s_pool::brief - count 10 pool.size 4 num_root 1



HMM can see the difficulty sn pid 2 needs to be hoiked upwards::

    sn::CommonOperatorTree after populate_leaves num_leaves 3 level 2

    sn::desc pid    9 idx    6 type   1 num_node   7 num_leaf   4 maxdepth  2 is_positive_form Y
    sn::render mode 5 PID
             9              
                            
       5           8        
                            
    0     1     2     7     
                            

Does prune do that ?


HMM: thats the impl difference, snd is using sndtree 


I need to do something closer to the below with sn.h 
Cannot just directly place into the tree due to hoiking problem. 

::

    071 /**
     72 sndtree::Build_r
     73 ------------------
     74 
     75 Builds snd tree based on the "skeleton" provided by the sn tree.
     76 
     77 Postorder visit after recursive call : so children reached before parents  
     78 
     79 **/
     80 
     81 inline int sndtree::Build_r(sn* n, int& num_leaves_placed, const std::vector<int>& leaves, int d )
     82 {
     83     int N = -1 ;
     84     if( n->is_operator() )
     85     {
     86         int op = n->type ;
     87         int nc = n->num_child();
     88         assert( nc == 2 );
     89         sn* l = n->get_child(0);
     90         sn* r = n->get_child(1);
     91         int L = Build_r(l, num_leaves_placed, leaves, d+1) ;
     92         int R = Build_r(r, num_leaves_placed, leaves, d+1) ;
     93         N = snd::Boolean( op, L, R );
     94     }
     95     else
     96     {
     97         N = leaves[num_leaves_placed] ;
     98         num_leaves_placed += 1 ;
     99     }
    100     return N ;
    101 }



U4Polycone_test.sh now giving node match with sn.h WITH_CHILD
----------------------------------------------------------------


::

    In [5]:
    epsilon:opticks blyth$ ./u4/tests/U4Polycone_test.sh ana
    f

    CMDLINE:/Users/blyth/opticks/u4/tests/U4Polycone_test.py
    f.base:/tmp/U4Polycone_test

      : f.csg                                              :                 None : 18:40:05.497211 
      : f._csg                                             :                 None : 18:44:05.229495 

     min_stamp : 2023-08-16 16:01:01.997569 
     max_stamp : 2023-08-16 16:05:01.729853 
     dif_stamp : 0:03:59.732284 
     age_stamp : 18:40:05.497211 
    snd[:,:11]
    [[105   0  -1  -1   0   0   3   0   0  -1   1]
     [105   0  -1  -1   1   1   3   1   0  -1  -1]
     [108   0  -1  -1   2   2   4   1   0  -1  -1]
     [  1   0  -1  -1  -1  -1   4   0   2   0   2]
     [  1   0  -1  -1  -1  -1  -1   0   2   3  -1]]
    sn 
    [[105   0  -1  -1   0   0   3   0   0  -1   1]
     [105   0  -1  -1   1   1   3   1   0  -1  -1]
     [108   0  -1  -1   2   2   4   1   0  -1  -1]
     [  1   0  -1  -1  -1  -1   4   0   2   0   2]
     [  1   0  -1  -1  -1  -1  -1   0   2   3  -1]]
    np.all( snd[:,:11] == sn )
    True




U4TreeCreateTest.sh with sn.h WITH_CHILD impl
--------------------------------------------------

::

    In [12]: np.unique( np.where( snd[:,:11] != sn )[1] )
    Out[12]: array([1, 7])

    # complement differs : always -1 in snd, always 0 in sn 
    # sibdex differs


    In [23]: np.unique( snd[:,7], return_counts=True )
    Out[23]: (array([-1,  0,  1], dtype=int32), array([139, 207, 207]))

    In [24]: np.unique( sn[:,7], return_counts=True )
    Out[24]: (array([0, 1], dtype=int32), array([346, 207]))


HMM: need to rerun the x4/ggeo on workstation and pullback 
to the complement/sibdex changes to snd.hh reflected 






::

    In [3]: np.where( snd[:,:11] != sn ) 
    Out[3]: 
    (array([  2,   5,   6,   9,  10,  13,  16,  17,  18,  19,  20,  21,  22,  23,  26,  29,  30,  31,  36,  41,  46,  51,  56,  61,  66,  71,  76,  81,  86,  91,  96, 101, 106, 111, 116, 121, 126, 131,
            136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 201, 206, 211, 216, 221, 226, 231, 236, 241, 246, 251, 256, 261, 266, 271, 276, 281, 286, 291, 296, 301, 306, 311, 316, 321,
            326, 331, 336, 341, 346, 351, 356, 361, 366, 371, 376, 381, 386, 391, 394, 397, 398, 399, 400, 407, 412, 419, 430, 433, 436, 439, 442, 445, 446, 449, 456, 463, 475, 486, 489, 492, 495, 496,
            497, 498, 504, 505, 506, 510, 513, 514, 517, 518, 521, 524, 531, 536, 541, 544, 547, 549, 552]),
     array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7]))



snd.hh side still has sibdex:-1 nodes that are all sibdex:0 on sn.h side
---------------------------------------------------------------------------

::

    In [16]: sn[np.where(snd[:,7] == -1)][:,7]
    Out[16]: 
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0], dtype=int32)


Parent is -1, so its the root nodes that have the unset sibdex on snd.hh side::

    In [18]: sn[np.where(snd[:,7] == -1)][:,6]
    Out[18]:
    array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=int32)

    In [19]: sn[np.where(snd[:,7] == -1)][:,6].max()
    Out[19]: -1

    In [20]: sn[np.where(snd[:,7] == -1)][:,6].min()
    Out[20]: -1


