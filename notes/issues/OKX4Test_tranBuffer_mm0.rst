OKX4Test_tranBuffer_mm0
=========================


Confirm the match
---------------------

::

    epsilon:0 blyth$ ab-;ab-t
    import os, numpy as np
    from opticks.ana.mesh import Mesh
    from opticks.ana.prim import Dir
    from opticks.sysrap.OpticksCSG import CSG_

    a_dir = "/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0"
    b_dir = "/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0"


    da = Dir(a_dir)
    db = Dir(b_dir)

    a_idpath = "/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103"
    b_idpath = "/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1"

    a_load = lambda _:np.load(os.path.join(a_dir, _))
    b_load = lambda _:np.load(os.path.join(b_dir, _))

    a = a_load("partBuffer.npy")
    ta = a_load("tranBuffer.npy")
    pa = a_load("primBuffer.npy")

    b = b_load("partBuffer.npy")
    tb = b_load("tranBuffer.npy")
    pb = b_load("primBuffer.npy")

    xb = b_load("idxBuffer.npy")

    ma = Mesh.make(a_idpath)
    mb = Mesh.make(b_idpath)


    cut = 0.1
    where_discrepant_tr = da.where_discrepant_tr(db, cut) 
    wd = np.array(where_discrepant_tr, dtype=np.uint32)
    lvd = np.unique(xb[wd][:,2])

    print " prim with discrepant_tr %d cut %s " % ( len(where_discrepant_tr), cut ) 


    detail = False

    lvs = set()

    for i in where_discrepant_tr:

        primIdx = i 
        _,soIdx,lvIdx,height = xb[i]
        name = ma.idx2name[lvIdx]

        print " %s primIdx:%3d soIdx:%3d lvIdx:%3d height:%d name:%s  %s " % ( "-" * 30, primIdx, soIdx,lvIdx,height, name,   "-" * 60 )
        lvs.add(lvIdx)
        if detail:
            dap = da.prims[i]
            dbp = db.prims[i]
            print dap.tr_maxdiff(dbp)
            print dap
            print dbp
            print
            print
        pass
    pass


    lvs = np.array( sorted(list(lvs)), dtype=np.uint32 )
    print "lvs: %s " % repr(lvs)




    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-t.py
    [2018-07-03 16:31:04,871] p69763 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
    [2018-07-03 16:31:05,851] p69763 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
    [2018-07-03 16:31:06,850] p69763 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
    [2018-07-03 16:31:06,851] p69763 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
     prim with discrepant_tr 0 cut 0.1 
    lvs: array([], dtype=uint32) 

    In [1]: 



    In [15]: tab = np.max( ta[:,0] - tb[:,0] , axis=(1,2)) 

    In [16]: np.where( tab > 0.1 )
    Out[16]: (array([], dtype=int64),)

    In [17]: np.where( tab > 1e-8 )
    Out[17]: 
    (array([ 347,  348,  349,  350,  351,  352,  353,  354,  355,  356,  357,  358,  359,  360,  361,  362,  363,  364,  365,  366,  367,  403,  404,  405,  406,  407,  408,  409,  410,  411,  412,  413,
             414,  415,  416,  417,  418,  419,  420,  421,  422,  423,  459,  460,  461,  462,  463,  464,  465,  466,  467,  468,  469,  470,  471,  472,  473,  474,  475,  476,  477,  478,  479,  515,
             516,  517,  518,  519,  520,  521,  522,  523,  524,  525,  526,  527,  528,  529,  530,  531,  532,  533,  534,  535,  669,  670,  671,  672,  673,  675,  676,  677,  678,  679,  681,  682,
             683,  684,  685,  687,  688,  690,  691, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1689, 1690, 1691, 1692,
            1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759,
            1760, 1761, 1762, 1763, 1764, 1765, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1955, 1956, 1957, 1958, 1959,
            1961, 1962, 1963, 1964, 1965, 1967, 1968, 1969, 1970, 1971, 1973, 1974, 1976, 1977, 2612, 2613, 2616, 2617, 2624, 2625, 2632, 2633, 2640, 2641, 2744, 2745, 2752, 2753, 2760, 2761, 2768, 2769,
            2940, 2941, 2948, 2949, 2956, 2957, 2964, 2965, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3156, 3157, 3160, 3161, 3164, 3165, 3168, 3169,
            3172, 3173, 3176, 3177, 3180, 3181, 3184, 3185, 3252, 3253, 3260, 3261, 3268, 3269, 3276, 3277, 3412, 3413, 3416, 3417, 3420, 3421, 3424, 3425, 3428, 3429, 3432, 3433, 3436, 3437, 3440, 3441,
            3632, 3633, 3636, 3637, 3640, 3641, 3644, 3645, 3648, 3649, 3652, 3653, 3656, 3657, 3660, 3661, 3684, 3685, 3692, 3693, 3696, 3697, 3700, 3701, 3704, 3705, 3708, 3709, 3764, 3765, 3772, 3773,
            3776, 3777, 3780, 3781, 3784, 3785, 3788, 3789]),)

    In [18]: np.where( tab > 1e-5 )
    Out[18]: (array([], dtype=int64),)

    In [19]: np.where( tab > 1e-6 )
    Out[19]: (array([], dtype=int64),)

    In [20]: 





Collecting boolean transforms in py and C++ and comparing 
-----------------------------------------------------------------

* reveals that transposing the boolean rotation transfrom 
  gives a match beteen a and b 


::


    import os, numpy as np

    aa = np.load(os.path.expandvars("/tmp/blyth/opticks/Boolean_all_transforms.npy"))
    bb = np.load(os.path.expandvars("/tmp/blyth/opticks/X4Transform3D.npy"))

    assert np.all( aa[:,3] == bb[:,3] )  ## translation matches
    assert aa.shape == bb.shape 

    cut = 1e-19
    discrep = 0 

    for _ in range(len(aa)):
        a = aa[_][:3,:3]
        b = bb[_][:3,:3]
        ab = a - b.T       ## ahha : transposing the rotation gives agreement
        mx = np.max( ab )
        mi = np.min( ab ) 

        if abs(mx) < cut and abs(mi) < cut:
            pass
        else:
            discrep +=  1
            print _, mx, mi
            print np.hstack([a,b])
        pass
    pass

    print " cut %s discrep %s " % ( cut, discrep ) 



    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-s.py
    12 5.9604645e-08 -5.9604645e-08
    [[ 0.8 -0.6  0.   0.8  0.6  0. ]
     [ 0.6  0.8  0.  -0.6  0.8  0. ]
     [ 0.   0.   1.   0.   0.   1. ]]
    81 6.0786277e-18 -2.0554644e-16
    [[ 0.     -0.1274  0.9919  0.      0.     -1.    ]
     [ 0.      0.9919  0.1274 -0.1274  0.9919 -0.    ]
     [-1.     -0.      0.      0.9919  0.1274  0.    ]]
    82 0.0 -1.02948954e-16
    [[ 0.     -0.3797  0.9251  0.      0.     -1.    ]
     [ 0.      0.9251  0.3797 -0.3797  0.9251  0.    ]
     [-1.     -0.      0.      0.9251  0.3797  0.    ]]
    83 0.0 -7.620281e-17
    [[ 0.     -0.6062  0.7953  0.      0.     -1.    ]
     [ 0.      0.7953  0.6062 -0.6062  0.7953  0.    ]
     [-1.     -0.      0.      0.7953  0.6062  0.    ]]
    84 6.0786277e-18 -2.0554644e-16
    [[ 0.     -0.1274  0.9919  0.      0.     -1.    ]
     [ 0.      0.9919  0.1274 -0.1274  0.9919 -0.    ]
     [-1.     -0.      0.      0.9919  0.1274  0.    ]]
    85 0.0 -1.02948954e-16
    [[ 0.     -0.3797  0.9251  0.      0.     -1.    ]
     [ 0.      0.9251  0.3797 -0.3797  0.9251  0.    ]
     [-1.     -0.      0.      0.9251  0.3797  0.    ]]
    86 0.0 -7.620281e-17
    [[ 0.     -0.6062  0.7953  0.      0.     -1.    ]
     [ 0.      0.7953  0.6062 -0.6062  0.7953  0.    ]
     [-1.     -0.      0.      0.7953  0.6062  0.    ]]
    98 0.0 -5.9604645e-08
    [[-0.3827 -0.9239  0.     -0.3827  0.9239  0.    ]
     [ 0.9239 -0.3827  0.     -0.9239 -0.3827  0.    ]
     [ 0.      0.      1.      0.      0.      1.    ]]
    125 7.450581e-09 -7.450581e-09
    [[ 0.9941  0.      0.1088  0.9941  0.     -0.1088]
     [ 0.      1.      0.      0.      1.      0.    ]
     [-0.1088  0.      0.9941  0.1088  0.      0.9941]]
    126 7.450581e-09 -7.450581e-09
    [[ 0.9941  0.      0.1088  0.9941  0.     -0.1088]
     [ 0.      1.      0.      0.      1.      0.    ]
     [-0.1088  0.      0.9941  0.1088  0.      0.9941]]
    151 1.2246469e-16 -1.2246469e-16
    [[-1.  0. -0. -1.  0.  0.]
     [ 0.  1. -0. -0.  1.  0.]
     [ 0. -0. -1. -0.  0. -1.]]
    152 1.2246469e-16 -1.2246469e-16
    [[-1.  0. -0. -1.  0.  0.]
     [ 0.  1. -0. -0.  1.  0.]
     [ 0. -0. -1. -0.  0. -1.]]
    154 1.2246469e-16 -1.2246469e-16
    [[-1.  0. -0. -1.  0.  0.]
     [ 0.  1. -0. -0.  1.  0.]
     [ 0. -0. -1. -0.  0. -1.]]
    165 1.2246469e-16 -1.2246469e-16
    [[-1.  0. -0. -1.  0.  0.]
     [ 0.  1. -0. -0.  1.  0.]
     [ 0. -0. -1. -0.  0. -1.]]
    166 1.2246469e-16 -1.2246469e-16
    [[-1.  0. -0. -1.  0.  0.]
     [ 0.  1. -0. -0.  1.  0.]
     [ 0. -0. -1. -0.  0. -1.]]
    168 1.2246469e-16 -1.2246469e-16
    [[-1.  0. -0. -1.  0.  0.]
     [ 0.  1. -0. -0.  1.  0.]
     [ 0. -0. -1. -0.  0. -1.]]
    177 1.2246469e-16 -1.2246469e-16
    [[-1.  0. -0. -1.  0.  0.]
     [ 0.  1. -0. -0.  1.  0.]
     [ 0. -0. -1. -0.  0. -1.]]
    178 1.2246469e-16 -1.2246469e-16
    [[-1.  0. -0. -1.  0.  0.]
     [ 0.  1. -0. -0.  1.  0.]
     [ 0. -0. -1. -0.  0. -1.]]
    180 1.2246469e-16 -1.2246469e-16
    [[-1.  0. -0. -1.  0.  0.]
     [ 0.  1. -0. -0.  1.  0.]
     [ 0. -0. -1. -0.  0. -1.]]
     cut 1e-19 discrep 19 

     




12 lv : [ 36,  56,  57,  60,  63,  65,  67,  69,  70,  74, 131, 200]
------------------------------------------------------------------------


::

     28 template <typename T>
     29 T* NTreeProcess<T>::Process( T* root_ , unsigned soIdx, unsigned lvIdx )  // static
     30 {
     31     //if( LVList == NULL )
     32     //     LVList = new std::vector<unsigned> {25,  26,  29,  60,  68,  75,  77,  81,  85, 131};
     33     if( LVList == NULL )
     34          LVList = new std::vector<unsigned> {36,  56,  57,  60,  63,  65,  67,  69,  70,  74, 131, 200 } ;
     35 
     36     if( ProcBuffer == NULL ) ProcBuffer = NPY<unsigned>::make(0,4) ;



Dump those, note the simplest one, lvIdx 70::

    ------------------------------ primIdx:360 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 


    2018-07-03 14:42:04.902 INFO  [560388] [*NTreeProcess<nnode>::Process@40] before
    NTreeAnalyse height 1 count 3
          di    

      bo      co


    2018-07-03 14:42:04.902 INFO  [560388] [*NTreeProcess<nnode>::Process@55] after
    NTreeAnalyse height 1 count 3
          di    

      bo      co


    2018-07-03 14:42:04.902 INFO  [560388] [*NTreeProcess<nnode>::Process@56]  soIdx 81 lvIdx 70 height0 1 height1 1 ### LISTED



::

    In [1]: print da.prims[360]

    primIdx 360 idx array([4294967295, 4294967295, 4294967295, 4294967295], dtype=uint32) lvName - partOffset 2352 numParts 3 tranOffset 725 numTran 2 planOffset 288  
        Part   3  0       difference    35 MineralOil///StainlessSteel   tz:     0.000      
        Part  17  1             box3    35 MineralOil///StainlessSteel   tz: -7132.500       x:   120.000 y:    25.000 z:  4875.000    
        Part  19  2  convexpolyhedron    35 MineralOil///StainlessSteel   tz: -4695.000      
    array([[[     -0.5432,       0.8396,       0.    ,       0.    ],
            [     -0.8396,      -0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -19398.281 , -797660.8   ,   -7132.5   ,       1.    ]],

           [[     -0.    ,       0.    ,      -1.    ,       0.    ],
            [     -0.8396,      -0.5432,       0.    ,       0.    ],
            [     -0.5432,       0.8396,       0.    ,       0.    ],
            [ -19379.268 , -797690.2   ,   -4695.    ,       1.    ]]], dtype=float32)

    In [2]: print db.prims[360]

    primIdx 360 idx array([ 0, 81, 70,  1], dtype=uint32) lvName SstInnVerRibBase0xbf30b50 partOffset 2352 numParts 3 tranOffset 725 numTran 2 planOffset 288  
        Part   3  0       difference    35 MineralOil///StainlessSteel   tz:     0.000      
        Part  17  1             box3    35 MineralOil///StainlessSteel   tz: -7132.500       x:   120.000 y:    25.000 z:  4875.000    
        Part  19  2  convexpolyhedron    35 MineralOil///StainlessSteel   tz: -4695.000      
    array([[[     -0.5432,       0.8396,       0.    ,       0.    ],
            [     -0.8396,      -0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -19398.281 , -797660.8   ,   -7132.5   ,       1.    ]],

           [[     -0.    ,       0.    ,       1.    ,       0.    ],
            [     -0.8396,      -0.5432,       0.    ,       0.    ],
            [      0.5432,      -0.8396,       0.    ,       0.    ],
            [ -19379.268 , -797690.2   ,   -4695.    ,       1.    ]]], dtype=float32)

    In [3]: 



     1099     <box lunit="mm" name="SstInnVerRibBox0xbf310d8" x="120" y="25" z="4875"/>
     1100     <trd lunit="mm" name="SstInnVerRibCut0xbf31118" x1="100" x2="237.2" y1="27" y2="27" z="50.02"/>
     1101     <subtraction name="SstInnVerRibBase0xbf30b50">
     1102       <first ref="SstInnVerRibBox0xbf310d8"/>
     1103       <second ref="SstInnVerRibCut0xbf31118"/>
     1104       <position name="SstInnVerRibBase0xbf30b50_pos" unit="mm" x="-35.0050000000002" y="0" z="2437.5"/>
     1105       <rotation name="SstInnVerRibBase0xbf30b50_rot" unit="deg" x="0" y="-90" z="0"/>
     1106     </subtraction>


::

    180 void X4Solid::convertDisplacedSolid()
    181 {
    182     const G4DisplacedSolid* const disp = static_cast<const G4DisplacedSolid*>(m_solid);
    183     G4VSolid* moved = disp->GetConstituentMovedSolid() ;
    184     assert( dynamic_cast<G4DisplacedSolid*>(moved) == NULL ); // only a single displacement is handled
    185 
    186     X4Solid* xmoved = new X4Solid(moved);
    187     nnode* a = xmoved->root();
    188 
    189     glm::mat4 xf_disp = X4Transform3D::GetDisplacementTransform(disp);
    190     a->transform = new nmat4triple(xf_disp);
    191 
    192     // a->update_gtransforms();  
    193     // without update_transforms does nothing 
    194     // YES : but should be done for the full solid, not just from one of the nodes
    195     //LOG(error) << gpresent("\n      disp", xf_disp) ; 
    196 
    197     setRoot(a);
    198 }


    033 glm::mat4 X4Transform3D::GetDisplacementTransform(const G4DisplacedSolid* const disp)
     34 {   
     35     G4RotationMatrix rot = disp->GetObjectRotation();  
     36     G4ThreeVector    tla = disp->GetObjectTranslation();
     37     G4Transform3D    tra(rot,tla);
     38     return Convert( tra ) ;
     39 }
     40 
     41 glm::mat4 X4Transform3D::Convert( const G4Transform3D& t ) // static
     42 {   
     43     // M44T
     44     std::array<float, 16> a ; 
     45     a[ 0] = t.xx() ; a[ 1] = t.yx() ; a[ 2] = t.zx() ; a[ 3] = 0.f    ;
     46     a[ 4] = t.xy() ; a[ 5] = t.yy() ; a[ 6] = t.zy() ; a[ 7] = 0.f    ;
     47     a[ 8] = t.xz() ; a[ 9] = t.yz() ; a[10] = t.zz() ; a[11] = 0.f    ;
     48     a[12] = t.dx() ; a[13] = t.dy() ; a[14] = t.dz() ; a[15] = 1.f    ;
     49     
     50     unsigned n = checkArray(a);
     51     if(n > 0) LOG(fatal) << "nan/inf array values";
     52     assert( n == 0);
     53     
     54     return glm::make_mat4(a.data()) ;
     55 }













::

    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-t.py
    [2018-07-03 14:37:06,667] p60676 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 
    [2018-07-03 14:37:07,635] p60676 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 
    [2018-07-03 14:37:08,576] p60676 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
    [2018-07-03 14:37:08,577] p60676 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
     prim with discrepant_tr 180 cut 0.1 
     ------------------------------ primIdx: 34 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx: 35 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx: 36 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx: 37 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx: 38 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx: 39 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx: 40 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx: 41 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx:280 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:281 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:282 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:283 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:284 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:285 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:286 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:287 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:288 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:289 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:290 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:291 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:292 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:293 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:294 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:295 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:296 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:297 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:298 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:299 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:300 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:301 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:302 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:303 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:304 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:305 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:306 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:307 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:308 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:309 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:310 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:311 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:314 soIdx: 70 lvIdx: 57 height:4 name:TopESRCutHols0xbf9de10  ------------------------------------------------------------ 
     ------------------------------ primIdx:317 soIdx: 73 lvIdx: 60 height:3 name:BotESRCutHols0xbfa7368  ------------------------------------------------------------ 
     ------------------------------ primIdx:320 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:321 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:322 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:323 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:324 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:325 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:327 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:328 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:329 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:330 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:331 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:332 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:333 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:334 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:343 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:344 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:345 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:346 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:347 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:348 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:349 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:350 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:352 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:353 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:354 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:355 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:356 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:357 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:358 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:359 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:360 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:361 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:364 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:365 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:366 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:367 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:384 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:385 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:386 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:387 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:388 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:389 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:390 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:391 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:454 soIdx:126 lvIdx:131 height:2 name:AmCCo60AcrylicContainer0xc0b23b8  ------------------------------------------------------------ 
     ------------------------------ primIdx:542 soIdx:126 lvIdx:131 height:2 name:AmCCo60AcrylicContainer0xc0b23b8  ------------------------------------------------------------ 
     ------------------------------ primIdx:624 soIdx:126 lvIdx:131 height:2 name:AmCCo60AcrylicContainer0xc0b23b8  ------------------------------------------------------------ 
     ------------------------------ primIdx:734 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx:735 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx:736 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx:737 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx:738 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx:739 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx:740 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx:741 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
     ------------------------------ primIdx:980 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:981 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:982 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:983 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:984 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:985 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:986 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:987 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:988 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:989 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:990 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:991 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:992 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:993 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:994 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:995 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:996 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:997 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:998 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:999 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1000 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1001 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1002 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1003 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1004 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1005 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1006 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1007 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1008 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1009 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1010 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1011 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1014 soIdx: 70 lvIdx: 57 height:4 name:TopESRCutHols0xbf9de10  ------------------------------------------------------------ 
     ------------------------------ primIdx:1017 soIdx: 73 lvIdx: 60 height:3 name:BotESRCutHols0xbfa7368  ------------------------------------------------------------ 
     ------------------------------ primIdx:1020 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1021 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1022 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1023 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1024 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1025 soIdx: 74 lvIdx: 63 height:2 name:SstBotRib0xc26c4c0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1027 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1028 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1029 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1030 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1031 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1032 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1033 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1034 soIdx: 76 lvIdx: 65 height:3 name:SstBotCirRibBase0xc26e2d0  ------------------------------------------------------------ 
     ------------------------------ primIdx:1043 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:1044 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:1045 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:1046 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:1047 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:1048 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:1049 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:1050 soIdx: 78 lvIdx: 67 height:2 name:SstTopTshapeRib0xc272c80  ------------------------------------------------------------ 
     ------------------------------ primIdx:1052 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:1053 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:1054 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:1055 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:1056 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:1057 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:1058 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:1059 soIdx: 80 lvIdx: 69 height:3 name:SstTopCirRibBase0xc264f78  ------------------------------------------------------------ 
     ------------------------------ primIdx:1060 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:1061 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:1064 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:1065 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:1066 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:1067 soIdx: 81 lvIdx: 70 height:1 name:SstInnVerRibBase0xbf30b50  ------------------------------------------------------------ 
     ------------------------------ primIdx:1084 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:1085 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:1086 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:1087 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:1088 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:1089 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:1090 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:1091 soIdx: 85 lvIdx: 74 height:2 name:OavTopRib0xc0d5e10  ------------------------------------------------------------ 
     ------------------------------ primIdx:1154 soIdx:126 lvIdx:131 height:2 name:AmCCo60AcrylicContainer0xc0b23b8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1242 soIdx:126 lvIdx:131 height:2 name:AmCCo60AcrylicContainer0xc0b23b8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1324 soIdx:126 lvIdx:131 height:2 name:AmCCo60AcrylicContainer0xc0b23b8  ------------------------------------------------------------ 
     ------------------------------ primIdx:1771 soIdx:208 lvIdx:200 height:3 name:table_panel_box0xc00f558  ------------------------------------------------------------ 
     ------------------------------ primIdx:1786 soIdx:208 lvIdx:200 height:3 name:table_panel_box0xc00f558  ------------------------------------------------------------ 
    lvs: array([ 36,  56,  57,  60,  63,  65,  67,  69,  70,  74, 131, 200], dtype=uint32) 




180 prim with discrepant tr
------------------------------

::

    epsilon:opticks blyth$ ab-;ab-t
    import os, numpy as np
    from opticks.ana.mesh import Mesh
    from opticks.ana.prim import Dir
    from opticks.sysrap.OpticksCSG import CSG_

    a_dir = "/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0"
    b_dir = "/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0"


    da = Dir(a_dir)
    db = Dir(b_dir)

    a_idpath = "/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103"
    b_idpath = "/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1"

    a_load = lambda _:np.load(os.path.join(a_dir, _))
    b_load = lambda _:np.load(os.path.join(b_dir, _))

    a = a_load("partBuffer.npy")
    ta = a_load("tranBuffer.npy")
    pa = a_load("primBuffer.npy")

    b = b_load("partBuffer.npy")
    tb = b_load("tranBuffer.npy")
    pb = b_load("primBuffer.npy")

    xb = b_load("idxBuffer.npy")

    ma = Mesh.make(a_idpath)
    mb = Mesh.make(b_idpath)


    cut = 0.1
    where_discrepant_tr = da.where_discrepant_tr(db, cut) 
    wd = np.array(where_discrepant_tr, dtype=np.uint32)
    lvd = np.unique(xb[wd][:,2])

    print " prim with discrepant_tr %d cut %s " % ( len(where_discrepant_tr), cut ) 

    for i in where_discrepant_tr:

        primIdx = i 
        _,soIdx,lvIdx,height = xb[i]
        name = ma.idx2name[lvIdx]

        print " %s primIdx:%3d soIdx:%3d lvIdx:%3d height:%d name:%s  %s " % ( "-" * 30, primIdx, soIdx,lvIdx,height, name,   "-" * 60 )
        dap = da.prims[i]
        dbp = db.prims[i]
        print dap.tr_maxdiff(dbp)
        print dap
        print dbp
        print
        print
    pass





    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-t.py
    [2018-07-03 14:28:32,102] p60579 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 
    [2018-07-03 14:28:33,133] p60579 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 
    [2018-07-03 14:28:34,143] p60579 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
    [2018-07-03 14:28:34,143] p60579 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
     prim with discrepant_tr 180 cut 0.1 
     ------------------------------ primIdx: 34 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
    1.0

    primIdx 34 idx array([4294967295, 4294967295, 4294967295, 4294967295], dtype=uint32) lvName - partOffset 90 numParts 7 tranOffset 44 numTran 3 planOffset 0  
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  15  3             cone    22 LiquidScintillator///Acrylic   tz: -5512.780       z1:   -37.220 z2:    37.220 r1:  1520.393 r2:   100.000   
        Part  17  1             box3    22 LiquidScintillator///Acrylic   tz: -5495.500       x:  1420.393 y:    50.000 z:   109.000    
        Part  17  2             box3    22 LiquidScintillator///Acrylic   tz: -5454.625       x:   218.000 y:    50.000 z:    94.397    
    array([[[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -17639.375 , -800379.7   ,   -5495.5   ,       1.    ]],

           [[      0.4704,      -0.7271,       0.5   ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [     -0.2716,       0.4198,       0.866 ,       0.    ],
            [ -17292.07  , -800916.56  ,   -5454.625 ,       1.    ]],

           [[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]]], dtype=float32)

    primIdx 34 idx array([ 0, 49, 36,  2], dtype=uint32) lvName IavTopRib0xbf8e168 partOffset 90 numParts 7 tranOffset 44 numTran 3 planOffset 0  
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  15  3             cone    22 LiquidScintillator///Acrylic   tz: -5512.780       z1:   -37.220 z2:    37.220 r1:  1520.393 r2:   100.000   
        Part  17  1             box3    22 LiquidScintillator///Acrylic   tz: -5495.500       x:  1420.393 y:    50.000 z:   109.000    
        Part  17  2             box3    22 LiquidScintillator///Acrylic   tz: -5454.625       x:   218.000 y:    50.000 z:    94.397    
    array([[[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -17639.375 , -800379.7   ,   -5495.5   ,       1.    ]],

           [[      0.4704,      -0.7271,      -0.5   ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.2716,      -0.4198,       0.866 ,       0.    ],
            [ -17292.07  , -800916.56  ,   -5454.625 ,       1.    ]],

           [[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]]], dtype=float32)


     ------------------------------ primIdx: 35 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
    1.0

    primIdx 35 idx array([4294967295, 4294967295, 4294967295, 4294967295], dtype=uint32) lvName - partOffset 97 numParts 7 tranOffset 47 numTran 3 planOffset 0  
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  15  3             cone    22 LiquidScintillator///Acrylic   tz: -5512.780       z1:   -37.220 z2:    37.220 r1:  1520.393 r2:   100.000   
        Part  17  1             box3    22 LiquidScintillator///Acrylic   tz: -5495.500       x:  1420.393 y:    50.000 z:   109.000    
        Part  17  2             box3    22 LiquidScintillator///Acrylic   tz: -5454.625       x:   218.000 y:    50.000 z:    94.397    
    array([[[      0.9778,      -0.2096,       0.    ,       0.    ],
            [      0.2096,       0.9778,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -17287.256 , -799869.25  ,   -5495.5   ,       1.    ]],

           [[      0.8468,      -0.1815,       0.5   ,       0.    ],
            [      0.2096,       0.9778,       0.    ,       0.    ],
            [     -0.4889,       0.1048,       0.866 ,       0.    ],
            [ -16662.062 , -800003.25  ,   -5454.625 ,       1.    ]],

           [[      0.9778,      -0.2096,       0.    ,       0.    ],
            [      0.2096,       0.9778,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]]], dtype=float32)

    primIdx 35 idx array([ 0, 49, 36,  2], dtype=uint32) lvName IavTopRib0xbf8e168 partOffset 97 numParts 7 tranOffset 47 numTran 3 planOffset 0  
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  15  3             cone    22 LiquidScintillator///Acrylic   tz: -5512.780       z1:   -37.220 z2:    37.220 r1:  1520.393 r2:   100.000   
        Part  17  1             box3    22 LiquidScintillator///Acrylic   tz: -5495.500       x:  1420.393 y:    50.000 z:   109.000    
        Part  17  2             box3    22 LiquidScintillator///Acrylic   tz: -5454.625       x:   218.000 y:    50.000 z:    94.397    
    array([[[      0.9778,      -0.2096,       0.    ,       0.    ],
            [      0.2096,       0.9778,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -17287.256 , -799869.25  ,   -5495.5   ,       1.    ]],

           [[      0.8468,      -0.1815,      -0.5   ,       0.    ],
            [      0.2096,       0.9778,       0.    ,       0.    ],
            [      0.4889,      -0.1048,       0.866 ,       0.    ],
            [ -16662.062 , -800003.25  ,   -5454.625 ,       1.    ]],

           [[      0.9778,      -0.2096,       0.    ,       0.    ],
            [      0.2096,       0.9778,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]]], dtype=float32)


     ------------------------------ primIdx: 36 soIdx: 49 lvIdx: 36 height:2 name:IavTopRib0xbf8e168  ------------------------------------------------------------ 
    1.0





which parts use the different transforms
------------------------------------------

::

    In [92]: gta = a[:,3,3].view(np.uint32) & 0x7fffffff

    In [93]: gtb = b[:,3,3].view(np.uint32) & 0x7fffffff

    In [94]: gta
    Out[94]: array([1, 1, 1, ..., 1, 1, 1], dtype=uint32)

    In [95]: gta.max()
    Out[95]: 11

    In [96]: gta.min()
    Out[96]: 0

    In [97]: gtb.max()
    Out[97]: 11

    In [98]: gtb.min()
    Out[98]: 0

* gt range is so small because of the transform offset 

::

    In [99]: pa
    Out[99]: 
    array([[    0,     1,     0,     0],
           [    1,     1,     1,     0],
           [    2,     1,     2,     0],
           ...,
           [11981,     1,  5341,   672],
           [11982,     1,  5342,   672],
           [11983,     1,  5343,   672]], dtype=int32)

    In [100]: ta.shape
    Out[100]: (5344, 3, 4, 4)

    In [101]: tb.shape
    Out[101]: (5344, 3, 4, 4)


::

    In [103]: wdt[0]
    Out[103]: 
    array([  45,   48,   51,   54,   57,   60,   63,   66,  327,  328,  329,  330,  331,  332,  334,  335,  336,  337,  338,  339,  341,  342,  343,  344,  345,  346,  348,  349,  350,  351,  352,  353,
            355,  356,  357,  358,  359,  360,  362,  363,  364,  365,  366,  367,  369,  370,  371,  372,  373,  374,  376,  377,  378,  379,  380,  381,  383,  384,  385,  386,  387,  388,  390,  391,
            392,  393,  394,  395,  397,  398,  399,  400,  401,  402,  404,  405,  406,  407,  408,  409,  411,  412,  413,  414,  415,  416,  418,  419,  420,  421,  422,  423,  425,  426,  427,  428,
            429,  430,  432,  433,  434,  435,  436,  437,  439,  440,  441,  442,  443,  444,  446,  447,  448,  449,  450,  451,  453,  454,  455,  456,  457,  458,  460,  461,  462,  463,  464,  465,
            467,  468,  469,  470,  471,  472,  474,  475,  476,  477,  478,  479,  481,  482,  483,  484,  485,  486,  488,  489,  490,  491,  492,  493,  495,  496,  497,  498,  499,  500,  502,  503,
            504,  505,  506,  507,  509,  510,  511,  512,  513,  514,  516,  517,  518,  519,  520,  521,  523,  524,  525,  526,  527,  528,  530,  531,  532,  533,  534,  535,  537,  538,  539,  540,
            541,  542,  544,  545,  546,  547,  548,  549,  564,  585,  597,  598,  601,  602,  605,  606,  609,  610,  613,  614,  622,  625,  628,  634,  637,  640,  643,  695,  696,  699,  700,  703,
            704,  711,  712,  715,  716,  719,  720,  723,  724,  726,  734,  736,  738,  740,  758,  761,  764,  767,  770,  773,  776,  779,  898,  899, 1061, 1062, 1199, 1200, 1331, 1334, 1337, 1340,
           1343, 1346, 1349, 1352, 1613, 1614, 1615, 1616, 1617, 1618, 1620, 1621, 1622, 1623, 1624, 1625, 1627, 1628, 1629, 1630, 1631, 1632, 1634, 1635, 1636, 1637, 1638, 1639, 1641, 1642, 1643, 1644,
           1645, 1646, 1648, 1649, 1650, 1651, 1652, 1653, 1655, 1656, 1657, 1658, 1659, 1660, 1662, 1663, 1664, 1665, 1666, 1667, 1669, 1670, 1671, 1672, 1673, 1674, 1676, 1677, 1678, 1679, 1680, 1681,
           1683, 1684, 1685, 1686, 1687, 1688, 1690, 1691, 1692, 1693, 1694, 1695, 1697, 1698, 1699, 1700, 1701, 1702, 1704, 1705, 1706, 1707, 1708, 1709, 1711, 1712, 1713, 1714, 1715, 1716, 1718, 1719,
           1720, 1721, 1722, 1723, 1725, 1726, 1727, 1728, 1729, 1730, 1732, 1733, 1734, 1735, 1736, 1737, 1739, 1740, 1741, 1742, 1743, 1744, 1746, 1747, 1748, 1749, 1750, 1751, 1753, 1754, 1755, 1756,
           1757, 1758, 1760, 1761, 1762, 1763, 1764, 1765, 1767, 1768, 1769, 1770, 1771, 1772, 1774, 1775, 1776, 1777, 1778, 1779, 1781, 1782, 1783, 1784, 1785, 1786, 1788, 1789, 1790, 1791, 1792, 1793,
           1795, 1796, 1797, 1798, 1799, 1800, 1802, 1803, 1804, 1805, 1806, 1807, 1809, 1810, 1811, 1812, 1813, 1814, 1816, 1817, 1818, 1819, 1820, 1821, 1823, 1824, 1825, 1826, 1827, 1828, 1830, 1831,
           1832, 1833, 1834, 1835, 1850, 1871, 1883, 1884, 1887, 1888, 1891, 1892, 1895, 1896, 1899, 1900, 1908, 1911, 1914, 1920, 1923, 1926, 1929, 1981, 1982, 1985, 1986, 1989, 1990, 1997, 1998, 2001,
           2002, 2005, 2006, 2009, 2010, 2012, 2020, 2022, 2024, 2026, 2044, 2047, 2050, 2053, 2056, 2059, 2062, 2065, 2184, 2185, 2347, 2348, 2485, 2486, 3065, 3066, 3067, 3068, 3084, 3085, 3086, 3087])

::


    In [118]: pb[34:40]       ## partOffset, numPart, tranOffset, planOffset
    Out[118]: 
    array([[ 90,   7,  44,   0],        ## primIdx 34 
           [ 97,   7,  47,   0],
           [104,   7,  50,   0],
           [111,   7,  53,   0],
           [118,   7,  56,   0],
           [125,   7,  59,   0]], dtype=int32)

    In [119]: pb.shape
    Out[119]: (3116, 4)



    In [2]: print db.prims[34]

    primIdx 34 partOffset 90 numParts 7 tranOffset 44 planOffset 0  
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  15  3             cone    22 LiquidScintillator///Acrylic   tz: -5512.780       z1:   -37.220 z2:    37.220 r1:  1520.393 r2:   100.000   
        Part  17  1             box3    22 LiquidScintillator///Acrylic   tz: -5495.500       x:  1420.393 y:    50.000 z:   109.000    
        Part  17  2             box3    22 LiquidScintillator///Acrylic   tz: -5454.625       x:   218.000 y:    50.000 z:    94.397    

                 ** gt are 1-based, 0 meaning None 

    In [3]: print da.prims[34]

    primIdx 34 partOffset 90 numParts 7 tranOffset 44 planOffset 0  
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  15  3             cone    22 LiquidScintillator///Acrylic   tz: -5512.780       z1:   -37.220 z2:    37.220 r1:  1520.393 r2:   100.000   
        Part  17  1             box3    22 LiquidScintillator///Acrylic   tz: -5495.500       x:  1420.393 y:    50.000 z:   109.000    
        Part  17  2             box3    22 LiquidScintillator///Acrylic   tz: -5454.625       x:   218.000 y:    50.000 z:    94.397    




    In [15]: _,soIdx,lvIdx,height = xb[34]

    In [16]: soIdx
    Out[16]: 49

    In [17]: lvIdx
    Out[17]: 36

    In [18]: height 
    Out[18]: 2

    In [23]: ma
    Out[23]: <opticks.ana.mesh.Mesh at 0x10bda5e50>

    In [24]: ma.idx2name[lvIdx]
    Out[24]: u'IavTopRib0xbf8e168'



::

      box - box - cone


      614     <subtraction name="IavTopRib0xbf8e168">
      615       <first ref="IavTopRibBase-ChildForIavTopRib0xbf8df50"/>

          607     <subtraction name="IavTopRibBase-ChildForIavTopRib0xbf8df50">
          608       <first ref="IavTopRibBase0xbf8e718"/>
              605     <box lunit="mm" name="IavTopRibBase0xbf8e718" x="1420.39278882354" y="50" z="109"/>

          609       <second ref="IavTopRibSidCut0xbf8e890"/>
              606     <box lunit="mm" name="IavTopRibSidCut0xbf8e890" x="218" y="50" z="94.3967690125038"/>

          610       <position name="IavTopRibBase-ChildForIavTopRib0xbf8df50_pos" unit="mm" x="639.398817652391" y="0" z="40.875"/>
          611       <rotation name="IavTopRibBase-ChildForIavTopRib0xbf8df50_rot" unit="deg" x="0" y="30" z="0"/>

                        ## this 2nd small box : has the deviant transform

          612     </subtraction>

      616       <second ref="IavTopRibBotCut0xbf8e068"/>

          613     <cone aunit="deg" deltaphi="360" lunit="mm" name="IavTopRibBotCut0xbf8e068" rmax1="1520.39278882354" rmax2="100" rmin1="0" rmin2="0" startphi="0" z="74.4396317718873"/>

      617       <position name="IavTopRib0xbf8e168_pos" unit="mm" x="-810.196394411769" y="0" z="-17.2801841140563"/>
      618     </subtraction>










::

    In [9]: tb[44+1-1,0]
    Out[9]: 
    array([[      0.5432,      -0.8396,       0.    ,       0.    ],
           [      0.8396,       0.5432,       0.    ,       0.    ],
           [      0.    ,       0.    ,       1.    ,       0.    ],
           [ -17639.375 , -800379.7   ,   -5495.5   ,       1.    ]], dtype=float32)

    In [12]: ta[44+1-1,0]
    Out[12]: 
    array([[      0.5432,      -0.8396,       0.    ,       0.    ],
           [      0.8396,       0.5432,       0.    ,       0.    ],
           [      0.    ,       0.    ,       1.    ,       0.    ],
           [ -17639.375 , -800379.7   ,   -5495.5   ,       1.    ]], dtype=float32)


    ## first discrepant transform

    In [10]: tb[44+2-1,0]
    Out[10]: 
    array([[      0.4704,      -0.7271,      -0.5   ,       0.    ],
           [      0.8396,       0.5432,       0.    ,       0.    ],
           [      0.2716,      -0.4198,       0.866 ,       0.    ],
           [ -17292.07  , -800916.56  ,   -5454.625 ,       1.    ]], dtype=float32)

    In [13]: ta[44+2-1,0]
    Out[13]: 
    array([[      0.4704,      -0.7271,      *0.5*   ,       0.    ],
           [      0.8396,       0.5432,       0.    ,       0.    ],
           [    *-0.2716*,     *0.4198*,      0.866 ,       0.    ],
           [ -17292.07  , -800916.56  ,   -5454.625 ,       1.    ]], dtype=float32)



    In [11]: tb[44+3-1,0]
    Out[11]: 
    array([[      0.5432,      -0.8396,       0.    ,       0.    ],
           [      0.8396,       0.5432,       0.    ,       0.    ],
           [      0.    ,       0.    ,       1.    ,       0.    ],
           [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]], dtype=float32)

    In [14]: ta[44+3-1,0]
    Out[14]: 
    array([[      0.5432,      -0.8396,       0.    ,       0.    ],
           [      0.8396,       0.5432,       0.    ,       0.    ],
           [      0.    ,       0.    ,       1.    ,       0.    ],
           [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]], dtype=float32)



    In [2]: db.prims[34].trans_
    Out[2]: 
    array([[[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -17639.375 , -800379.7   ,   -5495.5   ,       1.    ]],

           [[      0.4704,      -0.7271,      -0.5   ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.2716,      -0.4198,       0.866 ,       0.    ],
            [ -17292.07  , -800916.56  ,   -5454.625 ,       1.    ]],

           [[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]]], dtype=float32)

    In [3]: da.prims[34].trans_
    Out[3]: 
    array([[[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -17639.375 , -800379.7   ,   -5495.5   ,       1.    ]],

           [[      0.4704,      -0.7271,       0.5   ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [     -0.2716,       0.4198,       0.866 ,       0.    ],
            [ -17292.07  , -800916.56  ,   -5454.625 ,       1.    ]],

           [[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]]], dtype=float32)

    In [4]: da.prims[34].trans_ - db.prims[34].trans_
    Out[4]: 
    array([[[ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ]],

           [[ 0.    ,  0.    ,  1.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [-0.5432,  0.8396,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ]],

           [[ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ]]], dtype=float32)



::

    In [1]: print db.prims[34]

    primIdx 34 partOffset 90 numParts 7 tranOffset 44 numTran 3 planOffset 0  
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  15  3             cone    22 LiquidScintillator///Acrylic   tz: -5512.780       z1:   -37.220 z2:    37.220 r1:  1520.393 r2:   100.000   
        Part  17  1             box3    22 LiquidScintillator///Acrylic   tz: -5495.500       x:  1420.393 y:    50.000 z:   109.000    
        Part  17  2             box3    22 LiquidScintillator///Acrylic   tz: -5454.625       x:   218.000 y:    50.000 z:    94.397    
    array([[[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -17639.375 , -800379.7   ,   -5495.5   ,       1.    ]],

           [[      0.4704,      -0.7271,      -0.5   ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.2716,      -0.4198,       0.866 ,       0.    ],
            [ -17292.07  , -800916.56  ,   -5454.625 ,       1.    ]],

           [[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]]], dtype=float32)

    In [2]: print da.prims[34]

    primIdx 34 partOffset 90 numParts 7 tranOffset 44 numTran 3 planOffset 0  
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  15  3             cone    22 LiquidScintillator///Acrylic   tz: -5512.780       z1:   -37.220 z2:    37.220 r1:  1520.393 r2:   100.000   
        Part  17  1             box3    22 LiquidScintillator///Acrylic   tz: -5495.500       x:  1420.393 y:    50.000 z:   109.000    
        Part  17  2             box3    22 LiquidScintillator///Acrylic   tz: -5454.625       x:   218.000 y:    50.000 z:    94.397    
    array([[[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -17639.375 , -800379.7   ,   -5495.5   ,       1.    ]],

           [[      0.4704,      -0.7271,       0.5   ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [     -0.2716,       0.4198,       0.866 ,       0.    ],
            [ -17292.07  , -800916.56  ,   -5454.625 ,       1.    ]],

           [[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]]], dtype=float32)

    In [3]: 








    In [30]: ntran = np.ones( len(pa), dtype=np.uint32)

    In [31]: ntran[0:len(pa)-1] = pa[1:,2] - pa[:-1,2]    ## differencing primBuffer tranOffsets give numTran for all but the last, which is here set to 1 

          ## expect this will be the leaf count of the trees

    In [32]: ntran
    Out[32]: array([1, 1, 1, ..., 1, 1, 1], dtype=uint32)

    In [33]: ntran.min()
    Out[33]: 1

    In [34]: ntran.max()
    Out[34]: 11


    In [40]: ta[44:44+ntran[44],0]
    Out[40]: 
    array([[[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -17639.375 , -800379.7   ,   -5495.5   ,       1.    ]],

           [[      0.4704,      -0.7271,       0.5   ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [     -0.2716,       0.4198,       0.866 ,       0.    ],
            [ -17292.07  , -800916.56  ,   -5454.625 ,       1.    ]],

           [[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]]], dtype=float32)

    In [41]: 

    In [41]: tb[44:44+ntran[44],0]
    Out[41]: 
    array([[[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -17639.375 , -800379.7   ,   -5495.5   ,       1.    ]],

           [[      0.4704,      -0.7271,      -0.5   ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.2716,      -0.4198,       0.866 ,       0.    ],
            [ -17292.07  , -800916.56  ,   -5454.625 ,       1.    ]],

           [[      0.5432,      -0.8396,       0.    ,       0.    ],
            [      0.8396,       0.5432,       0.    ,       0.    ],
            [      0.    ,       0.    ,       1.    ,       0.    ],
            [ -18079.453 , -799699.44  ,   -5512.7803,       1.    ]]], dtype=float32)


    In [42]: tb[44:44+ntran[44],0] - ta[44:44+ntran[44],0]
    Out[42]: 
    array([[[ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ]],

           [[ 0.    ,  0.    , -1.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.5432, -0.8396,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ]],

           [[ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ]]], dtype=float32)






comparing transforms
---------------------

Have triplets of transforms : t, v, q   (transform, inverse, inverse transposed )

::

    In [15]: ta.shape
    Out[15]: (5344, 3, 4, 4)

    In [16]: tb.shape
    Out[16]: (5344, 3, 4, 4)

    In [12]: np.max( ta[:,0] - tb[:,0] )     ## difference up to 2, suggests rotation problem
    Out[12]: 2.0

    In [81]: np.max(ta[:,0,3] - tb[:,0,3])   ## translation matches : this suggests the problem is within the shapes, ie not at structural level 
    Out[81]: 0.0

    In [13]: np.max( ta[:,1] - tb[:,1] )   ## big difference in the inverse
    Out[13]: 1597900.8

    In [14]: np.max( ta[:,2] - tb[:,2] )   ## and inverse-transposed
    Out[14]: 1597900.8


    In [38]: dt = ta - tb 

    In [39]: dt.shape
    Out[39]: (5344, 3, 4, 4)

    In [40]: dtmx = np.max( dt[:,0], axis=(1,2) )   ## max deviations between the 4x4 matrices

    In [41]: dtmx.shape
    Out[41]: (5344,)

    In [50]: wdt = np.where( dtmx > 0.5 )     ## tranform buffer index locations of deviants

    In [53]: wdt[0].shape                  ## ~10% of deviants 
    Out[53]: (512,)


Early (top level?) transforms with rotation sign difference, later rotations
totally different, as problems will be multipled down the tree.

Some 10% of transforms with rotation differences.

Hmm which parts use the transforms with differences ? Where does the rot start ?


::

    In [76]: for _ in range(len(wdt[0])):print np.hstack( [ta[:,0][wdt][_], tb[:,0][wdt][_] ]) 

    [[      0.4704      -0.7271       0.5          0.           0.4704      -0.7271      -0.5          0.    ]
     [      0.8396       0.5432       0.           0.           0.8396       0.5432       0.           0.    ]
     [     -0.2716       0.4198       0.866        0.           0.2716      -0.4198       0.866        0.    ]
     [ -17292.07   -800916.56     -5454.625        1.      -17292.07   -800916.56     -5454.625        1.    ]]
    [[      0.8468      -0.1815       0.5          0.           0.8468      -0.1815      -0.5          0.    ]
     [      0.2096       0.9778       0.           0.           0.2096       0.9778       0.           0.    ]
     [     -0.4889       0.1048       0.866        0.           0.4889      -0.1048       0.866        0.    ]
     [ -16662.062  -800003.25     -5454.625        1.      -16662.062  -800003.25     -5454.625        1.    ]]
    [[      0.7271       0.4704       0.5          0.           0.7271       0.4704      -0.5          0.    ]
     [     -0.5432       0.8396       0.           0.          -0.5432       0.8396       0.           0.    ]
     [     -0.4198      -0.2716       0.866        0.           0.4198       0.2716       0.866        0.    ]
     [ -16862.344  -798912.06     -5454.625        1.      -16862.344  -798912.06     -5454.625        1.    ]]
    [[      0.1815       0.8468       0.5          0.           0.1815       0.8468      -0.5          0.    ]
     [     -0.9778       0.2096       0.           0.          -0.9778       0.2096       0.           0.    ]
     [     -0.1048      -0.4889       0.866        0.           0.1048       0.4889       0.866        0.    ]
     [ -17775.592  -798282.06     -5454.625        1.      -17775.592  -798282.06     -5454.625        1.    ]]
    [[     -0.4704       0.7271       0.5          0.          -0.4704       0.7271      -0.5          0.    ]
     [     -0.8396      -0.5432       0.           0.          -0.8396      -0.5432       0.           0.    ]
     [      0.2716      -0.4198       0.866        0.          -0.2716       0.4198       0.866        0.    ]
     [ -18866.836  -798482.3      -5454.625        1.      -18866.836  -798482.3      -5454.625        1.    ]]
    [[     -0.8468       0.1815       0.5          0.          -0.8468       0.1815      -0.5          0.    ]
     [     -0.2096      -0.9778       0.           0.          -0.2096      -0.9778       0.           0.    ]
     [      0.4889      -0.1048       0.866        0.          -0.4889       0.1048       0.866        0.    ]

     ...


    [[      0.2067       0.9784       0.           0.          -0.9772       0.2125       0.           0.    ]
     [     -0.9784       0.2067       0.           0.          -0.2125      -0.9772       0.           0.    ]
     [      0.           0.           1.           0.           0.           0.           1.           0.    ]
     [ -15051.269  -808242.44     -9620.           1.      -15051.269  -808242.44     -9620.           1.    ]]
    [[      0.2125       0.9772       0.           0.          -0.9784       0.2067       0.           0.    ]
     [     -0.9772       0.2125       0.           0.          -0.2067      -0.9784       0.           0.    ]
     [      0.           0.           1.           0.           0.           0.           1.           0.    ]
     [ -11528.547  -805963.5      -9620.           1.      -11528.547  -805963.5      -9620.           1.    ]]





