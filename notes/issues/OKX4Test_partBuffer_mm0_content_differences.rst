OKX4Test_partBuffer_mm0_content_differences
=============================================

Got analytic buffer dimensions to match in :doc:`OKX4Test_partBuffer_mm0_global_differences` now
try for the contents.


Summary
----------

* getting NNodeNudger operational brings the primIdx with discrepancies down to 4, 
  corresponding to two lvIdx 24, 42  (iav, oav)  

  * these two were mesh joined ? but surely not relevant for analytic geometry ?


iav/oav
---------


::

     ------------------------------ primIdx:  3 soIdx: 33 lvIdx: 42 height:2 name:oav0xc2ed7c8  ------------------------------------------------------------ 
    1.0

    primIdx 3 partOffset 3 numParts 7 tranOffset 3 planOffset 0  
        Part   1  0            union    20      MineralOil///Acrylic   tz:     0.000      
        Part  12  1         cylinder    20      MineralOil///Acrylic   tz: -7141.500     r:   2000.000 z1: -1968.500 z2:  *1969.500*    +1mm  
        Part   1  0            union    20      MineralOil///Acrylic   tz:     0.000      
        Part  12  2         cylinder    20      MineralOil///Acrylic   tz: -9110.000     r:   2040.000 z1:  3937.000 z2:  4000.025   
        Part  15  2             cone    20      MineralOil///Acrylic   tz: -9110.000      

    primIdx 3 partOffset 3 numParts 7 tranOffset 3 planOffset 0  
        Part   1  0            union    20      MineralOil///Acrylic   tz:     0.000      
        Part  12  1         cylinder    20      MineralOil///Acrylic   tz: -7141.500     r:   2000.000 z1: -1968.500 z2:  1968.500   
        Part   1  0            union    20      MineralOil///Acrylic   tz:     0.000      
        Part  12  2         cylinder    20      MineralOil///Acrylic   tz: -9110.000     r:   2040.000 z1:  3937.000 z2:  4000.025   
        Part  15  2             cone    20      MineralOil///Acrylic   tz: -9110.000      


     ------------------------------ primIdx:  5 soIdx: 35 lvIdx: 24 height:2 name:iav0xc346f90  ------------------------------------------------------------ 
    1.0

    primIdx 5 partOffset 17 numParts 7 tranOffset 7 planOffset 0  
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -7107.500     r:   1560.000 z1: -1542.500 z2:  *1543.500*   +1mm   
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  12  2         cylinder    22 LiquidScintillator///Acrylic   tz: -8650.000     r:   1565.000 z1:  3085.000 z2:  3100.000   
        Part  15  2             cone    22 LiquidScintillator///Acrylic   tz: -8650.000      

    primIdx 5 partOffset 17 numParts 7 tranOffset 7 planOffset 0  
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -7107.500     r:   1560.000 z1: -1542.500 z2:  1542.500   
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  12  2         cylinder    22 LiquidScintillator///Acrylic   tz: -8650.000     r:   1565.000 z1:  3085.000 z2:  3100.000   
        Part  15  2             cone    22 LiquidScintillator///Acrylic   tz: -8650.000      




When does NNodeNudger operate ?
--------------------------------

::

     094 // ctor : booting from in memory node tree : cannot be const because of the nudger 
      95 NCSG::NCSG(nnode* root )
      96    :
      97    m_meta(NULL),
      98    m_treedir(NULL),
      99    m_index(0),
     100    m_surface_epsilon(SURFACE_EPSILON),
     101    m_verbosity(root->verbosity),
     102    m_usedglobally(true),   // changed to true : June 2018, see notes/issues/subtree_instances_missing_transform.rst
     103    m_root(root),
     104    m_points(NULL),
     105    m_uncoincide(make_uncoincide()),
     106    m_nudger(make_nudger()),
     107    m_nodes(NULL),
     108    m_transforms(NULL),


::

     158 NNodeNudger* NCSG::make_nudger() const
     159 {
     160    // when test running from nnode there is no metadata or treedir
     161    // LOG(info) << soname() << " treeNameIdx " << getTreeNameIdx() ; 
     162 
     163     NNodeNudger* nudger = new NNodeNudger(m_root, m_surface_epsilon, m_root->verbosity);
     164     return nudger ;
     165 }
     166 
     167 

::

     16 NNodeNudger::NNodeNudger(nnode* root_, float epsilon_, unsigned /*verbosity*/)
     17      :
     18      root(root_),
     19      epsilon(epsilon_),
     20      verbosity(SSys::getenvint("VERBOSITY",1)),
     21      znudge_count(0)
     22 {
     23     init();
     24 }
     25 
     26 void NNodeNudger::init()
     27 {
     28     root->collect_prim_for_edit(prim);
     29     update_prim_bb();
     30     collect_coincidence();
     31     uncoincide();
     32 }



::

    375 void NNodeNudger::znudge_umaxmin(NNodeCoincidence* coin)
    376 {
    377     assert(can_znudge_umaxmin(coin));
    378     assert(coin->fixed == false);
    379 
    380     nnode* i = coin->i ;
    381     nnode* j = coin->j ;
    382     const NNodePairType p = coin->p ;
    383 
    384     nbbox ibb = i->bbox();
    385     nbbox jbb = j->bbox();
    386 
    387     float dz(1.);
    388 
    389     assert( p == PAIR_MAXMIN );
    390 
    391     float zi = ibb.max.z ;
    392     float zj = jbb.min.z ;
    393     float ri = i->r2() ;
    394     float rj = j->r1() ;
    395 
    396     NNodeJoinType join = NNodeEnum::JoinClassify( zi, zj, epsilon );
    397     assert(join == JOIN_COINCIDENT);
    398 
    399     if( ri > rj )
    400     {
    401         j->decrease_z1( dz );
    402     }
    403     else
    404     {
    405         i->increase_z2( dz );
    406     }
    407 
    408     nbbox ibb2 = i->bbox();
    409     nbbox jbb2 = j->bbox();
    410 
    411     float zi2 = ibb2.max.z ;
    412     float zj2 = jbb2.min.z ;
    413 
    414     NNodeJoinType join2 = NNodeEnum::JoinClassify( zi2, zj2, epsilon );
    415     assert(join2 != JOIN_COINCIDENT);
    416 
    417     coin->fixed = true ;
    418 }




Added recording of NNodeNudger activity 
------------------------------------------


::

    In [1]: nn = np.load(os.path.expandvars("$TMP/NNodeNudger.npy"))

    In [7]: nn[np.where( nn[:,3] > 0 )]
    Out[7]: 
    array([[ 42,   3,   1,   1],
           [ 37,   3,   1,   1],
           [ 24,   3,   1,   1],
           [ 22,   3,   1,   1],
           [ 25,   2,   1,   1],
           [ 26,   2,   1,   1],
           [ 29,   3,   2,   1],
           [ 54,   2,   1,   1],
           [ 68,   2,   1,   1],
           [ 75,   3,   2,   2],
           [ 77,   3,   2,   2],
           [ 81,   3,   2,   1],
           [ 85,   3,   2,   1],
           [130,   3,   2,   2],
           [145,   6,   5,   5],
           [144,   3,   2,   2],
           [143,   2,   1,   1]], dtype=uint32


lv with discrepant prim param
---------------------------------

::

    In [8]: lvd
    Out[8]: array([ 22,  24,  25,  29,  42,  75,  77,  81,  85, 130, 143, 145], dtype=uint32)

    nudged (lvIdx/treeidx,num_prim,coincidences,nudges)
     nn[np.where( nn[:,3] > 0 )] 
    [[ 42   3   1   1]
     [ 37   3   1   1]
     [ 24   3   1   1]
     [ 22   3   1   1]
     [ 25   2   1   1]
     [ 26   2   1   1]
     [ 29   3   2   1]
     [ 54   2   1   1]
     [ 68   2   1   1]
     [ 75   3   2   2]
     [ 77   3   2   2]
     [ 81   3   2   1]
     [ 85   3   2   1]
     [130   3   2   2]
     [145   6   5   5]
     [144   3   2   2]
     [143   2   1   1]]

    In [2]: np.unique(nn[np.where( nn[:,3] > 0 )][:,0])
    Out[2]: array([ 22,  24,  25,  *26*,  29, *37*,  42,  *54*,  *68*,  75,  77,  81,  85, 130, 143, *144*, 145], dtype=uint32)

    26, 37, 54, 68, 144       were nudged but not noticed as discrepant ?



32 prims with discrepant parts : 1mm polycone z-nudging ? 
------------------------------------------------------------

::

    epsilon:opticks blyth$ ab-;ab-p
    import os, numpy as np
    from opticks.ana.mesh import Mesh
    from opticks.ana.prim import Dir
    from opticks.sysrap.OpticksCSG import CSG_

    a_dir = "/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0"
    b_dir = "/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0"
    a_idpath = "/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103"
    b_idpath = "/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1"

    a_load = lambda _:np.load(os.path.join(a_dir, _))
    b_load = lambda _:np.load(os.path.join(b_dir, _))

    pa = a_load("primBuffer.npy")
    pb = b_load("primBuffer.npy")
    assert np.all( pa == pb )

    xb = b_load("idxBuffer.npy")
    assert len(pa) == len(xb)

    ma = Mesh.make(a_idpath)


    da = Dir(a_dir)
    db = Dir(b_dir)
    cut = 0.1
    where_discrepant = da.where_discrepant_prims(db, cut) 

    print " num_discrepant %d cut %s " % ( len(where_discrepant), cut ) 

    for i in where_discrepant:

        primIdx = i 
        _,soIdx,lvIdx,height = xb[i]
        name = ma.idx2name[lvIdx]

        print " %s primIdx:%3d soIdx:%3d lvIdx:%3d height:%d name:%s  %s " % ( "-" * 30, primIdx, soIdx,lvIdx,height, name,   "-" * 60 )
        dap = da.prims[i]
        dbp = db.prims[i]
        print dap.maxdiff(dbp)
        print dap
        print dbp
        print
        print

    rgs: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-p.py
    [2018-07-02 22:07:58,278] p57453 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
    nudged (lvIdx/treeidx,num_prim,coincidences,nudges)
     nn[np.where( nn[:,3] > 0 )] 
    []
     num_discrepant 32 cut 0.1 
     ------------------------------ primIdx:  3 soIdx: 33 lvIdx: 42 height:2 name:oav0xc2ed7c8  ------------------------------------------------------------ 
    1.0

    primIdx 3 partOffset 3 numParts 7 tranOffset 3 planOffset 0  
        Part   1  0            union    20      MineralOil///Acrylic   tz:     0.000      
        Part  12  1         cylinder    20      MineralOil///Acrylic   tz: -7141.500     r:   2000.000 z1: -1968.500 z2:  1969.500   
        Part   1  0            union    20      MineralOil///Acrylic   tz:     0.000      
        Part  12  2         cylinder    20      MineralOil///Acrylic   tz: -9110.000     r:   2040.000 z1:  3937.000 z2:  4000.025   
        Part  15  2             cone    20      MineralOil///Acrylic   tz: -9110.000      

    primIdx 3 partOffset 3 numParts 7 tranOffset 3 planOffset 0  
        Part   1  0            union    20      MineralOil///Acrylic   tz:     0.000      
        Part  12  1         cylinder    20      MineralOil///Acrylic   tz: -7141.500     r:   2000.000 z1: -1968.500 z2:  1968.500   
        Part   1  0            union    20      MineralOil///Acrylic   tz:     0.000      
        Part  12  2         cylinder    20      MineralOil///Acrylic   tz: -9110.000     r:   2040.000 z1:  3937.000 z2:  4000.025   
        Part  15  2             cone    20      MineralOil///Acrylic   tz: -9110.000      


     ------------------------------ primIdx:  5 soIdx: 35 lvIdx: 24 height:2 name:iav0xc346f90  ------------------------------------------------------------ 
    1.0

    primIdx 5 partOffset 17 numParts 7 tranOffset 7 planOffset 0  
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -7107.500     r:   1560.000 z1: -1542.500 z2:  1543.500   
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  12  2         cylinder    22 LiquidScintillator///Acrylic   tz: -8650.000     r:   1565.000 z1:  3085.000 z2:  3100.000   
        Part  15  2             cone    22 LiquidScintillator///Acrylic   tz: -8650.000      

    primIdx 5 partOffset 17 numParts 7 tranOffset 7 planOffset 0  
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -7107.500     r:   1560.000 z1: -1542.500 z2:  1542.500   
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  12  2         cylinder    22 LiquidScintillator///Acrylic   tz: -8650.000     r:   1565.000 z1:  3085.000 z2:  3100.000   
        Part  15  2             cone    22 LiquidScintillator///Acrylic   tz: -8650.000      


     ------------------------------ primIdx:  6 soIdx: 36 lvIdx: 22 height:2 name:gds0xc28d3f0  ------------------------------------------------------------ 
    1.0

    primIdx 6 partOffset 24 numParts 7 tranOffset 9 planOffset 0  
        Part   1  0            union    23       Acrylic///GdDopedLS   tz:     0.000      
        Part  12  1         cylinder    23       Acrylic///GdDopedLS   tz: -7100.000     r:   1550.000 z1: -1535.000 z2:  1535.000   
        Part   1  0            union    23       Acrylic///GdDopedLS   tz:     0.000      
        Part  15  2             cone    23       Acrylic///GdDopedLS   tz: -8635.000      
        Part  12  2         cylinder    23       Acrylic///GdDopedLS   tz: -8635.000     r:     75.000 z1:  3145.729 z2:  3159.440   

    primIdx 6 partOffset 24 numParts 7 tranOffset 9 planOffset 0  
        Part   1  0            union    23       Acrylic///GdDopedLS   tz:     0.000      
        Part  12  1         cylinder    23       Acrylic///GdDopedLS   tz: -7100.000     r:   1550.000 z1: -1535.000 z2:  1535.000   
        Part   1  0            union    23       Acrylic///GdDopedLS   tz:     0.000      
        Part  15  2             cone    23       Acrylic///GdDopedLS   tz: -8635.000      
        Part  12  2         cylinder    23       Acrylic///GdDopedLS   tz: -8635.000     r:     75.000 z1:  3145.729 z2:  3159.440   


     ------------------------------ primIdx:  8 soIdx: 38 lvIdx: 25 height:1 name:IavTopHub0xc405968  ------------------------------------------------------------ 
    1.0

    primIdx 8 partOffset 38 numParts 3 tranOffset 14 planOffset 0  
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -5475.561     r:    100.000 z1:     0.000 z2:    86.560   
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -5475.561     r:    150.000 z1:    85.560 z2:   110.560   

    primIdx 8 partOffset 38 numParts 3 tranOffset 14 planOffset 0  
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -5475.561     r:    100.000 z1:     0.000 z2:    85.560   
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -5475.561     r:    150.000 z1:    85.560 z2:   110.560   


     ------------------------------ primIdx: 12 soIdx: 42 lvIdx: 29 height:2 name:OcrGdsPrt0xc352518  ------------------------------------------------------------ 
    1.0

    primIdx 12 partOffset 48 numParts 7 tranOffset 18 planOffset 0  
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  15  2             cone    22 LiquidScintillator///Acrylic   tz: -5512.780      
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -5550.000     r:    100.000 z1:     0.000 z2:   161.000   
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -5550.000     r:    150.000 z1:   160.000 z2:   185.000   

    primIdx 12 partOffset 48 numParts 7 tranOffset 18 planOffset 0  
        Part   3  0       difference    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part   1  0            union    22 LiquidScintillator///Acrylic   tz:     0.000      
        Part  15  2             cone    22 LiquidScintillator///Acrylic   tz: -5512.780      
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -5550.000     r:    100.000 z1:     0.000 z2:   160.000   
        Part  12  1         cylinder    22 LiquidScintillator///Acrylic   tz: -5550.000     r:    150.000 z1:   160.000 z2:   185.000   





Finding some big prims
---------------------------

::

    In [34]: np.where( pa[:,1] > 15 )
    Out[34]: 
    (array([ 280,  281,  282,  283,  284,  285,  286,  287,  288,  289,  290,  291,  292,  293,  294,  295,  296,  297,  298,  299,  300,  301,  302,  303,  304,  305,  306,  307,  308,  309,  310,  311,
             314,  427,  438,  453,  495,  515,  526,  541,  597,  608,  623,  980,  981,  982,  983,  984,  985,  986,  987,  988,  989,  990,  991,  992,  993,  994,  995,  996,  997,  998,  999, 1000,
            1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1014, 1127, 1138, 1153, 1195, 1215, 1226, 1241, 1297, 1308, 1323]),)

    In [35]: app[280]
    Out[35]: primIdx 280 prim array([840,  31, 326,   0], dtype=int32) partOffset 840 numParts 31 tranOffset 326 planOffset 0  

    In [36]: print app[280]

    primIdx 280 prim array([840,  31, 326,   0], dtype=int32) partOffset 840 numParts 31 tranOffset 326 planOffset 0  
        Part   2  0     intersection    36 MineralOil/RSOilSurface//Acrylic   tz:     0.000      
        Part   2  0     intersection    36 MineralOil/RSOilSurface//Acrylic   tz:     0.000      
        Part !12  7         cylinder    36 MineralOil/RSOilSurface//Acrylic   tz: -8842.500     r:    106.600 z1:  -250.000 z2:   250.000   
        Part   2  0     intersection    36 MineralOil/RSOilSurface//Acrylic   tz:     0.000      
        Part   2  0     intersection    36 MineralOil/RSOilSurface//Acrylic   tz:     0.000      
        Part   2  0     intersection    36 MineralOil/RSOilSurface//Acrylic   tz:     0.000      
        Part   2  0     intersection    36 MineralOil/RSOilSurface//Acrylic   tz:     0.000      
        Part   2  0     intersection    36 MineralOil/RSOilSurface//Acrylic   tz:     0.000      
        Part   2  0     intersection    36 MineralOil/RSOilSurface//Acrylic   tz:     0.000      
        Part  12  1         cylinder    36 MineralOil/RSOilSurface//Acrylic   tz: -8592.500     r:   2262.150 z1:  -498.500 z2:   498.500   
        Part !12  1         cylinder    36 MineralOil/RSOilSurface//Acrylic   tz: -8592.500     r:   2259.150 z1:  -503.485 z2:   503.485   
        Part  19  1  convexpolyhedron    36 MineralOil/RSOilSurface//Acrylic   tz: -8592.500      
        Part !12  2         cylinder    36 MineralOil/RSOilSurface//Acrylic   tz: -8342.500     r:    106.600 z1:  -250.000 z2:   250.000   
        Part !12  3         cylinder    36 MineralOil/RSOilSurface//Acrylic   tz: -8342.500     r:    106.600 z1:  -250.000 z2:   250.000   
        Part !12  4         cylinder    36 MineralOil/RSOilSurface//Acrylic   tz: -8342.500     r:    106.600 z1:  -250.000 z2:   250.000   
        Part !12  5         cylinder    36 MineralOil/RSOilSurface//Acrylic   tz: -8842.500     r:    106.600 z1:  -250.000 z2:   250.000   
        Part !12  6         cylinder    36 MineralOil/RSOilSurface//Acrylic   tz: -8842.500     r:    106.600 z1:  -250.000 z2:   250.000   

    In [37]: print bpp[280]

    primIdx 280 prim array([840,  31, 326,   0], dtype=int32) partOffset 840 numParts 31 tranOffset 326 planOffset 0  
        Part   2  0     intersection    20      MineralOil///Acrylic   tz:     0.000      
        Part   2  0     intersection    20      MineralOil///Acrylic   tz:     0.000      
        Part !12  7         cylinder    20      MineralOil///Acrylic   tz: -8842.500     r:    106.600 z1:  -250.000 z2:   250.000   
        Part   2  0     intersection    20      MineralOil///Acrylic   tz:     0.000      
        Part   2  0     intersection    20      MineralOil///Acrylic   tz:     0.000      
        Part   2  0     intersection    20      MineralOil///Acrylic   tz:     0.000      
        Part   2  0     intersection    20      MineralOil///Acrylic   tz:     0.000      
        Part   2  0     intersection    20      MineralOil///Acrylic   tz:     0.000      
        Part   2  0     intersection    20      MineralOil///Acrylic   tz:     0.000      
        Part  12  1         cylinder    20      MineralOil///Acrylic   tz: -8592.500     r:   2262.150 z1:  -498.500 z2:   498.500   
        Part !12  1         cylinder    20      MineralOil///Acrylic   tz: -8592.500     r:   2259.150 z1:  -503.485 z2:   503.485   
        Part  19  1  convexpolyhedron    20      MineralOil///Acrylic   tz: -8592.500      
        Part !12  2         cylinder    20      MineralOil///Acrylic   tz: -8342.500     r:    106.600 z1:  -250.000 z2:   250.000   
        Part !12  3         cylinder    20      MineralOil///Acrylic   tz: -8342.500     r:    106.600 z1:  -250.000 z2:   250.000   
        Part !12  4         cylinder    20      MineralOil///Acrylic   tz: -8342.500     r:    106.600 z1:  -250.000 z2:   250.000   
        Part !12  5         cylinder    20      MineralOil///Acrylic   tz: -8842.500     r:    106.600 z1:  -250.000 z2:   250.000   
        Part !12  6         cylinder    20      MineralOil///Acrylic   tz: -8842.500     r:    106.600 z1:  -250.000 z2:   250.000   





added ab-p for prim differencing
-------------------------------------

Hmm the parts are mostly CSG constituents of compound shapes, 
to debug the 1mm shifts need a way to go from the constituent
to its root node and thence to find which primIdx and get 
identity info lvIdx etc..

primBuffer has partOffsets and partNumbers, so should 
be able to go from a partIdx to a primIdx  

Alternatively iterate over the primBuffer and
then compare the part range that it references.
Then can see max part difference for each primitive. 

::

    In [8]: np.all( pa == pb )
    Out[8]: True


::

    In [2]: b.shape
    Out[2]: (11984, 4, 4)

    In [5]: pb[:,1].sum()
    Out[5]: 11984


    In [12]: pa[:10]
    Out[12]: 
    array([[ 0,  1,  0,  0],
           [ 1,  1,  1,  0],
           [ 2,  1,  2,  0],
           [ 3,  7,  3,  0],
           [10,  7,  5,  0],
           [17,  7,  7,  0],
           [24,  7,  9,  0],
           [31,  7, 11,  0],
           [38,  3, 14,  0],
           [41,  3, 15,  0]], dtype=int32)

    In [13]: pa[:10,1]
    Out[13]: array([1, 1, 1, 7, 7, 7, 7, 7, 3, 3], dtype=int32)

    In [14]: np.cumsum( pa[:10,1] )
    Out[14]: array([ 1,  2,  3, 10, 17, 24, 31, 38, 41, 44])

    In [15]: np.cumsum( pa[:10,1] ).shape
    Out[15]: (10,)

    In [16]: pa[:10,1].shape
    Out[16]: (10,)







::

    epsilon:opticks blyth$ ab-;ab-i
    import numpy as np

    from opticks.ana.mesh import Mesh
    from opticks.sysrap.OpticksCSG import CSG_

    a = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/partBuffer.npy")
    ta = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/tranBuffer.npy")
    pa = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/primBuffer.npy")

    b = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/partBuffer.npy")
    tb = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/tranBuffer.npy")
    pb = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/primBuffer.npy")
    xb = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/idxBuffer.npy")

    ma = Mesh.make("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103")
    mb = Mesh.make("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1")


    def cfprim(pa,pb,xb,ma):
        """
        primBuffer will be matched when all prim trees have same heights
        and the usage of tranforms and planes within each prim are the same
        """
        assert np.all(pa == pb)

        w = np.where( pa[:,1] != pb[:,1] )[0]

        lv = np.unique(xb[w][:,2])

        print "\n".join(map(lambda _:ma.idx2name[_], lv ))
    pass
    #cfprim(pa,pb,xb,ma)


    def cfpart(a, b):
        """
        comparing part buffers (aka csg nodes) 

        1. typecode CSG_UNION/CSG_SPHERE/.. of each part (aka node)  
        2. global transform index 
        3. part parameter values 

        """
        assert len(a) == len(b)
        assert a.shape == b.shape
        count = 0 
        cut = 0.0005
        for i in range(len(a)):
            tca = a[i].view(np.int32)[2][3]
            tcb = b[i].view(np.int32)[2][3]
            assert tca == tcb
            if tca != tcb:
                print " tc mismatch %d %d " % (tca, tcb)
            pass
            tc = tca 
            tcn = CSG_.desc(tc)
     
            gta = a[i].view(np.int32)[3][3]
            gtb = b[i].view(np.int32)[3][3]
            assert gta == gtb
            msg = " gt mismatch " if gta != gtb else ""

            if gta < 0 or gtb < 0: msg += " : gta/gtb -ve " 

            mx = np.max(a[i]-b[i])

            if mx > cut:
                count += 1 
                print " i:%6d count:%6d tc:%3d tcn:%20s gta:%2d gtb:%2d mx:%10s %s  " % ( i, count, tc, tcn, gta, gtb, mx, msg  )
                #print (a[i]-b[i])/mx
            pass
        pass
        print " num_nodes %5d  num_discrepant : %5d   cut:%s  " % ( len(a), count, cut  ) 
    pass


    # boundaries differ due to lack of surfaces in the test, so scrub that  
    # as it hides other problems
    b.view(np.int32)[:,1,2] = a.view(np.int32)[:,1,2]

    cfpart(a,b)


    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/i.py
    [2018-07-02 17:04:07,269] p35495 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
    [2018-07-02 17:04:07,270] p35495 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
     i:     4 count:     1 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:    18 count:     2 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:    29 count:     3 tc: 15 tcn:                cone gta: 2 gtb: 2 mx:       1.0   
     i:    39 count:     4 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:    51 count:     5 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  2452 count:     6 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  2462 count:     7 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  2471 count:     8 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  2482 count:     9 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  2497 count:    10 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  2508 count:    11 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  2717 count:    12 tc: 12 tcn:            cylinder gta: 3 gtb: 3 mx: 1.0000001   
     i:  2794 count:    13 tc: 12 tcn:            cylinder gta: 5 gtb: 5 mx:       1.0   
     i:  2796 count:    14 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  2874 count:    15 tc: 12 tcn:            cylinder gta: 2 gtb: 2 mx:       1.0   
     i:  3081 count:    16 tc: 12 tcn:            cylinder gta: 3 gtb: 3 mx: 1.0000001   
     i:  3359 count:    17 tc: 12 tcn:            cylinder gta: 3 gtb: 3 mx: 1.0000001   
     i:  3496 count:    18 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  3510 count:    19 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  3521 count:    20 tc: 15 tcn:                cone gta: 2 gtb: 2 mx:       1.0   
     i:  3531 count:    21 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  3543 count:    22 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  5944 count:    23 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  5954 count:    24 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  5963 count:    25 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  5974 count:    26 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  5989 count:    27 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  6000 count:    28 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  6209 count:    29 tc: 12 tcn:            cylinder gta: 3 gtb: 3 mx: 1.0000001   
     i:  6286 count:    30 tc: 12 tcn:            cylinder gta: 5 gtb: 5 mx:       1.0   
     i:  6288 count:    31 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:  6366 count:    32 tc: 12 tcn:            cylinder gta: 2 gtb: 2 mx:       1.0   
     i:  6573 count:    33 tc: 12 tcn:            cylinder gta: 3 gtb: 3 mx: 1.0000001   
     i:  6851 count:    34 tc: 12 tcn:            cylinder gta: 3 gtb: 3 mx: 1.0000001   
     num_nodes 11984  num_discrepant :    34   cut:0.0005  

    In [1]: 


Those are cylinder/cone z1/z2 1mm uncoincidence nudges ? Where are they applied ?::

    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/i.py
    [2018-07-02 17:08:42,151] p35608 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
    [2018-07-02 17:08:42,151] p35608 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
     i:     4 count:     1 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
    [[    0.      0.      0.   2000.      0.      0.      0.   2000. ]
     [-1968.5  1969.5     0.      0.  -1968.5  1968.5     0.      0. ]
     [    0.      0.      0.      0.      0.      0.      0.      0. ]
     [    0.      0.      0.      0.      0.      0.      0.      0. ]]
     i:    18 count:     2 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
    [[    0.      0.      0.   1560.      0.      0.      0.   1560. ]
     [-1542.5  1543.5     0.      0.  -1542.5  1542.5     0.      0. ]
     [    0.      0.      0.      0.      0.      0.      0.      0. ]
     [    0.      0.      0.      0.      0.      0.      0.      0. ]]
     i:    29 count:     3 tc: 15 tcn:                cone gta: 2 gtb: 2 mx:       1.0   
    [[1520.     3069.       75.     3146.7292 1520.     3070.       75.     3145.7292]
     [   0.        0.        0.        0.        0.        0.        0.        0.    ]
     [   0.        0.        0.        0.        0.        0.        0.        0.    ]
     [   0.        0.        0.        0.        0.        0.        0.        0.    ]]
     i:    39 count:     4 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
    [[  0.       0.       0.     100.       0.       0.       0.     100.    ]
     [  0.      86.5604   0.       0.       0.      85.5604   0.       0.    ]
     [  0.       0.       0.       0.       0.       0.       0.       0.    ]
     [  0.       0.       0.       0.       0.       0.       0.       0.    ]]



Matched -ve gta/gtb are the complemented with their sign bits set::

    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/i.py
    [2018-07-02 17:11:43,757] p35665 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
    [2018-07-02 17:11:43,758] p35665 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
     i:     4 count:     1 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:    18 count:     2 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:    29 count:     3 tc: 15 tcn:                cone gta: 2 gtb: 2 mx:       1.0   
     i:    39 count:     4 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:    51 count:     5 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     i:   842 count:     6 tc: 12 tcn:            cylinder gta:-2147483641 gtb:-2147483641 mx:       0.0  : gta/gtb -ve   
     i:   856 count:     7 tc: 12 tcn:            cylinder gta:-2147483647 gtb:-2147483647 mx:       0.0  : gta/gtb -ve   
     i:   858 count:     8 tc: 12 tcn:            cylinder gta:-2147483646 gtb:-2147483646 mx:       0.0  : gta/gtb -ve   
     i:   859 count:     9 tc: 12 tcn:            cylinder gta:-2147483645 gtb:-2147483645 mx:       0.0  : gta/gtb -ve   
     i:   860 count:    10 tc: 12 tcn:            cylinder gta:-2147483644 gtb:-2147483644 mx:       0.0  : gta/gtb -ve   
     i:   861 count:    11 tc: 12 tcn:            cylinder gta:-2147483643 gtb:-2147483643 mx:       0.0  : gta/gtb -ve   
     i:   862 count:    12 tc: 12 tcn:            cylinder gta:-2147483642 gtb:-2147483642 mx:       0.0  : gta/gtb -ve   
     i:   873 count:    13 tc: 12 tcn:            cylinder gta:-2147483641 gtb:-2147483641 mx:       0.0  : gta/gtb -ve   
     i:   887 count:    14 tc: 12 tcn:            cylinder gta:-2147483647 gtb:-2147483647 mx:       0.0  : gta/gtb -ve   
     i:   889 count:    15 tc: 12 tcn:            cylinder gta:-2147483646 gtb:-2147483646 mx:       0.0  : gta/gtb -ve   



Boundaries are very different due to lack of the surfaces, get rid of that difference for now, until have 
reconstructed surfaces::

    In [3]: b.view(np.int32)[:,1,2] 
    Out[3]: array([17, 18, 19, ..., 84, 84, 84], dtype=int32)

    In [4]: a.view(np.int32)[:,1,2] 
    Out[4]: array([ 17,  18,  19, ..., 120, 120, 120], dtype=int32)

    In [5]: b.view(np.int32)[:,1,2] = a.view(np.int32)[:,1,2]



