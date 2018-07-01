OKX4Test_mm0_has_85k_parts_vs_12k
===================================


what are tree height actual limits ?
-------------------------------------

::

     542 #define USE_TWIDDLE_POSTORDER 1
     543 
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



Hmm maxcsgheight is 3 : not 4
--------------------------------

Argh, maybe trivial cut difference (misread the code, the default maxcsgheight of 4)
is overrden down to 3::


    509 def gdml2gltf_main( args ):
    510     """
    511     main used by bin/gdml2gltf.py 
    512     """
    513     # envvars are set within opticks_main
    514     gdmlpath = os.environ['OPTICKS_GDMLPATH']
    515     gltfpath = os.environ['OPTICKS_GLTFPATH']
    516 
    517     assert gdmlpath.replace('.gdml','.gltf') == gltfpath
    518     assert gltfpath.replace('.gltf','.gdml') == gdmlpath
    519 
    520     log.info("start GDML parse")
    521     gdml = GDML.parse(gdmlpath)
    522 
    523     log.info("start treeify")
    524     tree = Tree(gdml.world)
    525 
    526     log.info("start apply_selection")
    527     tree.apply_selection(args.query)   # sets node.selected "volume mask" 
    528 
    529     log.info("start Sc.ctor")
    530     sc = Sc(maxcsgheight=3)
    531 
    532     sc.extras["verbosity"] = 1
    533     sc.extras["targetnode"] = 0   # args.query.query_range[0]   # hmm get rid of this ?
    534 
    535     log.info("start Sc.add_tree_gdml")
    536 
    537     tg = sc.add_tree_gdml( tree.root, maxdepth=0)
    538 
    539     log.info("start Sc.add_tree_gdml DONE")
    540 
    541     #path = args.gltfpath
    542     gltf = sc.save(gltfpath)
    543 
    544     sc.gdml = gdml
    545     sc.tree = tree
    546 
    547     return sc




Huh most of the listed are not balanced as not overheight 
--------------------------------------------------------------

* perhaps the peculiarity is with the old conversion, try to proceed asis 

::

    2018-07-01 14:11:39.987 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx  38 lvIdx  25 height0 2 height1 2 ### LISTED
    2018-07-01 14:11:39.988 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx  39 lvIdx  26 height0 2 height1 2 ### LISTED
    2018-07-01 14:11:39.992 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx  42 lvIdx  29 height0 3 height1 3 ### LISTED
    2018-07-01 14:11:41.067 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx  73 lvIdx  60 height0 8 height1 4 ### LISTED
    2018-07-01 14:11:41.084 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx  76 lvIdx  65 height0 4 height1 4 ### LISTED
    2018-07-01 14:11:41.260 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx  79 lvIdx  68 height0 2 height1 2 ### LISTED
    2018-07-01 14:11:41.634 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx  86 lvIdx  75 height0 3 height1 3 ### LISTED
    2018-07-01 14:11:41.638 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx  88 lvIdx  77 height0 3 height1 3 ### LISTED
    2018-07-01 14:11:41.644 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx  92 lvIdx  81 height0 3 height1 3 ### LISTED
    2018-07-01 14:11:41.654 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx  96 lvIdx  85 height0 3 height1 3 ### LISTED
    2018-07-01 14:11:42.665 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx 126 lvIdx 131 height0 3 height1 3 ### LISTED
    2018-07-01 14:11:42.726 INFO  [1847231] [*NTreeProcess<nnode>::Process@35]  soIdx 151 lvIdx 140 height0 4 height1 4 ### LISTED



mm0 has 12 discrepant trees : with different tree height 
------------------------------------------------------------

::

    w = np.where( pa[:,1] != pb[:,1] )[0]

    lv = np.unique(xb[w][:,2])

    print "\n".join(map(lambda _:mb.idx2name[_], lv ))


    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/i.py
    [2018-07-01 12:17:46,914] p44730 {opticks/ana/mesh.py:32} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 

    near_pool_dead_box0xbf8a280
    near_pool_liner_box0xc2dcc28
    PLACEHOLDER_near_pool_iws_box0xc288ce8
    headon-pmt-assy0xbf55198
    headon-pmt-base0xc25cb40
    TopRefCutHols0xbf9bd50
    SsTBotHub0xc26d1d0
    SstTopRadiusRib0xc271720
    SstInnVerRibBase0xbf30b50
    OavTopRib0xc0d5e10
    Co60AlSource0xc3cebf8
    SidePinSSTube0xc3d1f08

    In [1]: lv
    Out[1]: array([ 25,  26,  29,  60,  65,  68,  75,  77,  81,  85, 131, 140], dtype=uint32)



use mm5 to check the idx
---------------------------

::

    In [1]: idx = np.load("idxBuffer.npy")

    In [2]: idx
    Out[2]: 
    array([[ 0, 54, 47,  3],
           [ 0, 55, 46,  3],
           [ 0, 56, 43,  2],
           [ 0, 57, 44,  1],
           [ 0, 58, 45,  0]], dtype=uint32)






added idxBuffer to NCSG/GParts so can see soIdx/lvIdx with problem parts
-------------------------------------------------------------------------------



::

    In [19]: lv = np.unique(idx[w][:,2])



::

    In [3]: idx = np.load("idxBuffer.npy")

    In [4]: idx[w]
    Out[4]: 
    array([[  0,  38,  25,   2],
           [  0,  39,  26,   2],
           [  0,  42,  29,   3],
           [  0,  39,  26,   2],
           [  0,  73,  60,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  79,  68,   2],
           [  0,  86,  75,   3],
           [  0,  88,  77,   3],
           [  0,  92,  81,   3],
           [  0,  88,  77,   3],
           [  0,  96,  85,   3],
           [  0,  88,  77,   3],
           [  0, 126, 131,   3],
           [  0, 151, 140,   4],
           [  0, 126, 131,   3],
           [  0, 126, 131,   3],

           [  0,  38,  25,   2],
           [  0,  39,  26,   2],
           [  0,  42,  29,   3],
           [  0,  39,  26,   2],
           [  0,  73,  60,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  76,  65,   4],
           [  0,  79,  68,   2],
           [  0,  86,  75,   3],
           [  0,  88,  77,   3],
           [  0,  92,  81,   3],
           [  0,  88,  77,   3],
           [  0,  96,  85,   3],
           [  0,  88,  77,   3],
           [  0, 126, 131,   3],
           [  0, 151, 140,   4],
           [  0, 126, 131,   3],
           [  0, 126, 131,   3]], dtype=uint32)



require access from a part to  soIdx/lvIdx/ndIdx 
------------------------------------------------------

* hmm to debug need access to identity indices soIdx/lvIdx/ndIdx : in a prim-level (ie Volume level) array 
  hmm isnth that already held in the merged mesh ?  not quite what is needed

  * added idxBuf to GParts to provide a slot of 4 uint to go with every NCSG/GPart that gets
    combined 


::

    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GMergedMesh/0
    epsilon:0 blyth$ np.py identity.npy 
    (12230, 4)
    u32
    [[    0     0     0     0]
     [    1     1     1     0]
     [    2     2     2     0]
     ...
     [12227   246    85     0]
     [12228   247    85     0]
     [12229   248    85     0]]
    epsilon:0 blyth$ 




* but the primBuffer is derived from the partBuffer in GParts::makePrimBuffer()
  so to follow that need to repeat identity at part-level 


all bits of AD : notice the repeated pattern, for each AD
------------------------------------------------------------


* 24 shapes, repeated for each AD have different tree size
* checking boundaries of geometry with discrepant tree sizes 

::

    pp = map(str.strip, open("GParts.txt").readlines())

    In [51]: for _ in pb[w][:,0]: print pp[_]   

    LiquidScintillator///Acrylic
    LiquidScintillator///Acrylic
    LiquidScintillator///Acrylic
    LiquidScintillator///Acrylic
    Air///ESR
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///Acrylic
    MineralOil///Acrylic
    MineralOil///Acrylic
    MineralOil///Acrylic
    MineralOil///Acrylic
    MineralOil///Acrylic
    Vacuum///Acrylic
    NitrogenGas///Acrylic
    Vacuum///Acrylic
    Vacuum///Acrylic

    LiquidScintillator///Acrylic
    LiquidScintillator///Acrylic
    LiquidScintillator///Acrylic
    LiquidScintillator///Acrylic
    Air///ESR
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///StainlessSteel
    MineralOil///Acrylic
    MineralOil///Acrylic
    MineralOil///Acrylic
    MineralOil///Acrylic
    MineralOil///Acrylic
    MineralOil///Acrylic
    Vacuum///Acrylic
    NitrogenGas///Acrylic
    Vacuum///Acrylic
    Vacuum///Acrylic


mm0 part count differences all one level up, C++ (b) larger than the python (a)
---------------------------------------------------------------------------------

::

    In [31]: w = np.where( pa[:,1] != pb[:,1] )[0]

    In [36]: w
    Out[36]: 
    array([   8,    9,   12,   13,  317,  327,  328,  329,  330,  331,  332,  333,  334,  351,  392,  394,  398,  400,  404,  406,  454,  493,  542,  624,  708,  709,  712,  713, 1017, 1027, 1028, 1029,
           1030, 1031, 1032, 1033, 1034, 1051, 1092, 1094, 1098, 1100, 1104, 1106, 1154, 1193, 1242, 1324])


    


    In [35]: np.hstack( [pa[w], pb[w]] )
    Out[35]: 
    array([[  38,    3,   14,    0,   38,    7,   14,    0],
           [  41,    3,   15,    0,   45,    7,   15,    0],
           [  48,    7,   18,    0,   56,   15,   18,    0],
           [  55,    3,   20,    0,   71,    7,   20,    0],
           [1923,   15,  579,  160, 1943,   31,  579,  160],
           [1997,   15,  620,  160, 2033,   31,  620,  160],
           [2012,   15,  623,  165, 2064,   31,  623,  165],
           [2027,   15,  626,  170, 2095,   31,  626,  170],
           [2042,   15,  629,  175, 2126,   31,  629,  175],
           [2057,   15,  632,  180, 2157,   31,  632,  180],
           [2072,   15,  635,  185, 2188,   31,  635,  185],
           [2087,   15,  638,  190, 2219,   31,  638,  190],
           [2102,   15,  641,  195, 2250,   31,  641,  195],
           [2229,    3,  692,  248, 2393,    7,  692,  248],
           [2448,    7,  781,  336, 2616,   15,  781,  336],
           [2458,    7,  783,  336, 2634,   15,  783,  336],
           [2468,    7,  787,  336, 2652,   15,  787,  336],
           [2478,    7,  790,  336, 2670,   15,  790,  336],
           [2494,    7,  797,  336, 2694,   15,  797,  336],
           [2504,    7,  800,  336, 2712,   15,  800,  336],
           [2708,    7,  897,  336, 2924,   15,  897,  336],
           [2811,   15,  954,  336, 3035,   31,  954,  336],
           [3072,    7, 1060,  336, 3312,   15, 1060,  336],
           [3350,    7, 1198,  336, 3598,   15, 1198,  336],
           [3530,    3, 1300,  336, 3786,    7, 1300,  336],
           [3533,    3, 1301,  336, 3793,    7, 1301,  336],
           [3540,    7, 1304,  336, 3804,   15, 1304,  336],
           [3547,    3, 1306,  336, 3819,    7, 1306,  336],
           [5415,   15, 1865,  496, 5691,   31, 1865,  496],
           [5489,   15, 1906,  496, 5781,   31, 1906,  496],
           [5504,   15, 1909,  501, 5812,   31, 1909,  501],
           [5519,   15, 1912,  506, 5843,   31, 1912,  506],
           [5534,   15, 1915,  511, 5874,   31, 1915,  511],
           [5549,   15, 1918,  516, 5905,   31, 1918,  516],
           [5564,   15, 1921,  521, 5936,   31, 1921,  521],
           [5579,   15, 1924,  526, 5967,   31, 1924,  526],
           [5594,   15, 1927,  531, 5998,   31, 1927,  531],
           [5721,    3, 1978,  584, 6141,    7, 1978,  584],
           [5940,    7, 2067,  672, 6364,   15, 2067,  672],
           [5950,    7, 2069,  672, 6382,   15, 2069,  672],
           [5960,    7, 2073,  672, 6400,   15, 2073,  672],
           [5970,    7, 2076,  672, 6418,   15, 2076,  672],
           [5986,    7, 2083,  672, 6442,   15, 2083,  672],
           [5996,    7, 2086,  672, 6460,   15, 2086,  672],
           [6200,    7, 2183,  672, 6672,   15, 2183,  672],
           [6303,   15, 2240,  672, 6783,   31, 2240,  672],
           [6564,    7, 2346,  672, 7060,   15, 2346,  672],
           [6842,    7, 2484,  672, 7346,   15, 2484,  672]], dtype=int32)





mm0 part counts 48/3116 have different part counts
-----------------------------------------------------

ab-i::

    In [27]: np.where( pa[:,1] != pb[:,1] )[0]
    Out[27]: 
    array([   8,    9,   12,   13,  317,  327,  328,  329,  330,  331,  332,  333,  334,  351,  392,  394,  398,  400,  404,  406,  454,  493,  542,  624,  708,  709,  712,  713, 1017, 1027, 1028, 1029,
           1030, 1031, 1032, 1033, 1034, 1051, 1092, 1094, 1098, 1100, 1104, 1106, 1154, 1193, 1242, 1324])

    In [28]: np.where( pa[:,1] != pb[:,1] )[0].shape
    Out[28]: (48,)

    In [29]: pa.shape
    Out[29]: (3116, 4)

    In [30]: pb.shape
    Out[30]: (3116, 4)




mm0 plane and transform offsets match
----------------------------------------

ab-i::

    In [13]: pa[:,2]
    Out[13]: array([   0,    1,    2, ..., 5341, 5342, 5343], dtype=int32)

    In [14]: pb[:,2]
    Out[14]: array([   0,    1,    2, ..., 5341, 5342, 5343], dtype=int32)

    In [15]: np.all( pa[:,2] == pb[:,2] )
    Out[15]: True

    In [16]: np.all( pa[:,3] == pb[:,3] )
    Out[16]: True



With balancing implemented are now in the same ballpark::

    epsilon:issues blyth$ ab-diff
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/GParts.txt and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/GParts.txt differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/partBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/partBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/planBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/planBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/primBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/primBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/tranBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/tranBuffer.npy differ
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0
            ./GParts.txt : 11984 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (11984, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = 5eeee07e08a9a50278a2339dd0b47ac4
    MD5 (partBuffer.npy) = 8d837fba380dfc643968bd23f99d656f
    MD5 (planBuffer.npy) = 94e18d5e55d190c9ed73e04b45ebb404
    MD5 (primBuffer.npy) = e21f1c240c4d5e9450aff3ddc0fb78d6
    MD5 (tranBuffer.npy) = 77359e6d3d628e93cb7cf0a4a3824ab3
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0
            ./GParts.txt : 12496 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (12496, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = b15ee45a4d00018105cc858c6e9dca2a
    MD5 (partBuffer.npy) = 89b03b89698585d2172e58cf139e7aa4
    MD5 (planBuffer.npy) = 43f2892dbf4b8e91231e5d830dee9e03
    MD5 (primBuffer.npy) = 486732059344a6448c955e7d90d14d74
    MD5 (tranBuffer.npy) = 74a6d92ff0d830990e81e10434865714
    epsilon:0 blyth$ 







Large differnce in number of parts from the lack of 
tree balancing implementation in the direct approach.


Need to tranlate some more pythin into C++

::

    292     @classmethod
    293     def translate_lv(cls, lv, maxcsgheight, maxcsgheight2=0 ):
    294         """
    295         NB dont be tempted to convert to node here as CSG is a mesh level thing, not node level
    296 
    297         :param lv:
    298         :param maxcsgheight:  CSG trees greater than this are balanced
    299         :param maxcsgheight2:  required post-balanced height to avoid skipping 
    300 
    301         There are many `solid.as_ncsg` implementations, one for each the supported GDML solids, 
    302         some of them return single primitives others return boolean composites, some
    303         such as the Polycone invokes treebuilder to provide uniontree composites.
    304 
    305         """
    306 
    307         if maxcsgheight2 == 0 and maxcsgheight != 0:
    308             maxcsgheight2 = maxcsgheight + 1
    309         pass
    310 
    311         solid = lv.solid
    312         log.debug("translate_lv START %-15s %s  " % (solid.__class__.__name__, lv.name ))
    313 
    314         rawcsg = solid.as_ncsg()
    315 
    316         if rawcsg is None:
    317             err = "translate_lv solid.as_ncsg failed for solid %r lv %r " % ( solid, lv )
    318             log.fatal(err)
    319             rawcsg = CSG.MakeUndefined(err=err,lv=lv)
    320         pass
    321         rawcsg.analyse()
    322 
    323         log.debug("translate_lv DONE %-15s height %3d csg:%s " % (solid.__class__.__name__, rawcsg.height, rawcsg.name))
    324 
    325         csg = cls.optimize_csg(rawcsg, maxcsgheight, maxcsgheight2 )
    326 
    327         polyconfig = PolyConfig(lv.shortname)
    328         csg.meta.update(polyconfig.meta )
    329         csg.meta.update(lvname=lv.name, soname=lv.solid.name, height=csg.height)
    330 
    331         ### Nope pvname is not appropriate in the CSG, CSG is a mesh level tink not a node/volume level thing 
    332 
    333         return csg

::

    336     @classmethod
    337     def optimize_csg(self, rawcsg, maxcsgheight, maxcsgheight2):
    338         """
    339         :param rawcsg:
    340         :param maxcsgheight:  tree balancing is for height > maxcsgheight
    341         :param maxcsgheight2: error is raised if balanced tree height reamains > maxcsgheight2 
    342         :return csg:  balanced csg tree
    343         """
    344         overheight_ = lambda csg,maxheight:csg.height > maxheight and maxheight != 0
    345 
    346         is_balance_disabled = rawcsg.is_balance_disabled()
    347 
    348         #log.info(" %s %s " % ( is_balance_disabled, rawcsg.name ))
    349 
    350         is_overheight = overheight_(rawcsg, maxcsgheight)
    351         if is_overheight:
    352             if is_balance_disabled:
    353                 log.warning("tree is_overheight but marked balance_disabled leaving raw : %s " % rawcsg.name )
    354                 return rawcsg
    355             else:
    356                 log.debug("proceed to balance")
    357         else:
    358             return rawcsg
    359         pass
    360         log.debug("optimize_csg OVERHEIGHT h:%2d maxcsgheight:%d maxcsgheight2:%d %s " % (rawcsg.height,maxcsgheight, maxcsgheight2, rawcsg.name))
    361 
    362         rawcsg.positivize()
    363 
    364         csg = TreeBuilder.balance(rawcsg)
    365 
    366         log.debug("optimize_csg compressed tree from height %3d to %3d " % (rawcsg.height, csg.height ))
    367 
    368         #assert not overheight_(csg, maxcsgheight2)
    369         if overheight_(csg, maxcsgheight2):
    370             csg.meta.update(err="optimize_csg.overheight csg.height %s maxcsgheight:%s maxcsgheight2:%s " % (csg.height,maxcsgheight,maxcsgheight2) )
    371         pass
    372 
    373         return csg


::

    In [9]: pb[:20]
    Out[9]: 
    array([[ 0,  1,  0,  0],
           [ 1,  1,  1,  0],
           [ 2,  1,  2,  0],
           [ 3,  7,  3,  0],
           [10,  7,  5,  0],
           [17,  7,  7,  0],
           [24,  7,  9,  0],
           [31,  7, 11,  0],
           [38,  7, 14,  0],
           [45,  7, 15,  0],
           [52,  3, 16,  0],
           [55,  1, 17,  0],
           [56, 15, 18,  0],
           [71,  7, 20,  0],
           [78,  7, 21,  0],
           [85,  7, 23,  0],
           [92,  1, 26,  0],
           [93,  1, 27,  0],
           [94,  1, 28,  0],
           [95,  1, 29,  0]], dtype=int32)

    In [10]: pa[:20]
    Out[10]: 
    array([[ 0,  1,  0,  0],
           [ 1,  1,  1,  0],
           [ 2,  1,  2,  0],
           [ 3,  7,  3,  0],
           [10,  7,  5,  0],
           [17,  7,  7,  0],
           [24,  7,  9,  0],
           [31,  7, 11,  0],
           [38,  3, 14,  0],
           [41,  3, 15,  0],
           [44,  3, 16,  0],
           [47,  1, 17,  0],
           [48,  7, 18,  0],
           [55,  3, 20,  0],
           [58,  7, 21,  0],
           [65,  7, 23,  0],
           [72,  1, 26,  0],
           [73,  1, 27,  0],
           [74,  1, 28,  0],
           [75,  1, 29,  0]], dtype=int32)


::

    epsilon:GParts blyth$ AB_TAIL="0" ab-diff
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/GParts.txt and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/GParts.txt differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/partBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/partBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/planBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/planBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/primBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/primBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/tranBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/tranBuffer.npy differ
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0
            ./GParts.txt : 11984 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (11984, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = 5eeee07e08a9a50278a2339dd0b47ac4
    MD5 (partBuffer.npy) = 8d837fba380dfc643968bd23f99d656f
    MD5 (planBuffer.npy) = 94e18d5e55d190c9ed73e04b45ebb404
    MD5 (primBuffer.npy) = e21f1c240c4d5e9450aff3ddc0fb78d6
    MD5 (tranBuffer.npy) = 77359e6d3d628e93cb7cf0a4a3824ab3
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0
            ./GParts.txt : 85264 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (85264, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = 6f533aade1075bb4419f716f575ee114
    MD5 (partBuffer.npy) = 95d75b7805b1aca5754de4db4514c3a3
    MD5 (planBuffer.npy) = 43f2892dbf4b8e91231e5d830dee9e03
    MD5 (primBuffer.npy) = bb75be942f2a3efbf60bfc793ff58cbe
    MD5 (tranBuffer.npy) = 74a6d92ff0d830990e81e10434865714
    epsilon:0 blyth$ 
    epsilon:0 blyth$ 

