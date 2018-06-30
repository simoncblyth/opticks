OKX4Test_mm0_has_85k_parts_vs_12k
===================================

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

