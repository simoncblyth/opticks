Impinging Volumes
=====================


Plan
-------


* start by testing each node bbox against its parent bbox 

  * within solid uncoincence done in NCSG::postimport, analogous
    place for volume overlap testing would be NScene/GScene ? 

  * start with NScene::postimport

  * testing with: tgltf-t 
  

tgltf-t : Look at gds example
----------------------------------

::

    tgltf-;tgltf-t  ## with OPTICKS_QUERY selection to pick two volumes only, and manual dumping


Comparing gds and parent nd volumes in NScene::postimportmesh find that they have coincident bbox in Z.

* this is highly likely to explain the tachyon behaviour



NScene::postimportmesh
-------------------------

Top of the z-bbox is coincident at -5475.5::

    2017-06-27 15:51:06.834 INFO  [1455881] [NScene::postimportmesh@415] NScene::postimportmesh numNd 12230 dbgnode 3159
    2017-06-27 15:51:06.834 INFO  [1455881] [NScene::dumpNd@702] NScene::dumpNd nidx 3159 node exists  verbosity 1

    nd idx/repeatIdx/mesh/nch/depth/nprog  [3159:  0: 36:  0:14:   0] bnd:Acrylic///GdDopedLS
       nd.tr.t  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   7.500   1.000 

      nd.gtr.t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7100.000   1.000 


     mesh_id 36 meshmeta NScene::meshmeta mesh_id  36 lvidx  22 height  2 soname                        gds0xc28d3f0 lvname      /dd/Geometry/AD/lvGDS0xbf6cbb8
     mesh_idx 36 pmesh_idx 35 root [ 0:un] proot [ 0:un]
        node_t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7100.000   1.000 

       pnode_t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7107.500   1.000 
     csg_bb   mi  (-1550.00 -1550.00 -1535.00)  mx  (1550.00 1550.00 1624.44) 
     pcsg_bb  mi  (-1565.00 -1565.00 -1542.50)  mx  (1565.00 1565.00 1631.94) 
     csg_tbb   mi  (-20222.79 -801842.75 -8635.00)  mx  (-15936.12 -797556.12 -5475.56) 
     pcsg_tbb  mi  (-20243.53 -801863.50 -8650.00)  mx  (-15915.38 -797535.38 -5475.56) 
    Assertion failed: (0 && "NScene::postimportmesh HARIKARI"), function postimportmesh, file /Users/blyth/opticks/opticksnpy/NScene.cpp, line 478.
    Process 89361 stopped





Checking the solids individually
-----------------------------------


::

   opticks-tbool 24    # cylinder with conical top hat, with a bit of lip
   opticks-tbool 22    # similar but with hub cap at middle

   opticks-tbool-vi 24
   opticks-tbool-vi 22



        3158 (24)
          |
        3159 (22)  

::

     62 tbool24--(){ cat << EOP
     63 
     64 import logging
     65 log = logging.getLogger(__name__)
     66 from opticks.ana.base import opticks_main
     67 from opticks.analytic.csg import CSG  
     68 args = opticks_main(csgpath="$TMP/tbool/24")
     69 
     70 CSG.boundary = args.testobject
     71 CSG.kwa = dict(verbosity="0", poly="IM", resolution="20")
     72 
     76 a = CSG("cylinder", param = [0.000,0.000,0.000,1560.000],param1 = [-1542.500,1542.500,0.000,0.000])
                                                         r                   z1       z2


     77 b = CSG("cylinder", param = [0.000,0.000,0.000,1565.000],param1 = [3085.000,3100.000,0.000,0.000])   # (5 mm lip in radius, of 15mm height)
                                                         r                   z1      z2
     In [1]: 1542.5*2                                                     1542.500  1557.5   
     Out[1]: 3085.0
           
     78 c = CSG("cone", param = [1520.393,3100.000,100.000,3174.440],param1 = [0.000,0.000,0.000,0.000])
                                     r1    z1       r2      z2        cone starts from 43 mm smaller radius                                 

     79 bc = CSG("union", left=b, right=c)
     80 bc.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-1542.500,1.000]]
     81 
     82 abc = CSG("union", left=a, right=bc)
     86 
     87 
     88 obj = abc

::

     62 tbool22--(){ cat << EOP
     63 
     64 import logging
     65 log = logging.getLogger(__name__)
     66 from opticks.ana.base import opticks_main
     67 from opticks.analytic.csg import CSG  
     68 args = opticks_main(csgpath="$TMP/tbool/22")
     69 
     70 CSG.boundary = args.testobject
     71 CSG.kwa = dict(verbosity="0", poly="IM", resolution="20")
     72 
     75 
     76 a = CSG("cylinder", param = [0.000,0.000,0.000,1550.000],param1 = [-1535.000,1535.000,0.000,0.000])
                                                         r                   z1       z2            
                                             # 10 mm smaller radius       smaller             


     77 b = CSG("cone", param = [1520.000,3070.000,75.000,3145.729],param1 = [0.000,0.000,0.000,0.000])
                                   r1 z1           r2      z2
     78 c = CSG("cylinder", param = [0.000,0.000,0.000,75.000],param1 = [3145.729,3159.440,0.000,0.000])   # hub cap, 
                                                        r                z1       z2
     79 bc = CSG("union", left=b, right=c)
     80 bc.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-1535.000,1.000]]
     81 
     82 abc = CSG("union", left=a, right=bc)
     83 
     87 
     88 obj = abc




::


    tgltf-;tgltf-t  ## with OPTICKS_QUERY selection to pick two volumes only, and manual dumping



    2017-06-27 14:32:42.057 INFO  [1429523] [NScene::postimport@384] NScene::postimport numNd 12230
    2017-06-27 14:32:42.057 INFO  [1429523] [NScene::dumpNd@613] NScene::dumpNd nidx 3158 node exists  verbosity 1

    nd idx/repeatIdx/mesh/nch/depth/nprog  [3158:  0: 35:  2:13:   0] bnd:LiquidScintillator///Acrylic   
       nd.tr.t  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   2.500   1.000 

      nd.gtr.t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7107.500   1.000 


     mesh_id 35 meshmeta NScene::meshmeta mesh_id  35 lvidx  24 height  2 soname                        iav0xc346f90 lvname      /dd/Geometry/AD/lvIAV0xc404ee8


    2017-06-27 14:32:42.057 INFO  [1429523] [NScene::dumpNd@613] NScene::dumpNd nidx 3159 node exists  verbosity 1

    nd idx/repeatIdx/mesh/nch/depth/nprog  [3159:  0: 36:  0:14:   0] bnd:Acrylic///GdDopedLS   
       nd.tr.t  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   7.500   1.000 

      nd.gtr.t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7100.000   1.000 


     mesh_id 36 meshmeta NScene::meshmeta mesh_id  36 lvidx  22 height  2 soname                        gds0xc28d3f0 lvname      /dd/Geometry/AD/lvGDS0xbf6cbb8



     Hmm : is there 2.5mm of z translation missing in the parent (iav) gtransform ?

             -7101.5
             -7100.0

 







