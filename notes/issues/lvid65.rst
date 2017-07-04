lvidx65 : SstBotCirRibBase0xc26e2d0 /dd/Geometry/AdDetails/lvSstBotCirRibBase0xc26e220
==========================================================================================


* CAUSE FOUND : need to implement tube deltaphi

* :doc:`lvid65`  


::

     DBGMESH=SstBotCirRibBase0x NSceneMeshTest 

     op --dlv65             # just one instance of lvid 65 inside container : gives just small portion of cylinder
     op --dlv65 --gltf 1    # hmm .. instancing kicks in and duplicates it...

     op --dlv65 --gltf 1 --gltfconfig disable_instancing=1    #  gives entire cylinder with cut ?




::

        Difference of two big cylinders with two long "spoke" boxes subtracted, ..
        but spokes are offset and one is rotated.

        ~/opticks_refs/lvid65_cycybobo_to_union.png
             changing difference to union, its clear that only one of boxes intersects
             yielding the cut : suspicious 
              


                     0:di

                 1:di      2:bo

              3:di   4:bo

           7:cy 8:cy


::

     4214     <volume name="/dd/Geometry/AdDetails/lvSstBotCirRibBase0xc26e220">
     4215       <materialref ref="/dd/Materials/StainlessSteel0xc2adc00"/>
     4216       <solidref ref="SstBotCirRibBase0xc26e2d0"/>
     4217     </volume>

     1028     <subtraction name="SstBotCirRibBase0xc26e2d0">
     1029       <first ref="SstBotCirRibPri-ChildForSstBotCirRibBase0xc26e0a0"/>
     1030       <second ref="SstBotRibBase10xbfa4950"/>
     1031       <position name="SstBotCirRibBase0xc26e2d0_pos" unit="mm" x="887.419010389117" y="887.419010389117" z="0"/>
     1032       <rotation name="SstBotCirRibBase0xc26e2d0_rot" unit="deg" x="0" y="0" z="45"/>
     1033     </subtraction>

     1022     <subtraction name="SstBotCirRibPri-ChildForSstBotCirRibBase0xc26e0a0">
     1023       <first ref="SstBotCirRibPri0xc26d4e0"/>
     1024       <second ref="SstBotRibBase00xc0d1e90"/>
     1025       <position name="SstBotCirRibPri-ChildForSstBotCirRibBase0xc26e0a0_pos" unit="mm" x="1255" y="0" z="0"/>
     1026     </subtraction>

     1020     <tube aunit="deg" deltaphi="45" lunit="mm" name="SstBotCirRibPri0xc26d4e0" rmax="2000" rmin="1980" startphi="0" z="430"/>
     1027     <box lunit="mm" name="SstBotRibBase10xbfa4950" x="2020" y="25" z="440"/>



lvidx65 : First Cause : tube deltaphi is not handled
-----------------------------------------------------------

::

    simon:boostrap blyth$ grep deltaphi /tmp/g4_00.gdml  | grep -v \"360\"
        <tube aunit="deg" deltaphi="44.6352759021238" lunit="mm" name="BlackCylinder0xc1762e8" rmax="2262.15" rmin="2259.15" startphi="0" z="997"/>
        <tube aunit="deg" deltaphi="45" lunit="mm" name="SstBotCirRibPri0xc26d4e0" rmax="2000" rmin="1980" startphi="0" z="430"/>
        <tube aunit="deg" deltaphi="45" lunit="mm" name="SstTopCirRibPri0xc2648b8" rmax="1220" rmin="1200" startphi="0" z="231.89"/>
        <sphere aunit="deg" deltaphi="180" deltatheta="180" lunit="mm" name="UpperAcrylicHemisphere0xc0b2ac0" rmax="10.035" rmin="0" startphi="0" starttheta="0"/>
        <sphere aunit="deg" deltaphi="180" deltatheta="180" lunit="mm" name="LowerAcrylicHemisphere0xc0b2be8" rmax="10.035" rmin="0" startphi="0" starttheta="0"/>
    simon:boostrap blyth$ 



::

     opticks-tbool-vi 65
     opticks-tbool 65     # ana CSG level viz : looks like a cylinder ring with one cut 



     62 tbool65--(){ cat << EOP
     63 
     64 import logging
     65 import numpy as np
     66 log = logging.getLogger(__name__)
     67 from opticks.ana.base import opticks_main
     68 from opticks.analytic.csg import CSG  
     69 args = opticks_main(csgpath="$TMP/tbool/65")
     70 
     71 CSG.boundary = args.testobject
     72 CSG.kwa = dict(verbosity="0", poly="IM", resolution="20")
     73 #CSG.kwa = dict(verbosity="0", poly="HY", level="5")
     74 
     75 
     76 a = CSG("cylinder", param = [0.000,0.000,0.000,2000.000],param1 = [-215.000,215.000,0.000,0.000])
     77 b = CSG("cylinder", param = [0.000,0.000,0.000,1980.000],param1 = [-217.150,217.150,0.000,0.000])
     78 ab = CSG("difference", left=a, right=b)
     79 
     80 c = CSG("box3", param = [2020.000,25.000,440.000,0.000],param1 = [0.000,0.000,0.000,0.000])
     81 c.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[1255.000,0.000,0.000,1.000]]
     82 abc = CSG("difference", left=ab, right=c)
     83 
     84 d = CSG("box3", param = [2020.000,25.000,440.000,0.000],param1 = [0.000,0.000,0.000,0.000])
     85 d.transform = [[0.707,-0.707,0.000,0.000],[0.707,0.707,0.000,0.000],[0.000,0.000,1.000,0.000],[887.419,887.419,0.000,1.000]]
     86 abcd = CSG("difference", left=abc, right=d)
     87 
     88 
     89 
     90 obj = abcd
     91 
     92 con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=args.container , poly="IM", resolution="20" )
     93 CSG.Serialize([con, obj], args.csgpath )
     94 
     95 EOP
     96 }


::


    DBGMESH=SstBotCirRibBase0x NSceneMeshTest 
    ...
    2017-07-03 20:58:17.359 INFO  [2996117] [NScene::dumpCSG@412] NScene::dumpCSG num_csg 249 dbgmesh SstBotCirRibBase0x
    2017-07-03 20:58:17.359 INFO  [2996117] [NCSG::dump@905] NCSG::dump
     NCSG  ix   76 surfpoints   33 so SstBotCirRibBase0xc26e2d0                lv /dd/Geometry/AdDetails/lvSstBotCirRibBase0xc26e220
    NCSG::dump (root) [ 0:di] OPER  v:0
             L [ 1:di] OPER  v:0
             L [ 3:di] OPER  v:0
             L [ 7:cy] PRIM  v:0 bb  mi  (-2000.00 -2000.00 -215.00)  mx  (2000.00 2000.00  215.00)  si  (4000.00 4000.00  430.00) 
             R [ 8:cy] PRIM  v:0 bb  mi  (-1980.00 -1980.00 -217.15)  mx  (1980.00 1980.00  217.15)  si  (3960.00 3960.00  434.30) 
             R [ 4:bo] PRIM  v:0 bb  mi  ( 245.00  -12.50 -220.00)  mx  (2265.00   12.50  220.00)  si  (2020.00   25.00  440.00) 
             R [ 2:bo] PRIM  v:0 bb  mi  ( 164.40  164.40 -220.00)  mx  (1610.44 1610.44  220.00)  si  (1446.03 1446.03  440.00) 
     composite_bb  mi  (-2000.00 -2000.00 -215.00)  mx  (2000.00 2000.00  215.00)  si  (4000.00 4000.00  430.00) 
    NParameters::dump
 


::

    simon:ggeo blyth$ DBGMESH=SstBotCirRibBase0x NSceneMeshTest 
    2017-07-03 20:58:12.915 INFO  [2996117] [NGLTF::load@35] NGLTF::load path /tmp/blyth/opticks/tgltf-t-/sc.gltf
    2017-07-03 20:58:13.431 INFO  [2996117] [NGLTF::load@62] NGLTF::load DONE
    2017-07-03 20:58:13.457 INFO  [2996117] [NSceneConfig::NSceneConfig@12] NSceneConfig::NSceneConfig cfg check_surf_containment=0,check_aabb_containment=0
            check_surf_containment :                    0
            check_aabb_containment :                    0
    2017-07-03 20:58:13.457 INFO  [2996117] [NScene::init@140] NScene::init START age(s) 12311 days   0.142
    2017-07-03 20:58:13.457 INFO  [2996117] [NScene::load_csg_metadata@252] NScene::load_csg_metadata verbosity 1 num_meshes 249
    2017-07-03 20:58:13.875 INFO  [2996117] [NScene::postimportnd@491] NScene::postimportnd numNd 12230 num_selected 12230 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-07-03 20:58:14.048 INFO  [2996117] [NScene::count_progeny_digests@866] NScene::count_progeny_digests verbosity 1 node_count 12230 digest_size 249
    2017-07-03 20:58:15.328 INFO  [2996117] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-03 20:58:15.727 INFO  [2996117] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-03 20:58:15.769 INFO  [2996117] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-03 20:58:17.358 INFO  [2996117] [NScene::postimportmesh@509] NScene::postimportmesh numNd 12230 dbgnode -1 dbgnode_list 0 verbosity 1
                      check_surf_containment : 0
                      check_aabb_containment : 0
    2017-07-03 20:58:17.359 INFO  [2996117] [NScene::init@180] NScene::init DONE
    2017-07-03 20:58:17.359 INFO  [2996117] [NScene::dumpCSG@412] NScene::dumpCSG num_csg 249 dbgmesh SstBotCirRibBase0x
    2017-07-03 20:58:17.359 INFO  [2996117] [NCSG::dump@905] NCSG::dump
     NCSG  ix   76 surfpoints   33 so SstBotCirRibBase0xc26e2d0                lv /dd/Geometry/AdDetails/lvSstBotCirRibBase0xc26e220
    NCSG::dump (root) [ 0:di] OPER  v:0
             L [ 1:di] OPER  v:0
             L [ 3:di] OPER  v:0
             L [ 7:cy] PRIM  v:0 bb  mi  (-2000.00 -2000.00 -215.00)  mx  (2000.00 2000.00  215.00)  si  (4000.00 4000.00  430.00) 
             R [ 8:cy] PRIM  v:0 bb  mi  (-1980.00 -1980.00 -217.15)  mx  (1980.00 1980.00  217.15)  si  (3960.00 3960.00  434.30) 
             R [ 4:bo] PRIM  v:0 bb  mi  ( 245.00  -12.50 -220.00)  mx  (2265.00   12.50  220.00)  si  (2020.00   25.00  440.00) 
             R [ 2:bo] PRIM  v:0 bb  mi  ( 164.40  164.40 -220.00)  mx  (1610.44 1610.44  220.00)  si  (1446.03 1446.03  440.00) 
     composite_bb  mi  (-2000.00 -2000.00 -215.00)  mx  (2000.00 2000.00  215.00)  si  (4000.00 4000.00  430.00) 
    NParameters::dump
             lvname : /dd/Geometry/AdDetails/lvSstBotCirRibBase0xc26e220
             soname : SstBotCirRibBase0xc26e2d0
          verbosity :               0
         resolution :              20
               poly :              IM
             height :               3
    2017-07-03 20:58:17.359 INFO  [2996117] [NCSG::dump_surface_points@1195] dsp num_sp 33 dmax 200
     i    0 sp (     -0.000  2000.000  -215.000)
     i    1 sp (  -2000.000    -0.000  -215.000)
     i    2 sp (      0.000 -2000.000  -215.000)
     i    3 sp (     -0.000  2000.000  -112.875)
     i    4 sp (  -2000.000    -0.000  -112.875)
     i    5 sp (      0.000 -2000.000  -112.875)
     i    6 sp (     -0.000  2000.000   -10.750)
     i    7 sp (  -2000.000    -0.000   -10.750)
     i    8 sp (      0.000 -2000.000   -10.750)
     i    9 sp (     -0.000  2000.000    91.375)
     i   10 sp (  -2000.000    -0.000    91.375)
     i   11 sp (      0.000 -2000.000    91.375)
     i   12 sp (     -0.000  2000.000   193.500)
     i   13 sp (  -2000.000    -0.000   193.500)
     i   14 sp (      0.000 -2000.000   193.500)
     i   15 sp (     -0.000  2000.000   215.000)
     i   16 sp (  -2000.000    -0.000   215.000)
     i   17 sp (      0.000 -2000.000   215.000)
     i   18 sp (     -0.000  2000.000  -215.000)
     i   19 sp (  -2000.000    -0.000  -215.000)
     i   20 sp (      0.000 -2000.000  -215.000)
     i   21 sp (     -0.000  1980.000  -114.004)
     i   22 sp (  -1980.000    -0.000  -114.004)
     i   23 sp (      0.000 -1980.000  -114.004)
     i   24 sp (     -0.000  1980.000   -10.857)
     i   25 sp (  -1980.000    -0.000   -10.857)
     i   26 sp (      0.000 -1980.000   -10.857)
     i   27 sp (     -0.000  1980.000    92.289)
     i   28 sp (  -1980.000    -0.000    92.289)
     i   29 sp (      0.000 -1980.000    92.289)
     i   30 sp (     -0.000  1980.000   195.435)
     i   31 sp (  -1980.000    -0.000   195.435)
     i   32 sp (      0.000 -1980.000   195.435)
     csg.index (mesh_id) 76 num nodes 16
     node idx :  4440 4441 4442 4443 4444 4445 4446 4447 6100 6101 ... 




