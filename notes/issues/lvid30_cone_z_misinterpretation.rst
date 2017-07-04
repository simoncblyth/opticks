
lvid 30 OcrGdsTfbInLso0xbfa2370 : cone-z misinterpretation
=============================================================

FIXED
--------

Adopt centered cone and regenerate GLTF::

::

    simon:analytic blyth$ vi gdml.py 
    simon:analytic blyth$ gdml2gltf.py 
    args: /Users/blyth/opticks/bin/gdml2gltf.py
    [2017-07-04 13:31:12,899] p78920 {/Users/blyth/opticks/analytic/gdml.py:993} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 
    [2017-07-04 13:31:12,947] p78920 {/Users/blyth/opticks/analytic/gdml.py:1007} INFO - wrapping gdml element  
    [2017-07-04 13:31:13,838] p78920 {/Users/blyth/opticks/analytic/treebase.py:504} INFO - apply_selection OpticksQuery  range [] index 0 depth 0   Node.selected_count 12230 
    [2017-07-04 13:31:13,838] p78920 {/Users/blyth/opticks/analytic/sc.py:345} INFO - add_tree_gdml START maxdepth:0 maxcsgheight:3 nodesCount:    0
    [2017-07-04 13:31:13,838] p78920 {/Users/blyth/opticks/analytic/treebase.py:34} WARNING - returning DummyTopPV placeholder transform
    [2017-07-04 13:31:16,980] p78920 {/Users/blyth/opticks/analytic/sc.py:348} INFO - add_tree_gdml DONE maxdepth:0 maxcsgheight:3 nodesCount:12230 tlvCount:249  tgNd:                           top Nd ndIdx:  0 soIdx:0 nch:1 par:-1 matrix:[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]   
    [2017-07-04 13:31:16,980] p78920 {/Users/blyth/opticks/analytic/sc.py:381} INFO - saving to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf 
    [2017-07-04 13:31:17,314] p78920 {/Users/blyth/opticks/analytic/sc.py:370} INFO - save_extras /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras  : saved 249 
    [2017-07-04 13:31:17,314] p78920 {/Users/blyth/opticks/analytic/sc.py:374} INFO - write 249 lines to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/csg.txt 
    [2017-07-04 13:31:18,102] p78920 {/Users/blyth/opticks/analytic/sc.py:390} INFO - also saving to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.pretty.gltf 
    simon:analytic blyth$ 



::



     opticks-tbool 30     # huh : just looks like cy-cy pipe 
     opticks-tbool-vi 30  # checking just the cone... tis very flat, the intersection will just slightly change the top tube edge 

     op --dlv30           # looks like offset pipe, not much impact from the cone
     op --dlv30 --gltf 1  # ditto .. looks same 

     op --dlv30 --gltf 3  # viewing ana raytrace together with tri poly : shows z offset 
            ~/opticks_refs/lvidx30_cycyco_intersect_z_offset.png


     op --dlv30 --gmeshlib --dbgmesh OcrGdsTfbInLso0xbfa2370



     DBGMESH=OcrGdsTfbInLso0xbfa2370 NSceneMeshTest


     62 tbool30--(){ cat << EOP
     63 
     64 import logging
     65 import numpy as np
     66 log = logging.getLogger(__name__)
     67 from opticks.ana.base import opticks_main
     68 from opticks.analytic.csg import CSG  
     69 args = opticks_main(csgpath="$TMP/tbool/30")
     70 
     71 CSG.boundary = args.testobject
     72 CSG.kwa = dict(verbosity="0", poly="IM", resolution="20")
     73 #CSG.kwa = dict(verbosity="0", poly="HY", level="5")
     74 
     75 
     76 a = CSG("cone", param = [5879.795,0.000,125.000,301.596],param1 = [0.000,0.000,0.000,0.000])
     77 b = CSG("cylinder", param = [0.000,0.000,0.000,32.500],param1 = [-150.798,150.798,0.000,0.000])
     78 c = CSG("cylinder", param = [0.000,0.000,0.000,31.500],param1 = [-152.306,152.306,0.000,0.000])
     79 bc = CSG("difference", left=b, right=c)
     80 bc.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[516.623,-1247.237,0.000,1.000]]
     81 
     82 # add another cylinder beside the other that doesnt suffer the cone intersect 
     83 # shows the cone intersect is acting to chop off the bottom half of the tube

     84 bc2 = CSG("difference", left=b, right=c)
     85 bc2.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[516.623+65,-1247.237,0.000,1.000]]
     86 
     87 abc = CSG("intersection", left=a, right=bc)
     88 
     89 #obj = a
     90 obj = abc
     91 
     92 con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=args.container , poly="IM", resolution="20" )
     93 CSG.Serialize([con, obj, bc2, a], args.csgpath )
     94 
     95 EOP
     96 }



Am I misinterpreting cone-z or tube-z 

* apparently, i'm using centered-z for tube but uncentered for cone ?
* :doc:`gdml_cone_tube_z_interpretation`




::

     583     <intersection name="OcrGdsTfbInLso0xbfa2370">
     584       <first ref="OcrGdsTfbInLsoCon0xc3527a0"/>
     585       <second ref="OcrGdsTfbInLsoTub0xc352858"/>
     586       <position name="OcrGdsTfbInLso0xbfa2370_pos" unit="mm" x="516.622633692872" y="-1247.23736889024" z="0"/>
     587     </intersection>

     581     <cone aunit="deg" deltaphi="360" lunit="mm" name="OcrGdsTfbInLsoCon0xc3527a0" rmax1="5879.79529435974" rmax2="125" rmin1="0" rmin2="0" startphi="0" z="301.596041605889"/>
     582     <tube aunit="deg" deltaphi="360" lunit="mm" name="OcrGdsTfbInLsoTub0xc352858" rmax="32.5" rmin="31.5" startphi="0" z="301.596041605889"/>

     In [1]: 301.596041605889/2.
     Out[1]: 150.7980208029445




::

       1214.74    OcrGdsTfbInLso0xbfa2370 lvidx  30 

       amn (      0.000 -1279.737     0.000) 
       amx (    549.123     0.000   150.798) 

       bmn (    484.130 -1279.740  -150.798) dmn (   -484.130     0.003   150.798) 
       bmx (    549.130 -1214.740    87.691) dmx (     -0.007  1214.740    63.107)

 


    simon:~ blyth$ op --dlv30 --gmeshlib --dbgmesh OcrGdsTfbInLso0xbfa2370
    === op-cmdline-binary-match : finds 1st argument with associated binary : --gmeshlib
    240 -rwxr-xr-x  1 blyth  staff  120332 Jul  4 09:51 /usr/local/opticks/lib/GMeshLibTest
    proceeding : /usr/local/opticks/lib/GMeshLibTest --dlv30 --gmeshlib --dbgmesh OcrGdsTfbInLso0xbfa2370
    2017-07-04 11:51:29.326 INFO  [3062338] [OpticksQuery::dumpQuery@81] OpticksQuery::init queryType range query_string range:3155:3156,range:3167:3168 query_name NULL query_index 0 nrange 4 : 3155 : 3156 : 3167 : 3168
    2017-07-04 11:51:29.326 INFO  [3062338] [Opticks::init@319] Opticks::init DONE OpticksResource::desc digest 54dce5b6a7a226fb440eab1c42e16616 age.tot_seconds    569 age.tot_minutes  9.483 age.tot_hours  0.158 age.tot_days      0.007
    2017-07-04 11:51:29.342 INFO  [3062338] [GMeshLib::loadMeshes@206] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.54dce5b6a7a226fb440eab1c42e16616.dae
    2017-07-04 11:51:29.380 INFO  [3062338] [GMesh::dump@1133] GMesh::dump num_vertices 96 num_faces 192 num_solids 0 name OcrGdsTfbInLso0xbfa2370
     low  -
     high -
     dim  -
     cen  - extent 0
     ce   (   516.630  -1247.240    -31.554    119.244)
     bb.max   (   549.130  -1214.740     87.691)
     bb.min   (   484.130  -1279.740   -150.798)
        0 vtx (   548.022  -1255.650     84.942) nrm (     0.020     -0.049      0.999)
        1 vtx (   549.130  -1247.240     85.330) nrm (     0.020     -0.049      0.999)
        2 vtx (   547.056  -1255.400     84.974) nrm (     0.020     -0.049      0.999)
        3 vtx (   548.130  -1247.240     85.350) nrm (     0.020     -0.049      0.999)
        4 vtx (   544.775  -1263.490     84.625) nrm (     0.020     -0.049      0.999)
        5 vtx (   547.056  -1239.090     85.770) nrm (     0.020     -0.049      0.999)
        6 vtx (   543.909  -1262.990     84.667) nrm (     0.020     -0.049      0.999)
        7 vtx (   548.022  -1238.830     85.763) nrm (     0.020     -0.049      0.999)
        8 vtx (   539.611  -1270.220     84.400) nrm (     0.020     -0.049      0.999)
        9 vtx (   543.909  -1231.490     86.205) nrm (     0.020     -0.049      0.999)
       10 vtx (   538.903  -1269.520     84.449) nrm (     0.020     -0.049      0.999)
       11 vtx (   544.775  -1230.990     86.212) nrm (     0.020     -0.049      0.999)





::

    DBGMESH=OcrGdsTfbInLso0xbfa2370 NSceneMeshTest 

    simon:boostrap blyth$ DBGMESH=OcrGdsTfbInLso0xbfa2370 NSceneMeshTest 
    2017-07-04 11:29:34.442 INFO  [3055850] [main@29] NSceneMeshTest gltfbase /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300 gltfname g4_00.gltf gltfconfig check_surf_containment=0,check_aabb_containment=0
    2017-07-04 11:29:34.443 INFO  [3055850] [NGLTF::load@35] NGLTF::load path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf
    2017-07-04 11:29:34.951 INFO  [3055850] [NGLTF::load@62] NGLTF::load DONE
    2017-07-04 11:29:34.976 INFO  [3055850] [NSceneConfig::NSceneConfig@13] NSceneConfig::NSceneConfig cfg check_surf_containment=0,check_aabb_containment=0
            check_surf_containment :                    0
            check_aabb_containment :                    0
    2017-07-04 11:29:34.976 INFO  [3055850] [NScene::init@154] NScene::init START age(s) 61405 days   0.711
    2017-07-04 11:29:34.976 INFO  [3055850] [NScene::load_csg_metadata@274] NScene::load_csg_metadata verbosity 1 num_meshes 249
    2017-07-04 11:29:35.393 INFO  [3055850] [NScene::postimportnd@528] NScene::postimportnd numNd 12230 num_selected 2 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-07-04 11:29:35.564 INFO  [3055850] [NScene::count_progeny_digests@902] NScene::count_progeny_digests verbosity 1 node_count 12230 digest_size 249
    2017-07-04 11:29:38.826 INFO  [3055850] [NScene::postimportmesh@546] NScene::postimportmesh numNd 12230 dbgnode -1 dbgnode_list 0 verbosity 1
                      check_surf_containment : 0
                      check_aabb_containment : 0
                          disable_instancing : 0
    2017-07-04 11:29:38.826 INFO  [3055850] [NScene::init@202] NScene::init DONE
    2017-07-04 11:29:38.826 INFO  [3055850] [NScene::dumpCSG@434] NScene::dumpCSG num_csg 249 dbgmesh OcrGdsTfbInLso0xbfa2370
    2017-07-04 11:29:38.826 INFO  [3055850] [NCSG::dump@907] NCSG::dump
     NCSG  ix   43 surfpoints   25 so OcrGdsTfbInLso0xbfa2370                  lv /dd/Geometry/AdDetails/lvOcrGdsTfbInLso0xc3529c0
    NCSG::dump (root) [ 0:in] OPER  v:0
             L [ 1:co] PRIM  v:0 bb  mi  (-5879.80 -5879.80    0.00)  mx  (5879.80 5879.80  301.60)  si  (11759.59 11759.59  301.60) 
             R [ 2:di] OPER  v:0
             L [ 5:cy] PRIM  v:0 bb  mi  ( 484.12 -1279.74 -150.80)  mx  ( 549.12 -1214.74  150.80)  si  (  65.00   65.00  301.60) 
             R [ 6:cy] PRIM  v:0 bb  mi  ( 485.12 -1278.74 -152.31)  mx  ( 548.12 -1215.74  152.31)  si  (  63.00   63.00  304.61) 
     composite_bb  mi  (   0.00 -1279.74    0.00)  mx  ( 549.12    0.00  150.80)  si  ( 549.12 1279.74  150.80) 
    NParameters::dump
             lvname : /dd/Geometry/AdDetails/lvOcrGdsTfbInLso0xc3529c0
             soname : OcrGdsTfbInLso0xbfa2370
          verbosity :               0
         resolution :              20
               poly :              IM
             height :               2
    2017-07-04 11:29:38.826 INFO  [3055850] [NCSG::dump_surface_points@1197] dsp num_sp 25 dmax 200
     i    0 sp (    549.123 -1247.237    64.089)
     i    1 sp (    516.623 -1214.737    64.089)
     i    2 sp (    484.123 -1247.237    64.089)
     i    3 sp (    516.623 -1279.737    64.089)
     i    4 sp (    549.123 -1247.237    64.089)
     i    5 sp (    549.123 -1247.237   135.718)
     i    6 sp (    516.623 -1214.737   135.718)
     i    7 sp (    484.123 -1247.237   135.718)
     i    8 sp (    516.623 -1279.737   135.718)
     i    9 sp (    549.123 -1247.237   135.718)
     i   10 sp (    549.123 -1247.237   150.798)
     i   11 sp (    516.623 -1214.737   150.798)
     i   12 sp (    484.123 -1247.237   150.798)
     i   13 sp (    516.623 -1279.737   150.798)
     i   14 sp (    549.123 -1247.237   150.798)
     i   15 sp (    548.123 -1247.237    64.730)
     i   16 sp (    516.623 -1215.737    64.730)
     i   17 sp (    485.123 -1247.237    64.730)
     i   18 sp (    516.623 -1278.737    64.730)
     i   19 sp (    548.123 -1247.237    64.730)
     i   20 sp (    548.123 -1247.237   137.075)
     i   21 sp (    516.623 -1215.737   137.075)
     i   22 sp (    485.123 -1247.237   137.075)
     i   23 sp (    516.623 -1278.737   137.075)
     i   24 sp (    548.123 -1247.237   137.075)
     csg.index (mesh_id) 43 num nodes 2
     node idx :  3167 4827 . 


       1214.74    OcrGdsTfbInLso0xbfa2370 lvidx  30 

       amn (      0.000 -1279.737     0.000) 
       amx (    549.123     0.000   150.798) 

       bmn (    484.130 -1279.740  -150.798) dmn (   -484.130     0.003   150.798) 
       bmx (    549.130 -1214.740    87.691) dmx (     -0.007  1214.740    63.107)





