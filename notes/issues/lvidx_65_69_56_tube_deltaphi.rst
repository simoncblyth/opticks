
lvidx_65_69_56_tube_deltaphi
===============================

TODO 
-----

* revisit the numbers/viz with deltaphi enabled
* revisit tree balancing : current kludge is to disable balancing for trees that include deltaphi segmenting...
  but that causes poor performance



Are the top 3 biggies all tube deltaphi ?
--------------------------------------------


::

    2017-07-06 16:09:32.174 INFO  [3663887] [GScene::compareMeshes_GMeshBB@436] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF parsurf_level 2 parsurf_target 500
       3869.75               RadialShieldUnit0xc3d7da8 lvidx  56 nsp    507 intersection cylinder 
       3407.72               SstBotCirRibBase0xc26e2d0 lvidx  65 nsp   1212 difference cylinder box3 
       2074.65               SstTopCirRibBase0xc264f78 lvidx  69 nsp   1728 intersection cylinder box3 



Looks like lvid 56 too
-------------------------

::

    simon:tmp blyth$ grep deltaphi g4_00.gdml | grep -v deltaphi=\"360\" 
        <tube aunit="deg" deltaphi="44.6352759021238" lunit="mm" name="BlackCylinder0xc1762e8" rmax="2262.15" rmin="2259.15" startphi="0" z="997"/>
        <tube aunit="deg" deltaphi="45" lunit="mm" name="SstBotCirRibPri0xc26d4e0" rmax="2000" rmin="1980" startphi="0" z="430"/>
        <tube aunit="deg" deltaphi="45" lunit="mm" name="SstTopCirRibPri0xc2648b8" rmax="1220" rmin="1200" startphi="0" z="231.89"/>
        <sphere aunit="deg" deltaphi="180" deltatheta="180" lunit="mm" name="UpperAcrylicHemisphere0xc0b2ac0" rmax="10.035" rmin="0" startphi="0" starttheta="0"/>
        <sphere aunit="deg" deltaphi="180" deltatheta="180" lunit="mm" name="LowerAcrylicHemisphere0xc0b2be8" rmax="10.035" rmin="0" startphi="0" starttheta="0"/>
    simon:tmp blyth$ 


::

  799     <subtraction name="RadialShieldUnit0xc3d7da8">
  800       <first ref="BlackCylinder-ChildForRadialShieldUnit0xc3d8628"/>
  801       <second ref="PmtHole60xc3d7cb8"/>
  802       <position name="RadialShieldUnit0xc3d7da8_pos" unit="mm" x="1797.86532031977" y="1370.48119742355" z="-250"/>
  803       <rotation name="RadialShieldUnit0xc3d7da8_rot" unit="deg" x="-37.3176379510619" y="90" z="0"/>
  804     </subtraction>



NCylinder : how to do phi segment SDF ? think 2 cutting planes
-----------------------------------------------------------------

* brought NSlab up to scratch 
* tested slicing by slab intersects in tboolean-cyslab

::

    1385 tboolean-cyslab(){ TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- ; }
    1386 tboolean-cyslab-(){  $FUNCNAME- | python $* ; } 
    1387 tboolean-cyslab--(){ cat << EOP 
    1388 import numpy as np
    1389 from opticks.ana.base import opticks_main
    1390 from opticks.analytic.csg import CSG  
    1391 args = opticks_main(csgpath="$TMP/$FUNCNAME")
    1392 
    1393 CSG.boundary = args.testobject
    1394 CSG.kwa = dict(poly="IM", resolution="50")
    1395 
    1396 container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="MC", nx="20" )
    1397   
    1398 ca = CSG("cylinder", param=[0,0,0,500], param1=[-100,100,0,0] )
    1399 cb = CSG("cylinder", param=[0,0,0,400], param1=[-101,101,0,0] )
    1400 cy = ca - cb 
    1401 
    1402 
    1403 sa = CSG("slab", param=[1,1,0,0],param1=[0,501,0,0] )  # normalization done in NSlab.hpp/init_slab
    1404 sb = CSG("slab", param=[-1,1,0,0],param1=[0,501,0,0] )  # normalization done in NSlab.hpp/init_slab
    1405 
    1406 cysa = cy*sa 
    1407 cysb = cy*sb 
    1408 cysasb = cy*sa*sb 
    1409 
    1410 obj = cysasb
    1411 
    1412 CSG.Serialize([container, obj], args.csgpath )
    1413 
    1414 EOP
    1415 }




multi unbound ?
----------------

::

    simon:analytic blyth$ gdml2gltf.py 
    args: /Users/blyth/opticks/bin/gdml2gltf.py
    [2017-07-07 20:16:36,124] p98538 {/Users/blyth/opticks/analytic/gdml.py:1045} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 
    [2017-07-07 20:16:36,164] p98538 {/Users/blyth/opticks/analytic/gdml.py:1059} INFO - wrapping gdml element  
    [2017-07-07 20:16:37,081] p98538 {/Users/blyth/opticks/analytic/treebase.py:504} INFO - apply_selection OpticksQuery  range [] index 0 depth 0   Node.selected_count 12230 
    [2017-07-07 20:16:37,081] p98538 {/Users/blyth/opticks/analytic/sc.py:345} INFO - add_tree_gdml START maxdepth:0 maxcsgheight:3 nodesCount:    0
    [2017-07-07 20:16:37,081] p98538 {/Users/blyth/opticks/analytic/treebase.py:34} WARNING - returning DummyTopPV placeholder transform
    [2017-07-07 20:16:37,970] p98538 {/Users/blyth/opticks/analytic/gdml.py:460} INFO - as_cylinder doing slab segmenting : name BlackCylinder0xc1762e8 phi0 0.0 phi1 44.6352759021 dist 2263.15 
    [2017-07-07 20:16:37,998] p98538 {/Users/blyth/opticks/analytic/gdml.py:460} INFO - as_cylinder doing slab segmenting : name SstBotCirRibPri0xc26d4e0 phi0 0.0 phi1 45.0 dist 2001.0 
    [2017-07-07 20:16:38,010] p98538 {/Users/blyth/opticks/analytic/gdml.py:460} INFO - as_cylinder doing slab segmenting : name SstTopCirRibPri0xc2648b8 phi0 0.0 phi1 45.0 dist 1221.0 
    [2017-07-07 20:16:40,279] p98538 {/Users/blyth/opticks/analytic/sc.py:348} INFO - add_tree_gdml DONE maxdepth:0 maxcsgheight:3 nodesCount:12230 tlvCount:249  tgNd:                           top Nd ndIdx:  0 soIdx:0 nch:1 par:-1 matrix:[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]   
    [2017-07-07 20:16:40,279] p98538 {/Users/blyth/opticks/analytic/sc.py:381} INFO - saving to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf 
    [2017-07-07 20:16:40,756] p98538 {/Users/blyth/opticks/analytic/sc.py:370} INFO - save_extras /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras  : saved 249 
    [2017-07-07 20:16:40,757] p98538 {/Users/blyth/opticks/analytic/sc.py:374} INFO - write 249 lines to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/csg.txt 
    [2017-07-07 20:16:41,585] p98538 {/Users/blyth/opticks/analytic/sc.py:390} INFO - also saving to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.pretty.gltf 
    simon:analytic blyth$ 
    simon:analytic blyth$ 
    simon:analytic blyth$ op --gltf 4
    288 -rwxr-xr-x  1 blyth  staff  143804 Jul  7 17:51 /usr/local/opticks/lib/OKTest
    proceeding : /usr/local/opticks/lib/OKTest --gltf 4
    2017-07-07 20:17:06.172 INFO  [3955557] [OpticksQuery::dumpQuery@81] OpticksQuery::init queryType range query_string range:3153:12221 query_name NULL query_index 0 nrange 2 : 3153 : 12221
    2017-07-07 20:17:06.173 INFO  [3955557] [Opticks::init@319] Opticks::init DONE OpticksResource::desc digest 96ff965744a2f6b78c24e33c80d3a4cd age.tot_seconds 348711 age.tot_minutes 5811.850 age.tot_hours 96.864 age.tot_days      4.036
    2017-07-07 20:17:06.340 INFO  [3955557] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-07-07 20:17:06.471 INFO  [3955557] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-07-07 20:17:06.547 INFO  [3955557] [GMeshLib::loadMeshes@206] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-07-07 20:17:06.586 INFO  [3955557] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-07-07 20:17:06.586 INFO  [3955557] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-07-07 20:17:06.586 INFO  [3955557] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-07-07 20:17:06.587 INFO  [3955557] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-07-07 20:17:06.592 INFO  [3955557] [GGeo::loadAnalyticPmt@761] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    2017-07-07 20:17:06.593 INFO  [3955557] [NGLTF::load@35] NGLTF::load path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf
    2017-07-07 20:17:07.102 INFO  [3955557] [NGLTF::load@62] NGLTF::load DONE
    2017-07-07 20:17:07.127 INFO  [3955557] [NSceneConfig::NSceneConfig@42] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0]
            check_surf_containment :                    0
            check_aabb_containment :                    0
    2017-07-07 20:17:07.127 INFO  [3955557] [NScene::init@177] NScene::init START age(s) 26 days   0.000
    2017-07-07 20:17:07.127 INFO  [3955557] [NScene::load_csg_metadata@297] NScene::load_csg_metadata verbosity 1 num_meshes 249
    2017-07-07 20:17:07.546 INFO  [3955557] [NScene::postimportnd@543] NScene::postimportnd numNd 12230 num_selected 12230 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-07-07 20:17:07.714 INFO  [3955557] [NScene::count_progeny_digests@917] NScene::count_progeny_digests verbosity 1 node_count 12230 digest_size 249
    2017-07-07 20:17:09.946 INFO  [3955557] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    Assertion failed: (!(l_unbound && r_unbound) && " combination of two unbounded prmitives is not allowed "), function get_composite_bbox, file /Users/blyth/opticks/opticksnpy/NNode.cpp, line 313.
    /Users/blyth/opticks/bin/op.sh: line 648: 98750 Abort trap: 6           /usr/local/opticks/lib/OKTest --gltf 4
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:analytic blyth$ 
    simon:analytic blyth$ 
    simon:analytic blyth$ 



Hmm looks like it got balanced and messed up in the process
-------------------------------------------------------------


::

    065 tbool69--(){ cat << EOP
     66 
     67 import logging
     68 import numpy as np
     69 log = logging.getLogger(__name__)
     70 from opticks.ana.base import opticks_main
     71 from opticks.analytic.csg import CSG  
     72 args = opticks_main(csgpath="$TMP/tbool/69")
     73 
     74 CSG.boundary = args.testobject
     75 CSG.kwa = dict(verbosity="0", poly="IM", resolution="20")
     76 #CSG.kwa = dict(verbosity="0", poly="HY", level="5")
     77 
     78 # generated by tboolean.py : 20170707-2016 
     79 # opticks-;opticks-tbool 69 
     80 # opticks-;opticks-tbool-vi 69 
     81 
     82 
     83 a = CSG("cylinder", param = [0.000,0.000,0.000,1220.000],param1 = [-115.945,115.945,0.000,0.000])
     84 b = CSG("cylinder", param = [0.000,0.000,0.000,1200.000],param1 = [-117.104,117.104,0.000,0.000],complement = True)
     85 ab = CSG("intersection", left=a, right=b)
     86 
     87 c = CSG("slab", param = [0.000,1.000,0.000,0.000],param1 = [0.000,1221.000,0.000,0.000])
     88 d = CSG("slab", param = [0.707,-0.707,0.000,0.000],param1 = [0.000,1221.000,0.000,0.000])
     89 cd = CSG("intersection", left=c, right=d)
     90 
     91 abcd = CSG("intersection", left=ab, right=cd)
     92 
     93 e = CSG("box3", param = [2460.000,20.000,231.890,0.000],param1 = [0.000,0.000,0.000,0.000],complement = True)
     94 e.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,0.000,1.000]]
     95 f = CSG("box3", param = [2460.000,100.000,20.000,0.000],param1 = [0.000,0.000,0.000,0.000],complement = True)
     96 f.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-105.945,1.000]]
     97 ef = CSG("intersection", left=e, right=f)
     98 
     99 g = CSG("box3", param = [2460.000,20.000,231.890,0.000],param1 = [0.000,0.000,0.000,0.000],complement = True)
    100 g.transform = [[0.707,-0.707,0.000,0.000],[0.707,0.707,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,0.000,1.000]]
    101 h = CSG("box3", param = [2460.000,100.000,20.000,0.000],param1 = [0.000,0.000,0.000,0.000],complement = True)
    102 h.transform = [[0.707,-0.707,0.000,0.000],[0.707,0.707,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-105.945,1.000]]
    103 gh = CSG("intersection", left=g, right=h)
    104 
    105 efgh = CSG("intersection", left=ef, right=gh)
    106 
    107 abcdefgh = CSG("intersection", left=abcd, right=efgh)
    108 
    109 
    110 
    111 obj = abcdefgh





Allowing double unbound
--------------------------

::

    opticks-;opticks-tbool 69   ## nothing visible

    opticks-:opticks-tbool 69   ## appears once disable tree balancing, segmenting works but note small artifact


    op --dlv65 --gltf 3  ## looks ok at a glance... need to revisit the numbers




::


    2017-07-07 20:54:11.485 INFO  [3968900] [GScene::importMeshes@316] GScene::importMeshes DONE num_meshes 249
    2017-07-07 20:54:11.485 INFO  [3968900] [GScene::compareMeshes_GMeshBB@435] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF parsurf_level 2 parsurf_target 200
       377.713               SstBotCirRibBase0xc26e2d0 lvidx  65 nsp    204        intersection difference cylinder slab box3   nds[ 16]  4440 4441 4442 4443 4444 4445 4446 4447 6100 6101 ... 
        345.51                SstTopRadiusRib0xc271720 lvidx  66 nsp    408                  difference box3 convexpolyhedron   nds[ 16]  4448 4449 4450 4451 4452 4453 4454 4455 6108 6109 ... 
       332.587               RadialShieldUnit0xc3d7da8 lvidx  56 nsp    288             intersection difference cylinder slab   nds[ 64]  4393 4394 4395 4396 4397 4398 4399 4400 4401 4402 ... 
           320                      SstTopHub0xc2643d8 lvidx  68 nsp    317                                    union cylinder   nds[  2]  4464 6124 . 
       28.0747              OcrGdsTfbInLsoOfl0xc2b5ba0 lvidx  83 nsp    243                          difference cylinder cone   nds[  2]  4515 6175 . 
       26.2183                   OcrGdsLsoPrt0xc104978 lvidx  81 nsp    342                    union difference cylinder cone   nds[  2]  4511 6171 . 
            20               headon-pmt-mount0xc2a7670 lvidx  55 nsp    365                         union difference cylinder   nds[ 12]  4357 4364 4371 4378 4385 4392 6017 6024 6031 6038 ... 
            12           near_side_long_hbeam0xbf3b5d0 lvidx  17 nsp    450                                        union box3   nds[  8]  2436 2437 2615 2616 2794 2795 2973 2974 . 
        10.035               led-source-shell0xc3068f0 lvidx 100 nsp    567                            union zsphere cylinder   nds[  6]  4541 4629 4711 6201 6289 6371 . 
        10.035                   weight-shell0xc307920 lvidx 103 nsp    567                            union zsphere cylinder   nds[ 36]  4543 4547 4558 4562 4591 4595 4631 4635 4646 4650 ... 
        10.035        AmCCo60AcrylicContainer0xc0b23b8 lvidx 131 nsp    219                             union sphere cylinder   nds[  6]  4567 4655 4737 6227 6315 6397 . 
        10.035                   source-shell0xc2d62d0 lvidx 111 nsp    567                            union zsphere cylinder   nds[  6]  4552 4640 4722 6212 6300 6382 . 
       10.0198               SstTopCirRibBase0xc264f78 lvidx  69 nsp    242        intersection difference cylinder slab box3   nds[ 16]  4465 4466 4467 4468 4469 4470 4471 4472 6125 6126 ... 
       8.09241                    OcrGdsInLso0xbfa2190 lvidx  31 nsp    287             intersection difference cylinder cone   nds[  2]  3168 4828 . 
       7.54053                   pmt-hemi-vac0xc21e248 lvidx  46 nsp    665                union intersection sphere cylinder   nds[672]  3200 3206 3212 3218 3224 3230 3236 3242 3248 3254 ... 
       5.01849                    source-assy0xc2d5d78 lvidx 112 nsp    480                            union zsphere cylinder   nds[  6]  4551 4639 4721 6211 6299 6381 . 
       5.01749                led-source-assy0xc3061d0 lvidx 105 nsp    480                            union zsphere cylinder   nds[  6]  4540 4628 4710 6200 6288 6370 . 
       5.01749            amcco60-source-assy0xc0b1df8 lvidx 132 nsp    480                            union zsphere cylinder   nds[  6]  4566 4654 4736 6226 6314 6396 . 
             5                      LsoOflTnk0xc17d928 lvidx 140 nsp    315                       union intersection cylinder   nds[  2]  4606 6266 . 
       4.87451                 OcrGdsTfbInLso0xbfa2370 lvidx  30 nsp    464             intersection difference cylinder cone   nds[  2]  3167 4827 . 
         3.882                   OcrCalLsoPrt0xc1076b0 lvidx  85 nsp    351                    union difference cylinder cone   nds[  2]  4517 6177 . 
         1.782                 OcrGdsTfbInOav0xbf8f6c0 lvidx  39 nsp    255             intersection difference cylinder cone   nds[  2]  3196 4856 . 
       1.41823                 OcrCalLsoInOav0xc541388 lvidx  41 nsp    375             intersection difference cylinder cone   nds[  2]  3198 4858 . 
       1.17236                 OcrGdsLsoInOav0xc354118 lvidx  40 nsp    510             intersection difference cylinder cone   nds[  2]  3195 4855 . 
       1.01001                SstTopTshapeRib0xc272c80 lvidx  67 nsp    421                          difference cylinder box3   nds[ 16]  4456 4457 4458 4459 4460 4461 4462 4463 6116 6117 ... 
      0.961575                    OcrGdsInOav0xc355130 lvidx  38 nsp    310             intersection difference cylinder cone   nds[  2]  3197 4857 . 
      0.799805                      near_rock0xc04ba08 lvidx 247 nsp    382                                   difference box3   nds[  1]  1 . 
      0.685471                    OcrGdsInIav0xc405b10 lvidx  23 nsp    294             intersection difference cylinder cone   nds[  2]  3160 4820 . 
           0.5            near_hall_top_dwarf0xc0316c8 lvidx  21 nsp    300                                        union box3   nds[  1]  2 . 
      0.358002                near_span_hbeam0xc2a27d8 lvidx   9 nsp    450                                        union box3   nds[ 18]  2359 2360 2432 2433 2434 2435 2611 2612 2613 2614 ... 
      0.247902                       pmt-hemi0xc0fed90 lvidx  47 nsp    674                union intersection sphere cylinder   nds[672]  3199 3205 3211 3217 3223 3229 3235 3241 3247 3253 ... 
        0.1313                   pmt-hemi-bot0xc22a958 lvidx  44 nsp    381                                difference zsphere   nds[672]  3202 3208 3214 3220 3226 3232 3238 3244 3250 3256 ... 
      0.119995                            oav0xc2ed7c8 lvidx  42 nsp    294                               union cylinder cone   nds[  2]  3156 4816 . 
    2017-07-07 20:54:11.545 INFO  [3968900] [GScene::compareMeshes_GMeshBB@526] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF num_discrepant 33 frac 0.13253
    Assertion failed: (0 && "GScene::init early exit for gltf==4 or gltf==44"), function init, file /Users/blyth/opticks/ggeo/GScene.cc, line 156.


