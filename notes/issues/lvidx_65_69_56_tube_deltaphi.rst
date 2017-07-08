
lvidx_65_69_56_tube_deltaphi
===============================

TODO 
-----

* revisit the numbers/viz with deltaphi enabled
* revisit tree balancing : current kludge is to disable balancing for trees that include deltaphi segmenting...
  but that causes poor performance



Intersect with nconvexpolyhedron rather than two slab intersects ?
----------------------------------------------------------------------

Advantages:

* one less level of tree height
* avoids unbound-unbound tree balancing issue
* avoids segmenting not working when viewed from the unbound direction  




Trapezoid is similar
~~~~~~~~~~~~~~~~~~~~~~

* a set of planes and bbox are passed from python

::

     643 class Trapezoid(Primitive):
     644     """
     645     The GDML Trapezoid is formed using 5 dimensions:
     646 
     647     x1: x length at -z
     648     x2: x length at +z
     649     y1: y length at -z
     650     y2: y length at +z
     651     z:  z length
     ...
     741     def as_ncsg(self):
     742         assert self.lunit == 'mm'
     743         cn = CSG("trapezoid", name=self.name)
     744         planes, verts, bbox = make_trapezoid(z=self.z, x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2 )
     745         cn.planes = planes
     746         cn.param2[:3] = bbox[0]
     747         cn.param3[:3] = bbox[1]
     748         return cn


Becoming an nconvexpolyhedron at nnode level::

     703 nnode* NCSG::import_primitive( unsigned idx, OpticksCSG_t typecode )
     704 {
     705     nquad p0 = getQuad(idx, 0);
     ...
     718     switch(typecode)
     719     {
     720        case CSG_SPHERE:   node = new nsphere(make_sphere(p0))           ; break ;
     721        case CSG_ZSPHERE:  node = new nzsphere(make_zsphere(p0,p1,p2))   ; break ;
     722        case CSG_BOX:      node = new nbox(make_box(p0))                 ; break ;
     723        case CSG_BOX3:     node = new nbox(make_box3(p0))                ; break ;
     724        case CSG_SLAB:     node = new nslab(make_slab(p0, p1))           ; break ;
     725        case CSG_PLANE:    node = new nplane(make_plane(p0))             ; break ;
     726        case CSG_CYLINDER: node = new ncylinder(make_cylinder(p0, p1))   ; break ;
     727        case CSG_DISC:     node = new ndisc(make_disc(p0, p1))           ; break ;
     728        case CSG_CONE:     node = new ncone(make_cone(p0))               ; break ;
     729        case CSG_TRAPEZOID:
     730        case CSG_CONVEXPOLYHEDRON:
     731                           node = new nconvexpolyhedron(make_convexpolyhedron(p0,p1,p2,p3))   ; break ;
     732        default:           node = NULL ; break ;
     733     }
     734 





Revisit following tube deltaphi via two slab intersects
----------------------------------------------------------

lvidx 56 : RadialShieldUnit : segmented ring with 6 cylinder cuts : tambourine  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* two slab intersects push tree height above max 7 : and tree balancing runs into slab-slab issue
* :doc:`lvidx56_RadialShieldUnit0xc3d7da8` TODO: implement nsegmentphi to regain raytrace

lvidx 65
~~~~~~~~~

* :doc:`lvidx65`

* unclear cause of bbox diff 
* suspiciously one subtracted box has no effect
* needs nsegmentphi to avoid living unbalanced

lvidx 69
~~~~~~~~~~~

* :doc:`lvidx69`

* forced to use raw to get raytrace to work 
* ~/opticks_refs/lvidx_69_ring_box_cuts_artifact.png

* coincidence artifact at one box cut corner
* fixing coincidence involving box is difficult as not z-nudgeable : need CSG_ZBOX ?


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
* high "CSG" level approach avoided the need to implement segmenting SDF, however 
  perhaps an *nsegmentphi* primitive (phi0,phi1,z,rmax) composed of 5 planes 
  would not be difficult (its just like trapezoid : a special case of nconvexpolyhedron)
  with advantage of avoiding bloating the CSG tree with multiple slab intersects

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


    op --gltf 4

    2017-07-07 20:54:11.485 INFO  [3968900] [GScene::importMeshes@316] GScene::importMeshes DONE num_meshes 249
    2017-07-07 20:54:11.485 INFO  [3968900] [GScene::compareMeshes_GMeshBB@435] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF parsurf_level 2 parsurf_target 200
       332.587               RadialShieldUnit0xc3d7da8 lvidx  56 nsp    288             intersection difference cylinder slab   nds[ 64]  4393 4394 4395 4396 4397 4398 4399 4400 4401 4402 ... 
       377.713               SstBotCirRibBase0xc26e2d0 lvidx  65 nsp    204        intersection difference cylinder slab box3   nds[ 16]  4440 4441 4442 4443 4444 4445 4446 4447 6100 6101 ... 
       10.0198               SstTopCirRibBase0xc264f78 lvidx  69 nsp    242        intersection difference cylinder slab box3   nds[ 16]  4465 4466 4467 4468 4469 4470 4471 4472 6125 6126 ... 

::

    op --gltf 44

    2017-07-08 09:05:25.809 INFO  [3977702] [GScene::compareMeshes_GMeshBB@435] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF parsurf_level 2 parsurf_target 200
       332.587               RadialShieldUnit0xc3d7da8 lvidx  56 nsp    288 amn (   1878.414     0.000  -498.500) bmn (   1607.600     0.000  -498.500) dmn (    270.814     0.000     0.000) amx (   2262.150  1256.783   498.500) bmx (   2262.150  1589.370   498.500) dmx (      0.000  -332.587     0.000)
       377.713               SstBotCirRibBase0xc26e2d0 lvidx  65 nsp    204 amn (   1400.071   390.181  -215.000) bmn (   1407.720    12.467  -215.000) dmn (     -7.649   377.713     0.000) amx (   1961.571  1414.214   215.000) bmx (   1998.360  1404.240   215.000) dmx (    -36.789     9.974     0.000)
       10.0198               SstTopCirRibBase0xc264f78 lvidx  69 nsp    242 amn (    848.528     0.000  -115.945) bmn (    854.653    10.020  -115.945) dmn (     -6.125   -10.020     0.000) amx (   1220.000   862.670   115.945) bmx (   1218.680   854.688   115.945) dmx (      1.320     7.982     0.000)




disabling balancing is not a solution
----------------------------------------

::

    delta:issues blyth$ gdml2gltf.py   
    args: /Users/blyth/opticks/bin/gdml2gltf.py
    [2017-07-08 10:36:34,417] p7179 {/Users/blyth/opticks/analytic/gdml.py:1046} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 
    [2017-07-08 10:36:34,460] p7179 {/Users/blyth/opticks/analytic/gdml.py:1060} INFO - wrapping gdml element  
    [2017-07-08 10:36:35,393] p7179 {/Users/blyth/opticks/analytic/treebase.py:504} INFO - apply_selection OpticksQuery  range [] index 0 depth 0   Node.selected_count 12230 
    [2017-07-08 10:36:35,393] p7179 {/Users/blyth/opticks/analytic/sc.py:357} INFO - add_tree_gdml START maxdepth:0 maxcsgheight:3 nodesCount:    0
    [2017-07-08 10:36:35,394] p7179 {/Users/blyth/opticks/analytic/treebase.py:34} WARNING - returning DummyTopPV placeholder transform
    [2017-07-08 10:36:36,325] p7179 {/Users/blyth/opticks/analytic/gdml.py:460} INFO - as_cylinder doing slab segmenting : name BlackCylinder0xc1762e8 phi0 0.0 phi1 44.6352759021 dist 2263.15 
    [2017-07-08 10:36:36,327] p7179 {/Users/blyth/opticks/analytic/sc.py:315} WARNING - tree is_overheight but marked balance_disabled leaving raw : RadialShieldUnit0xc3d7da8 
    [2017-07-08 10:36:36,352] p7179 {/Users/blyth/opticks/analytic/gdml.py:460} INFO - as_cylinder doing slab segmenting : name SstBotCirRibPri0xc26d4e0 phi0 0.0 phi1 45.0 dist 2001.0 
    [2017-07-08 10:36:36,353] p7179 {/Users/blyth/opticks/analytic/sc.py:315} WARNING - tree is_overheight but marked balance_disabled leaving raw : SstBotCirRibBase0xc26e2d0 
    [2017-07-08 10:36:36,363] p7179 {/Users/blyth/opticks/analytic/gdml.py:460} INFO - as_cylinder doing slab segmenting : name SstTopCirRibPri0xc2648b8 phi0 0.0 phi1 45.0 dist 1221.0 
    [2017-07-08 10:36:36,365] p7179 {/Users/blyth/opticks/analytic/sc.py:315} WARNING - tree is_overheight but marked balance_disabled leaving raw : SstTopCirRibBase0xc264f78 
    [2017-07-08 10:36:38,692] p7179 {/Users/blyth/opticks/analytic/sc.py:360} INFO - add_tree_gdml DONE maxdepth:0 maxcsgheight:3 nodesCount:12230 tlvCount:249  tgNd:                           top Nd ndIdx:  0 soIdx:0 nch:1 par:-1 matrix:[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]   
    [2017-07-08 10:36:38,692] p7179 {/Users/blyth/opticks/analytic/sc.py:393} INFO - saving to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf 
    [2017-07-08 10:36:39,155] p7179 {/Users/blyth/opticks/analytic/sc.py:382} INFO - save_extras /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras  : saved 249 
    [2017-07-08 10:36:39,155] p7179 {/Users/blyth/opticks/analytic/sc.py:386} INFO - write 249 lines to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/csg.txt 
    [2017-07-08 10:36:39,989] p7179 {/Users/blyth/opticks/analytic/sc.py:402} INFO - also saving to /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.pretty.gltf 
    delta:issues blyth$ 





