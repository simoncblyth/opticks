lvidx66 (formerly in 4th) NOW IN POLE POSITION
======================================================

My manual bbox matches the g4poly one, but the parsurf one stops at zero in x, rotated into position 
at Z-top of SST?

* suspect this issue is related to the trapezoid(convexpolyhedron)
  and the "manual" bbox that it needs ... so probably can leave ASIS : the final arbiter is photon intersects

::

        .       ~symm-x     thin-y    long-z
        amn (   -345.000   -10.000 -1114.250) 
        bmn (   -345.510   -10.000 -1114.250) 
        dmn (      0.510     0.000     0.000) 

        amx (  **0.000**    10.000  1114.250)   <-- max.x  
        bmx (    345.510    10.000  1114.250) 
        dmx (   -345.510     0.000     0.000)





parsurf surface points not getting above x=0 ? 
-------------------------------------------------

::

    2017-07-08 15:57:07.412 INFO  [4096976] [NSceneConfig::dump@63] bbox_type_string : CSG_BBOX_PARSURF
     verbosity 4 countdown 7 level 2 target 200 num_surface_points 0 tots (inside/surface/outside/selected)      25     121     304     121
     verbosity 4 countdown 6 level 3 target 200 num_surface_points 121 tots (inside/surface/outside/selected)      81     408     969     408
    2017-07-08 15:57:07.414 INFO  [4096976] [NCSG::dump@911] NCSG::dump
     NCSG  ix    0 surfpoints  408 so -                                        lv -                                       
     bbsp  mi (   -345.000   -10.000 -1114.250) mx (      0.000    10.000  1114.250) si (    345.000    20.000  2228.500)


surface point debugging
---------------------------


::

    delta:ggeo blyth$ VERBOSITY=10 opticks-nnt 66
    opticks-nnt : compiling /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/66/NNodeTest_66.cc
    2017-07-08 16:29:29.929 INFO  [4115210] [NSceneConfig::NSceneConfig@42] NSceneConfig::NSceneConfig cfg []
    2017-07-08 16:29:29.930 INFO  [4115210] [NCSG::collect_surface_points@1186] NCSG::collect_surface_points verbosity 10
                      check_surf_containment : 0
                      check_aabb_containment : 0
                          disable_instancing : 0
                           csg_bbox_analytic : 0
                               csg_bbox_poly : 0
                            csg_bbox_parsurf : 0
                             csg_bbox_g4poly : 0
                              parsurf_target : 200
                               parsurf_level : 2
                              parsurf_margin : 0
    2017-07-08 16:29:29.930 INFO  [4115210] [NSceneConfig::dump@63] bbox_type_string : CSG_BBOX_PARSURF
    nnode::getSurfacePoints verbosity  10 s   0 nu    4 nv    4 ndiv     5 expect     25 n0      0
    nnode::getSurfacePoints verbosity  10 s   1 nu    4 nv    4 ndiv     5 expect     25 n0     25
    nnode::getSurfacePoints verbosity  10 s   2 nu    4 nv    4 ndiv     5 expect     25 n0     50
    nnode::getSurfacePoints verbosity  10 s   3 nu    4 nv    4 ndiv     5 expect     25 n0     75
    nnode::getSurfacePoints verbosity  10 s   4 nu    4 nv    4 ndiv     5 expect     25 n0    100
    nnode::getSurfacePoints verbosity  10 s   5 nu    4 nv    4 ndiv     5 expect     25 n0    125
    nnode::getSurfacePointsAll prim   0 pointmask       POINT_SURFACE  primsurf    150 num_inside     25 num_surface    100 num_outside     25 num_select    100
    nnode::getSurfacePoints verbosity  10 s   0 nu    4 nv    4 ndiv     5 expect     25 n0      0
    nnode::getSurfacePoints verbosity  10 s   1 nu    4 nv    4 ndiv     5 expect     25 n0     25
    nnode::getSurfacePoints verbosity  10 s   2 nu    4 nv    4 ndiv     5 expect     25 n0     50





overview
----------

::

     ## opticks-tbool-vi 66 
     91 # convexpolyhedron are defined by planes and require manual aabbox definition
     92 a.param2[:3] = [-345.510,-10.000,-1114.250]
     93 a.param3[:3] = [345.510,10.000,1114.250]

     ## recall this is rotated into position , the long z-axis in model frame becomes x.... 
     ## so the max.x problem will be visible in z-dir

     ## the parsurf bbox is reflecting the cut at the lid, but g4poly 


::

        345.51                SstTopRadiusRib0xc271720 lvidx  66 nsp    408                  difference box3 convexpolyhedron   nds[ 16]  4448 4449 4450 4451 4452 4453 4454 4455 6108 6109 ... 


::


    op --gltf 44           # dump the compare meshes table

        345.51                SstTopRadiusRib0xc271720 lvidx  66 nsp    408 amn (   -345.000   -10.000 -1114.250) bmn (   -345.510   -10.000 -1114.250) dmn (      0.510     0.000     0.000) amx (      0.000    10.000  1114.250) bmx (    345.510    10.000  1114.250) dmx (   -345.510     0.000     0.000)


::

    op --dlv66 --gltf 1
    op --dlv66 --gltf 3

    ~/opticks_refs/lvidx66_gltf_1_SstTopRadiusRib0xc271720_from_below_loose_edges.png
         non-manifold loose edge pulls the bbox down, also bbox extends above the lid where geometry doesnt go 

    ~/opticks_refs/lvidx66_gltf_3_SstTopRadiusRib0xc271720_from_below_loose_edges.png
        loose edge but bbox not pulled down,  also bbox extends above the lid where geometry doesnt go

        345.51                SstTopRadiusRib0xc271720 lvidx  66 nsp    408 

    op --dlv66 --gltf 3 --gltfconfig disable_instancing=1
        ## ray trace no-show with instancing disables, ahh this was the planes placement thats not fixed

    op --dlv66 --gltf 3 
        ## g4poly bbox sticks right up thru lid, ray trace doesnt 

    op --dlv66 --gltf 1
        ## parsurf bbox also stick thru

    op --dlv66 
        ## g4poly and the tri-raytrace it follows sticks thru


    opticks-tbool 66
         ~/opticks_refs/lvidx66_tbool_bbox_extends_to_px_but_raytrace_does_not.png
         ~/opticks_refs/lvidx66_constituents_big_box_subtracted_from_trapezoid.png


::

    078 # generated by tboolean.py : 20170706-1446 
     79 # opticks-;opticks-tbool 66 
     80 # opticks-;opticks-tbool-vi 66 
     81 
     82 
     83 a = CSG("trapezoid", param = [0.000,0.000,0.000,0.000],param1 = [0.000,0.000,0.000,0.000])
     84 a.planes = np.zeros( (6,4), dtype=np.float32)
     85 a.planes[0] = [0.993,0.000,-0.118,211.261]    # ~+X      \     /
     86 a.planes[1] = [-0.993,0.000,-0.118,211.261]   # ~-X    .  \   /  .   
     87 a.planes[2] = [0.000,1.000,-0.000,10.000]     # +Y
     88 a.planes[3] = [0.000,-1.000,0.000,10.000]     # -Y 
     89 a.planes[4] = [0.000,-0.000,1.000,1114.250]   # +Z  : coincident b.z +2228.5/2 
     90 a.planes[5] = [0.000,-0.000,-1.000,1114.250]  # -Z  : coincident b.z -2228.5/2  
     91 # convexpolyhedron are defined by planes and require manual aabbox definition
     92 a.param2[:3] = [-345.510,-10.000,-1114.250]
     93 a.param3[:3] = [345.510,10.000,1114.250]
     94 
     95 b = CSG("box3", param = [40.000,40.000,2228.500,0.000],param1 = [0.000,0.000,0.000,0.000])
     96 b.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[-340.000,0.000,0.000,1.000]]
     97 ab = CSG("difference", left=a, right=b)
     98 
     99 c = CSG("box3", param = [691.020,24.000,2238.500,0.000],param1 = [0.000,0.000,0.000,0.000])
    100 c.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[345.510,0.000,0.000,1.000]]
    101 abc = CSG("difference", left=ab, right=c)

    /// parallel boxes same z range : 
    ///
    ///  action in x:  one thin, on thick, separated 
    ///        ... both intersect the trapezoid in between them : one by a sliver, other substantially 
    ///   
    ///      
    ///  b.x   -20,20                       ->  -360,-320 
    ///  c.x   -691.02/2.+345.51,+691.02/2.+345.51          -> (0.0, 691.02)





    102 
    103 
    104 
    105 obj = abc
    106 
    107 con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=args.container , poly="IM", resolution="20" )
    108 CSG.Serialize([con, obj], args.csgpath )





G4DAE mesh dump (g4poly)::

    simon:opticksnpy blyth$ op --dlv66 --gmeshlib --dbgmesh SstTopRadiusRib0xc271720
    === op-cmdline-binary-match : finds 1st argument with associated binary : --gmeshlib
    240 -rwxr-xr-x  1 blyth  staff  120332 Jul  6 18:08 /usr/local/opticks/lib/GMeshLibTest
    proceeding : /usr/local/opticks/lib/GMeshLibTest --dlv66 --gmeshlib --dbgmesh SstTopRadiusRib0xc271720
    2017-07-06 18:45:40.636 INFO  [3709302] [OpticksQuery::dumpQuery@81] OpticksQuery::init queryType range query_string range:3155:3156,range:4448:4449 query_name NULL query_index 0 nrange 4 : 3155 : 3156 : 4448 : 4449
    2017-07-06 18:45:40.637 INFO  [3709302] [Opticks::init@319] Opticks::init DONE OpticksResource::desc digest 48ce6eae7a859d5555e1e21c4bee206e age.tot_seconds 271425 age.tot_minutes 4523.750 age.tot_hours 75.396 age.tot_days      3.141
    2017-07-06 18:45:40.653 INFO  [3709302] [GMeshLib::loadMeshes@206] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.48ce6eae7a859d5555e1e21c4bee206e.dae
    2017-07-06 18:45:40.693 INFO  [3709302] [GMesh::dump@1139] GMesh::dump num_vertices 14 num_faces 24 num_solids 0 name SstTopRadiusRib0xc271720
     low  -
     high -
     dim  -
     cen  - extent 0
     ce   (     0.000      0.000      0.000   1114.250)
     bb.max   (   345.510     10.000   1114.250)
     bb.min   (  -345.510    -10.000  -1114.250)
        0 vtx (   -80.000    -10.000  -1114.250) nrm (     0.000      0.000     -1.000)
        1 vtx (   -80.000     10.000  -1114.250) nrm (     0.000      0.000     -1.000)
        2 vtx (    80.000     10.000  -1114.250) nrm (     0.000      0.000     -1.000)
        3 vtx (    80.000    -10.000  -1114.250) nrm (     0.000      0.000     -1.000)
        4 vtx (  *345.510*    10.000   1114.250) nrm (     0.993      0.000     -0.118)
        5 vtx (  *345.510*   -10.000   1114.250) nrm (     0.993      0.000     -0.118)
        6 vtx (  -345.510    -10.000   1114.250) nrm (     0.000     -0.000      1.000)
        7 vtx (  -345.510     10.000   1114.250) nrm (     0.000     -0.000      1.000)
        8 vtx (  -345.502     10.000   1114.180) nrm (    -0.993      0.000     -0.118)
        9 vtx (  -345.502    -10.000   1114.180) nrm (    -0.993      0.000     -0.118)
       10 vtx (  -319.949    -10.000    899.707) nrm (    -0.993     -0.000     -0.118)
       11 vtx (  -319.949     10.000    899.707) nrm (    -0.993     -0.000     -0.118)
       12 vtx (  -319.949     10.000   1114.180) nrm (    -0.000      1.000     -0.000)
       13 vtx (  -319.949    -10.000   1114.180) nrm (     0.000     -1.000      0.000)

    2017-07-06 18:45:40.693 INFO  [3709302] [GMesh::dump@1191]  num_faces 24
     fac     0      0     1     2 
     fac     1      0     2     3 
     fac     2     *4     5     3* 
     fac     3     *4     3     2* 
     fac     4     *6     5     4* 
     fac     5     *6     4     7* 
     fac     6      8     9     6 
     fac     7      6     7     8 
     fac     8     10    11     1 
     fac     9      1     0    10 
     fac    10     12     8     7 
     fac    11     *4     2     1* 
     fac    12      1    11    12 
     fac    13    *12     7     4* 
     fac    14      4     1    12 
     fac    15     13    10     0 
     fac    16      0     3     5 
     fac    17      5     6     9 
     fac    18     13     0     5 
     fac    19      5     9    13 
     fac    20     13    12    11 
     fac    21     11    10    13 
     fac    22      9     8    12 
     fac    23     12    13     9 
    /Users/blyth/opticks/bin/op.sh RC 0




::

    simon:opticks_refs blyth$ DBGNODE=4448 DBGMESH=SstTopRadiusRib0xc271720  NSceneMeshTest
    2017-07-06 18:53:06.772 INFO  [3710868] [main@29] NSceneMeshTest gltfbase /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300 gltfname g4_00.gltf gltfconfig check_surf_containment=0,check_aabb_containment=0
    2017-07-06 18:53:06.772 INFO  [3710868] [NGLTF::load@35] NGLTF::load path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf
    2017-07-06 18:53:07.285 INFO  [3710868] [NGLTF::load@62] NGLTF::load DONE
    2017-07-06 18:53:07.310 INFO  [3710868] [NSceneConfig::NSceneConfig@42] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0]
            check_surf_containment :                    0
            check_aabb_containment :                    0
    2017-07-06 18:53:07.310 INFO  [3710868] [NScene::init@177] NScene::init START age(s) 14803 days   0.171
    2017-07-06 18:53:07.310 INFO  [3710868] [NScene::load_csg_metadata@297] NScene::load_csg_metadata verbosity 1 num_meshes 249
    2017-07-06 18:53:07.724 INFO  [3710868] [NScene::postimportnd@543] NScene::postimportnd numNd 12230 num_selected 12230 dbgnode 4448 dbgnode_list 1 verbosity 1
    2017-07-06 18:53:07.894 INFO  [3710868] [NScene::count_progeny_digests@917] NScene::count_progeny_digests verbosity 1 node_count 12230 digest_size 249
    2017-07-06 18:53:10.130 INFO  [3710868] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-06 18:53:13.665 INFO  [3710868] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-06 18:53:13.713 INFO  [3710868] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-06 18:53:15.393 INFO  [3710868] [NScene::postimportmesh@561] NScene::postimportmesh numNd 12230 dbgnode 4448 dbgnode_list 1 verbosity 1
                      check_surf_containment : 0
                      check_aabb_containment : 0
                          disable_instancing : 0
                           csg_bbox_analytic : 0
                               csg_bbox_poly : 0
                            csg_bbox_parsurf : 0
                             csg_bbox_g4poly : 0
                              parsurf_target : 200
                               parsurf_level : 2
                              parsurf_margin : 0
    2017-07-06 18:53:15.393 INFO  [3710868] [NSceneConfig::dump@63] bbox_type_string : CSG_BBOX_PARSURF
    2017-07-06 18:53:15.394 INFO  [3710868] [NScene::init@225] NScene::init DONE
    2017-07-06 18:53:15.394 INFO  [3710868] [NScene::dumpCSG@457] NScene::dumpCSG num_csg 249 dbgmesh SstTopRadiusRib0xc271720
    2017-07-06 18:53:15.394 INFO  [3710868] [NCSG::dump@910] NCSG::dump
     NCSG  ix   77 surfpoints  408 so SstTopRadiusRib0xc271720                 lv /dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0
     bbsp  mi (   -345.000   -10.000 -1114.250) mx (      0.000    10.000  1114.250) si (    345.000    20.000  2228.500)
    2017-07-06 18:53:15.394 INFO  [3710868] [nnode::dump@894] NCSG::dump
     du [ 0:di]    OPER  v:0  bb  mi (   -345.510   -10.000 -1114.250) mx (    345.510    10.000  1114.250) si (    691.020    20.000  2228.500)

     du [ 1:di]    OPER  v:0  bb  mi (   -345.510   -10.000 -1114.250) mx (    345.510    10.000  1114.250) si (    691.020    20.000  2228.500)

     du [ 3:co]    PRIM  v:0  bb  mi (   -345.510   -10.000 -1114.250) mx (    345.510    10.000  1114.250) si (    691.020    20.000  2228.500)
     gt [ 3:co]         gt.t
                1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   0.000   1.000 

     du [ 4:bo]    PRIM  v:0  bb  mi (   -360.000   -20.000 -1114.250) mx (   -320.000    20.000  1114.250) si (     40.000    40.000  2228.500)
     gt [ 4:bo]         gt.t
                1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -340.000   0.000   0.000   1.000 

     gt [ 1:di]    NO gtransform 
     gt [ 3:co]         gt.t
                1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   0.000   1.000 

     gt [ 4:bo]         gt.t
                1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -340.000   0.000   0.000   1.000 

     du [ 2:bo]    PRIM  v:0  bb  mi (      0.000   -12.000 -1119.250) mx (    691.020    12.000  1119.250) si (    691.020    24.000  2238.500)
     gt [ 2:bo]         gt.t
                1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
              345.510   0.000   0.000   1.000 

     gt [ 0:di]    NO gtransform 
     gt [ 1:di]    NO gtransform 
     gt [ 3:co]         gt.t
                1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   0.000   1.000 

     gt [ 4:bo]         gt.t
                1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -340.000   0.000   0.000   1.000 

     gt [ 2:bo]         gt.t
                1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
              345.510   0.000   0.000   1.000 

    NParameters::dump
             lvname : /dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0
             soname : SstTopRadiusRib0xc271720
          verbosity :               0
         resolution :              20
               poly :              IM
             height :               2
    2017-07-06 18:53:15.394 INFO  [3710868] [NCSG::dump_surface_points@1253] dsp num_sp 408 dmax 200
     bbsp  mi (   -345.000   -10.000 -1114.250) mx (      0.000    10.000  1114.250) si (    345.000    20.000  2228.500)
     i    0 sp (      0.000    10.000     0.000)
     i   81 sp (      0.000   -10.000     0.000)
     i  162 sp (      0.000     0.000  1114.250)
     nds[ 16]  4448 4449 4450 4451 4452 4453 4454 4455 6108 6109 6110 6111 6112 6113 6114 6115 . 



