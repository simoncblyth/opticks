
FIXED : lvid 66 : unbelievable big impingement  : trapezoid/convexpolyhedron machinery bug
==============================================================================================

Issue *FIXED*
---------------

Parametric surface points from nodes using lvid 66 are impinging 1025mm into parent volume...

* parametric bug ?
* bbox bug ?

* looking at visualization suggests deliberate impingement ... it looks 
  too symmetrically positioned to be a bug  

    * BUT visualizations are very different between branches ?

* *FIXED* : was misinterpreting GDML trapezoid z by factor of 2 , found 
  by comparison of mesh dumps from the G4DAE and GDML/glTF branches


After fix : surface coincidence
---------------------------------

::

    NSceneLoadTest 

    NSc::csp n  4446 nlv  65 p  3155 n.pv lvOIL#pvSstBotCirRib#SstBotCir pp(nn.local) - nsdf: EE    33(in:/su/ou/er)  27   6   0   6   -430.000    -0.000 ep 1.000000e-03 [-4.300000e+02,-0.000000e+00] 
    NSc::csp n  4447 nlv  65 p  3155 n.pv lvOIL#pvSstBotCirRib#SstBotCir pp(nn.local) - nsdf: EE    33(in:/su/ou/er)  27   6   0   6   -430.000    -0.000 ep 1.000000e-03 [-4.300000e+02,-0.000000e+00] 
    NSc::csp n  4448 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBT pp(nn.local) - nsdf: EE   121(in:/su/ou/er)  12 109   0 109   -340.000     0.000 ep 1.000000e-03 [-3.400000e+02,0.000000e+00] 
    NSc::csp n  4449 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBT pp(nn.local) - nsdf: EE   121(in:/su/ou/er)  12 109   0 109   -340.000     0.000 ep 1.000000e-03 [-3.400000e+02,0.000000e+00] 
    NSc::csp n  4450 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBT pp(nn.local) - nsdf: EE   121(in:/su/ou/er)  12 109   0 109   -340.000     0.000 ep 1.000000e-03 [-3.400000e+02,0.000000e+00] 
    NSc::csp n  4451 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBT pp(nn.local) - nsdf: EE   121(in:/su/ou/er)  12 109   0 109   -340.000     0.000 ep 1.000000e-03 [-3.400000e+02,0.000000e+00] 
    NSc::csp n  4452 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBT pp(nn.local) - nsdf: EE   121(in:/su/ou/er)  12 109   0 109   -340.000     0.000 ep 1.000000e-03 [-3.400000e+02,0.000000e+00] 
    NSc::csp n  4453 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBT pp(nn.local) - nsdf: EE   121(in:/su/ou/er)  12 109   0 109   -340.000     0.000 ep 1.000000e-03 [-3.400000e+02,0.000000e+00] 
    NSc::csp n  4454 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBT pp(nn.local) - nsdf: EE   121(in:/su/ou/er)  12 109   0 109   -340.000     0.000 ep 1.000000e-03 [-3.400000e+02,0.000000e+00] 
    NSc::csp n  4455 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBT pp(nn.local) - nsdf: EE   121(in:/su/ou/er)  12 109   0 109   -340.000     0.000 ep 1.000000e-03 [-3.400000e+02,0.000000e+00] 
    NSc::csp n  4456 nlv  67 p  3155 n.pv lvOIL#pvSstTopTshapeRibs#SstTo pp(nn.local) - nsdf:      129(in:/su/ou/er) 129   0   0   0   -339.421   -80.579 ep 1.000000e-03 [-3.394209e+02,-8.057910e+01] 
    NSc::csp n  4457 nlv  67 p  3155 n.pv lvOIL#pvSstTopTshapeRibs#SstTo pp(nn.local) - nsdf:      129(in:/su/ou/er) 129   0   0   0   -339.421   -80.579 ep 1.000000e-03 [-3.394209e+02,-8.057910e+01] 


Before fix : huge (1025mm) impingement
----------------------------------------

::

    NSc::csp n  4445 nlv  65 p  3155 n.pv lvOIL#pvSstBotCirRib#SstBotCirpp(nn.local) - nsdf: EE    33(in:/su/ou/er)  27   6   0   6   -430.000    -0.000 ep 1.000000e-03 [-4.300000e+02,-0.000000e+00] 
    NSc::csp n  4446 nlv  65 p  3155 n.pv lvOIL#pvSstBotCirRib#SstBotCirpp(nn.local) - nsdf: EE    33(in:/su/ou/er)  27   6   0   6   -430.000    -0.000 ep 1.000000e-03 [-4.300000e+02,-0.000000e+00] 
    NSc::csp n  4447 nlv  65 p  3155 n.pv lvOIL#pvSstBotCirRib#SstBotCirpp(nn.local) - nsdf: EE    33(in:/su/ou/er)  27   6   0   6   -430.000    -0.000 ep 1.000000e-03 [-4.300000e+02,-0.000000e+00] 
    NSc::csp n  *4448* nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp(nn.local) - nsdf: EE   124(in:/su/ou/er)   0  96  28 124      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  4449 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp(nn.local) - nsdf: EE   124(in:/su/ou/er)   0  96  28 124      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  4450 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp(nn.local) - nsdf: EE   124(in:/su/ou/er)   0  96  28 124      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  4451 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp(nn.local) - nsdf: EE   124(in:/su/ou/er)   0  96  28 124      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  4452 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp(nn.local) - nsdf: EE   124(in:/su/ou/er)   0  96  28 124      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  4453 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp(nn.local) - nsdf: EE   124(in:/su/ou/er)   0  96  28 124      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  4454 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp(nn.local) - nsdf: EE   124(in:/su/ou/er)   0  96  28 124      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  4455 nlv  66 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp(nn.local) - nsdf: EE   124(in:/su/ou/er)   0  96  28 124      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 

    NSc::csp n  4464 nlv  68 p  3155 n.pv     lvOIL#pvSstTopHub0xc2476b8pp(nn.local) - nsdf: EE   100(in:/su/ou/er)  75  25   0  25   -340.000     0.000 ep 1.000000e-03 [-3.400000e+02,0.000000e+00] 
    NSc::csp n  4465 nlv  69 p  3155 n.pv lvOIL#pvSstTopCirRib#SstTopCirpp(nn.local) - nsdf: EE    31(in:/su/ou/er)  26   5   0   5   -231.890     0.000 ep 1.000000e-03 [-2.318901e+02,0.000000e+00] 
    NSc::csp n  4466 nlv  69 p  3155 n.pv lvOIL#pvSstTopCirRib#SstTopCirpp(nn.local) - nsdf: EE    31(in:/su/ou/er)  26   5   0   5   -231.890     0.000 ep 1.000000e-03 [-2.318901e+02,0.000000e+00] 
    NSc::csp n  4467 nlv  69 p  3155 n.pv lvOIL#pvSstTopCirRib#SstTopCirpp(nn.local) - nsdf: EE    31(in:/su/ou/er)  26   5   0   5   -231.890     0.000 ep 1.000000e-03 [-2.318901e+02,0.000000e+00] 
    NSc::csp n  4468 nlv  69 p  3155 n.pv lvOIL#pvSstTopCirRib#SstTopCirpp(nn.local) - nsdf: EE    31(in:/su/ou/er)  26   5   0   5   -231.890     0.000 ep 1.000000e-03 [-2.318901e+02,0.000000e+00] 



::

    2017-07-01 15:23:27.915 INFO  [2338737] [NScene::postimportmesh@429] NScene::postimportmesh numNd 12230 dbgnode 4448 dbgnode_list 1 verbosity 1
    2017-07-01 15:23:27.915 INFO  [2338737] [NScene::check_surf_containment@506] NScene::check_surf_containment (csc) verbosity 1
    2017-07-01 15:23:27.915 INFO  [2338737] [NScene::debug_node@674] NScene::debug_node  n  4448 n.mesh    77 n.lv  66 p.lv  90 p  3155 n.pv lvOIL#pvSstTopRadiusRibs#SstBT
    2017-07-01 15:23:27.916 INFO  [2338737] [nd::dump_transforms@84] n->dump_transforms num_anc 11



Running with just 2 nodes of the geometry
---------------------------------------------

GDML/glTF branch
~~~~~~~~~~~~~~~~~~~~

::

    103 tgltf-t() { TGLTFPATH=$($FUNCNAME- 2>/dev/null) tgltf-- $* ; }
    104 tgltf-t-()
    105 {   
    106     op-  # needs OPTICKS_QUERY envvar 
    109     export OPTICKS_QUERY="range:3155:3156,range:4448:4449"
    110     local gltfpath=$TMP/$FUNCNAME/sc.gltf
    111     gdml2gltf.py --gltfpath $gltfpath
    112     echo $gltfpath
    113 }


G4DAE branch
~~~~~~~~~~~~~~~~~

Visualize 2 nodes in G4DAE branch::

    # select 4448 and 3155
    export OPTICKS_QUERY="range:3155:3156,range:4448:4449"

    op --dsst
    op --dsst -G   ## dont use preexisting geocache : so can dump from full GSolid tree



GDML/glTF CSG dumping : before z/2 trapezoid fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:tests blyth$ DBGMESH=SstTopRadiusRib0x NSceneMeshTest 
    2017-07-02 15:36:15.554 INFO  [2656571] [NGLTF::load@35] NGLTF::load path /tmp/blyth/opticks/tgltf-t/sc.gltf
    2017-07-02 15:36:16.064 INFO  [2656571] [NGLTF::load@62] NGLTF::load DONE
    2017-07-02 15:36:16.089 INFO  [2656571] [NScene::init@126] NScene::init START
    2017-07-02 15:36:16.089 INFO  [2656571] [NScene::load_csg_metadata@235] NScene::load_csg_metadata verbosity 1 num_meshes 249
    2017-07-02 15:36:16.507 INFO  [2656571] [NScene::postimportnd@474] NScene::postimportnd numNd 12230 num_selected 2 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-07-02 15:36:16.676 INFO  [2656571] [NScene::count_progeny_digests@882] NScene::count_progeny_digests verbosity 1 node_count 12230 digest_size 249
    2017-07-02 15:36:17.918 INFO  [2656571] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-02 15:36:18.317 INFO  [2656571] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-02 15:36:18.358 INFO  [2656571] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-02 15:36:19.943 INFO  [2656571] [NScene::postimportmesh@492] NScene::postimportmesh numNd 12230 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-07-02 15:36:19.943 INFO  [2656571] [NScene::check_surf_containment@569] NScene::check_surf_containment (csc) verbosity 1
    2017-07-02 15:36:19.945 INFO  [2656571] [NScene::check_surf_containment@577] NScene::check_surf_containment (csc) verbosity 1 tot 12230 surferr       0       0       0       0
    2017-07-02 15:36:19.945 INFO  [2656571] [NScene::init@163] NScene::init DONE
    2017-07-02 15:36:19.945 INFO  [2656571] [NScene::dumpCSG@395] NScene::dumpCSG num_csg 249 dbgmesh SstTopRadiusRib0x
    2017-07-02 15:36:19.946 INFO  [2656571] [NCSG::dump@905] NCSG::dump
     NCSG  ix   77 surfpoints  124 so SstTopRadiusRib0xc271720                 lv /dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0
    NCSG::dump (root) [ 0:di] OPER  v:0
             L [ 1:di] OPER  v:0
             L [ 3:co] PRIM  v:0 bb  mi  (-345.51  -10.00 -2228.50)  mx  ( 345.51   10.00 2228.50)  si  ( 691.02   20.00 4457.00) 
             R [ 4:bo] PRIM  v:0 bb  mi  (-360.00  -20.00 -1114.25)  mx  (-320.00   20.00 1114.25)  si  (  40.00   40.00 2228.50) 
             R [ 2:bo] PRIM  v:0 bb  mi  (   0.00  -12.00 -1119.25)  mx  ( 691.02   12.00 1119.25)  si  ( 691.02   24.00 2238.50) 
     composite_bb  mi  (-345.51  -10.00 -2228.50)  mx  ( 345.51   10.00 2228.50)  si  ( 691.02   20.00 4457.00) 
    NParameters::dump
             lvname : /dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0
             soname : SstTopRadiusRib0xc271720
          verbosity :               0
         resolution :              20
               poly :              IM
             height :               2
    2017-07-02 15:36:19.946 INFO  [2656571] [NCSG::dump_surface_points@1195] dsp num_sp 124 dmax 200
     i    0 sp (      0.000    10.000     0.000)
     i   25 sp (      0.000   -10.000     0.000)
     i   50 sp (      0.000     0.000  2228.500)
     i   75 sp (      0.000     0.000 -2228.500)
     i  100 sp (      0.000    -6.000 -1119.250)
     i  101 sp (      0.000     0.000 -1119.250)
     i  102 sp (      0.000     6.000 -1119.250)
     i  103 sp (      0.000    -6.000  1119.250)
     i  104 sp (    172.755    -6.000  1119.250)
     i  105 sp (      0.000     0.000  1119.250)
     i  106 sp (    172.755     0.000  1119.250)
     i  107 sp (      0.000     6.000  1119.250)
     i  108 sp (    172.755     6.000  1119.250)
     i  109 sp (      0.000     6.000 -1119.250)
     i  110 sp (      0.000     0.000 -1119.250)
     i  111 sp (      0.000    -6.000 -1119.250)
     i  112 sp (      0.000     6.000  -559.625)
     i  113 sp (      0.000     0.000  -559.625)
     i  114 sp (      0.000    -6.000  -559.625)
     i  115 sp (      0.000     6.000     0.000)
     i  116 sp (      0.000     0.000     0.000)


GDML/glTF dump after trapezoid z/2 fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:analytic blyth$ DBGMESH=SstTopRadiusRib0x NSceneMeshTest 
    2017-07-02 15:51:06.974 INFO  [2660660] [NGLTF::load@35] NGLTF::load path /tmp/blyth/opticks/tgltf-t-/sc.gltf
    2017-07-02 15:51:07.484 INFO  [2660660] [NGLTF::load@62] NGLTF::load DONE
    2017-07-02 15:51:07.511 INFO  [2660660] [NScene::init@126] NScene::init START
    2017-07-02 15:51:07.511 INFO  [2660660] [NScene::load_csg_metadata@235] NScene::load_csg_metadata verbosity 1 num_meshes 249
    2017-07-02 15:51:07.928 INFO  [2660660] [NScene::postimportnd@474] NScene::postimportnd numNd 12230 num_selected 2 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-07-02 15:51:08.099 INFO  [2660660] [NScene::count_progeny_digests@882] NScene::count_progeny_digests verbosity 1 node_count 12230 digest_size 249
    2017-07-02 15:51:09.342 INFO  [2660660] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-02 15:51:09.740 INFO  [2660660] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-02 15:51:09.782 INFO  [2660660] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-07-02 15:51:11.369 INFO  [2660660] [NScene::postimportmesh@492] NScene::postimportmesh numNd 12230 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-07-02 15:51:11.369 INFO  [2660660] [NScene::check_surf_containment@569] NScene::check_surf_containment (csc) verbosity 1
    2017-07-02 15:51:11.371 INFO  [2660660] [NScene::check_surf_containment@577] NScene::check_surf_containment (csc) verbosity 1 tot 12230 surferr       0       0       0       0
    2017-07-02 15:51:11.371 INFO  [2660660] [NScene::init@163] NScene::init DONE
    2017-07-02 15:51:11.371 INFO  [2660660] [NScene::dumpCSG@395] NScene::dumpCSG num_csg 249 dbgmesh SstTopRadiusRib0x
    2017-07-02 15:51:11.372 INFO  [2660660] [NCSG::dump@905] NCSG::dump
     NCSG  ix   77 surfpoints  121 so SstTopRadiusRib0xc271720                 lv /dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0
    NCSG::dump (root) [ 0:di] OPER  v:0
             L [ 1:di] OPER  v:0
             L [ 3:co] PRIM  v:0 bb  mi  (-345.51  -10.00 -1114.25)  mx  ( 345.51   10.00 1114.25)  si  ( 691.02   20.00 2228.50) 
             R [ 4:bo] PRIM  v:0 bb  mi  (-360.00  -20.00 -1114.25)  mx  (-320.00   20.00 1114.25)  si  (  40.00   40.00 2228.50) 
             R [ 2:bo] PRIM  v:0 bb  mi  (   0.00  -12.00 -1119.25)  mx  ( 691.02   12.00 1119.25)  si  ( 691.02   24.00 2238.50) 
     composite_bb  mi  (-345.51  -10.00 -1114.25)  mx  ( 345.51   10.00 1114.25)  si  ( 691.02   20.00 2228.50) 
    NParameters::dump
             lvname : /dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0
             soname : SstTopRadiusRib0xc271720
          verbosity :               0
         resolution :              20
               poly :              IM
             height :               2
    2017-07-02 15:51:11.372 INFO  [2660660] [NCSG::dump_surface_points@1195] dsp num_sp 121 dmax 200
     i    0 sp (      0.000    10.000     0.000)
     i   25 sp (      0.000   -10.000     0.000)
     i   50 sp (      0.000     0.000  1114.250)
     i   75 sp (      0.000     0.000 -1114.250)
     i  100 sp (   -340.000   -10.000  1114.250)
     i  101 sp (   -330.000   -10.000  1114.250)
     i  102 sp (   -320.000   -10.000  1114.250)
     i  103 sp (   -340.000     0.000  1114.250)
     i  104 sp (   -330.000     0.000  1114.250)
     i  105 sp (   -320.000     0.000  1114.250)
     i  106 sp (   -340.000    10.000  1114.250)
     i  107 sp (   -330.000    10.000  1114.250)
     i  108 sp (   -320.000    10.000  1114.250)
     i  109 sp (   -320.000   -10.000  1114.250)
     i  110 sp (   -320.000     0.000  1114.250)
     i  111 sp (   -320.000    10.000  1114.250)
     i  112 sp (      0.000     6.000  -559.625)
     i  113 sp (      0.000     0.000  -559.625)
     i  114 sp (      0.000    -6.000  -559.625)
     i  115 sp (      0.000     6.000     0.000)
     i  116 sp (      0.000     0.000     0.000)
     i  117 sp (      0.000    -6.000     0.000)
     i  118 sp (      0.000     6.000   559.625)
     i  119 sp (      0.000     0.000   559.625)
     i  120 sp (      0.000    -6.000   559.625)


G4DAE GMesh dumping from G4DAE 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:opticksnpy blyth$ op --dsst --gmeshlib --dbgmesh SstTopRadiusRib0x
    === op-cmdline-binary-match : finds 1st argument with associated binary : --gmeshlib
    240 -rwxr-xr-x  1 blyth  staff  120332 Jul  2 15:24 /usr/local/opticks/lib/GMeshLibTest
    proceeding : /usr/local/opticks/lib/GMeshLibTest --dsst --gmeshlib --dbgmesh SstTopRadiusRib0x
    2017-07-02 15:24:29.815 INFO  [2653035] [GMeshLib::loadMeshes@182] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.4d0ba6665a8a501401e989b108a23ae1.dae
    2017-07-02 15:24:29.847 INFO  [2653035] [GMesh::dump@1119] GMesh::dump num_vertices 14 num_faces 24 num_solids 0 name SstTopRadiusRib0xc271720
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
        4 vtx (   345.510     10.000   1114.250) nrm (     0.993      0.000     -0.118)
        5 vtx (   345.510    -10.000   1114.250) nrm (     0.993      0.000     -0.118)
        6 vtx (  -345.510    -10.000   1114.250) nrm (     0.000     -0.000      1.000)
        7 vtx (  -345.510     10.000   1114.250) nrm (     0.000     -0.000      1.000)
        8 vtx (  -345.502     10.000   1114.180) nrm (    -0.993      0.000     -0.118)
        9 vtx (  -345.502    -10.000   1114.180) nrm (    -0.993      0.000     -0.118)
       10 vtx (  -319.949    -10.000    899.707) nrm (    -0.993     -0.000     -0.118)
       11 vtx (  -319.949     10.000    899.707) nrm (    -0.993     -0.000     -0.118)
       12 vtx (  -319.949     10.000   1114.180) nrm (    -0.000      1.000     -0.000)
       13 vtx (  -319.949    -10.000   1114.180) nrm (     0.000     -1.000      0.000)

    2017-07-02 15:24:29.847 INFO  [2653035] [GMesh::dump@1171]  num_faces 24
     fac     0      0     1     2 
     fac     1      0     2     3 
     fac     2      4     5     3 
     fac     3      4     3     2 
     fac     4      6     5     4 
     fac     5      6     4     7 
     fac     6      8     9     6 
     fac     7      6     7     8 
     fac     8     10    11     1 
     fac     9      1     0    10 
     fac    10     12     8     7 




Inspecting GDML
--------------------


Treebase level::

    In [11]: sc.tree.findnode(4448)
    Out[11]: 
    Node 4448 : dig 082c pig ed09 depth 11 nchild 0  
    pv:PhysVol /dd/Geometry/AD/lvOIL#pvSstTopRadiusRibs#SstBTopRibs#SstTopRadiusRibRot0xc247fa0
     Position mm 1284.75 0.0 2477.5  Rotation deg 0.0 90.0 0.0  
    lv:[66] Volume /dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0 /dd/Materials/StainlessSteel0xc2adc00 SstTopRadiusRib0xc271720
       [242] Subtraction SstTopRadiusRib0xc271720  
         l:[240] Subtraction SstTopRadiusRibBase-ChildForSstTopRadiusRib0xc26ed38  
         l:Trapezoid name:SstTopRadiusRibBase0xc271078 z:2228.5 x1:160.0 y1:20.0 x2:691.02 y2:20.0  
         r:[239] Box SstTopRadiusRibCut00xbf75428 mm rmin 0.0 rmax 0.0  x 40.0 y 40.0 z 2228.5  
         r:[241] Box SstTopRadiusRibCut10xc271190 mm rmin 0.0 rmax 0.0  x 691.02 y 24.0 z 2238.5  
       [17] Material /dd/Materials/StainlessSteel0xc2adc00 solid : Position mm 1284.75 0.0 2477.5  




GDML::


    .            X
                 |
                 |
                 | 
                 +------ Z
                /
               /
              Y 


    .            Z
                 |
                 |
                 | 
           X-----+
                /
               /
              Y 


    .  rotate 90 about Y ....    X -> Z,  Y->Y , Z-> -X 
                 

     6635       <physvol name="/dd/Geometry/AD/lvOIL#pvSstTopRadiusRibs#SstBTopRibs#SstTopRadiusRibRot0xc247fa0">
     6636         <volumeref ref="/dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0"/>
     6637         <position name="/dd/Geometry/AD/lvOIL#pvSstTopRadiusRibs#SstBTopRibs#SstTopRadiusRibRot0xc247fa0_pos" unit="mm" x="1284.75" y="0" z="2477.5"/>
     6638         <rotation name="/dd/Geometry/AD/lvOIL#pvSstTopRadiusRibs#SstBTopRibs#SstTopRadiusRibRot0xc247fa0_rot" unit="deg" x="0" y="90" z="0"/>
     6639       </physvol>

     ##  has both position and rotation, as is very common... 
     ##  scanning the GDML position appears to always preceed the rotation : 
     ##  but it makes no sense to position prior to rotating ?  
     ##  scale does appear but always uniform 1





::

    simon:analytic blyth$ DBGNODE=4448 NSceneLoadTest 

     i 10 a.idx   3155
        a.tr.t -1.000   0.000   0.000   0.000 
               -0.000  -1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   7.500   1.000 

     this.idx   4448
     this.tr.t  0.000   0.000   1.000   0.000 
                0.000   1.000   0.000   0.000 
               -1.000   0.000   0.000   0.000 
              1284.750   0.000 2477.500   1.000 

     local points are model points transformed with transform->t (the placing transform) 
    nn.dump_points
             t  0.000   0.000   1.000   0.000 
                0.000   1.000   0.000   0.000 
               -1.000   0.000   0.000   0.000 
              1284.750   0.000 2477.500   1.000 

     # hmm placement has both a rotation (axis swapping) and a translation
     # using wrong order would mess things up ...
     #
     #
     # lightening bolt shape pointing down in Z ...


     model (      0.000     0.000  2228.500) local (   -943.750     0.000  2477.500)


     model (      0.000    -6.000  1119.250) local (    165.500    -6.000  2477.500)
     model (    172.755    -6.000  1119.250) local (    165.500    -6.000  2650.255)
     model (    172.755     6.000  1119.250) local (    165.500     6.000  2650.255)
     model (    172.755     0.000  1119.250) local (    165.500     0.000  2650.255)
     model (      0.000     0.000  1119.250) local (    165.500     0.000  2477.500)
     model (      0.000     6.000  1119.250) local (    165.500     6.000  2477.500)
     model (      0.000     0.000  1119.250) local (    165.500     0.000  2477.500)
     model (      0.000    -6.000  1119.250) local (    165.500    -6.000  2477.500)


     model (      0.000    10.000     0.000) local (   1284.750    10.000  2477.500)
     model (      0.000     6.000     0.000) local (   1284.750     6.000  2477.500)
     model (      0.000     0.000     0.000) local (   1284.750     0.000  2477.500)
     model (      0.000    -6.000     0.000) local (   1284.750    -6.000  2477.500)
     model (      0.000   -10.000     0.000) local (   1284.750   -10.000  2477.500)


     model (      0.000     6.000   559.625) local (    725.125     6.000  2477.500)
     model (      0.000     0.000   559.625) local (    725.125     0.000  2477.500)
     model (      0.000    -6.000   559.625) local (    725.125    -6.000  2477.500)

     model (      0.000     6.000  -559.625) local (   1844.375     6.000  2477.500)
     model (      0.000     0.000  -559.625) local (   1844.375     0.000  2477.500)
     model (      0.000    -6.000  -559.625) local (   1844.375    -6.000  2477.500)

     model (      0.000     0.000 -1119.250) local (   2404.000     0.000  2477.500)
     model (      0.000     0.000 -1119.250) local (   2404.000     0.000  2477.500)
     model (      0.000    -6.000 -1119.250) local (   2404.000    -6.000  2477.500)
     model (      0.000    -6.000 -1119.250) local (   2404.000    -6.000  2477.500)
     model (      0.000     6.000 -1119.250) local (   2404.000     6.000  2477.500)

     model (      0.000     0.000 -2228.500) local (   3513.250     0.000  2477.500)



DONE : fixed tbool bash/python generation to handle convexpolyhedra such as trapezoid
----------------------------------------------------------------------------------------

* required dumping planes and bbox in param2 and param3


tbool90 : parent node big cylinder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     62 tbool90--(){ cat << EOP
     63 
     64 import logging
     65 import numpy as np
     66 log = logging.getLogger(__name__)
     67 from opticks.ana.base import opticks_main
     68 from opticks.analytic.csg import CSG  
     69 args = opticks_main(csgpath="$TMP/tbool/90")
     70 
     71 CSG.boundary = args.testobject
     72 CSG.kwa = dict(verbosity="0", poly="IM", resolution="20")
     73 #CSG.kwa = dict(verbosity="0", poly="HY", level="5")
     74 
     75 
     76 a = CSG("cylinder", param = [0.000,0.000,0.000,2488.000],param1 = [-2477.500,2477.500,0.000,0.000])
     77 
     78 
     79 obj = a
     80 
     81 con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=args.container , poly="IM", resolution="20" )
     82 CSG.Serialize([con, obj], args.csgpath )
     83 
     84 EOP
     85 }



NScene::debug_node point dumping, parent frame points on surface of cylinder::

    2017-07-01 15:07:10.633 INFO  [2334543] [NScene::debug_node@702] pp.classify(pp.local)
    NSDF::classify i    0 q (   2488.000     0.000 -2477.500) sd   -0.00000 sd(sci) -0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    1 q (     -0.000  2488.000 -2477.500) sd   -0.00000 sd(sci) -0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    2 q (  -2488.000    -0.000 -2477.500) sd   -0.00000 sd(sci) -0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    3 q (      0.000 -2488.000 -2477.500) sd   -0.00000 sd(sci) -0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    4 q (   2488.000     0.000 -2477.500) sd   -0.00000 sd(sci) -0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    5 q (   2488.000     0.000 -1300.688) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    6 q (     -0.000  2488.000 -1300.688) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    7 q (  -2488.000    -0.000 -1300.688) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    8 q (      0.000 -2488.000 -1300.688) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    9 q (   2488.000     0.000 -1300.688) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE

    2017-07-01 15:07:10.634 INFO  [2334543] [NScene::debug_node@707] nn.classify(nn.local)
    NSDF::classify i    0 q (   1284.750    10.000  2477.500) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    1 q (   1284.750    10.000  2477.500) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    2 q (   1284.750    10.000  2477.500) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    3 q (   1284.750    10.000  2477.500) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    4 q (   1284.750    10.000  2477.500) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    5 q (   1284.750    10.000  2477.500) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    6 q (   1284.750    10.000  2477.500) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    7 q (   1284.750    10.000  2477.500) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE
    NSDF::classify i    8 q (   1284.750    10.000  2477.500) sd    0.00000 sd(sci) 0.00000e+00 pt POINT_SURFACE







tbool66 generated
~~~~~~~~~~~~~~~~~~~~~

Its a trapezoid with two box3 subtracted.

* however playing around its apparent that box b doesnt intersect with a (or c)
  so are just getting a - c 


::

    opticks-tbool-vi 66

     62 tbool66--(){ cat << EOP
     63 
     64 import logging
     65 import numpy as np
     66 log = logging.getLogger(__name__)
     67 from opticks.ana.base import opticks_main
     68 from opticks.analytic.csg import CSG  
     69 args = opticks_main(csgpath="$TMP/tbool/66")
     70 
     71 CSG.boundary = args.testobject
     72 CSG.kwa = dict(verbosity="0", poly="IM", resolution="20")
     73 
     74 
     75 a = CSG("trapezoid", param = [0.000,0.000,0.000,0.000],param1 = [0.000,0.000,0.000,0.000])
     76 a.planes = np.zeros( (6,4), dtype=np.float32)
     77 a.planes[0] = [0.998,0.000,-0.059,212.379]
     78 a.planes[1] = [-0.998,0.000,-0.059,212.379]
     79 a.planes[2] = [0.000,1.000,-0.000,10.000]
     80 a.planes[3] = [0.000,-1.000,0.000,10.000]
     81 a.planes[4] = [0.000,-0.000,1.000,2228.500]
     82 a.planes[5] = [0.000,-0.000,-1.000,2228.500]
     83 # convexpolyhedron are defined by planes and require manual aabbox definition
     84 a.param2[:3] = [-345.510,-10.000,-2228.500]
     85 a.param3[:3] = [345.510,10.000,2228.500]
     86 
     87 b = CSG("box3", param = [40.000,40.000,2228.500,0.000],param1 = [0.000,0.000,0.000,0.000])
     88 b.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[-340.000,0.000,0.000,1.000]]
     89 ab = CSG("difference", left=a, right=b)
     90 
     91 c = CSG("box3", param = [691.020,24.000,2238.500,0.000],param1 = [0.000,0.000,0.000,0.000])
     92 c.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[345.510,0.000,0.000,1.000]]
     93 abc = CSG("difference", left=ab, right=c)
     94 
     95 
     96 
     97 obj = abc
     98 
     99 con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=args.container , poly="IM", resolution="20" )
    100 CSG.Serialize([con, obj], args.csgpath )
    101 
    102 EOP
    103 }


     4218     <volume name="/dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0">
     4219       <materialref ref="/dd/Materials/StainlessSteel0xc2adc00"/>
     4220       <solidref ref="SstTopRadiusRib0xc271720"/>
     4221     </volume>



     1042     <subtraction name="SstTopRadiusRib0xc271720">
     1043       <first ref="SstTopRadiusRibBase-ChildForSstTopRadiusRib0xc26ed38"/>
     1044       <second ref="SstTopRadiusRibCut10xc271190"/>
     1045       <position name="SstTopRadiusRib0xc271720_pos" unit="mm" x="345.51" y="0" z="0"/>
     1046     </subtraction>


     1034     <trd lunit="mm" name="SstTopRadiusRibBase0xc271078" x1="160" x2="691.02" y1="20" y2="20" z="2228.5"/>
     1035     <box lunit="mm" name="SstTopRadiusRibCut00xbf75428" x="40" y="40" z="2228.5"/>
     1036     <subtraction name="SstTopRadiusRibBase-ChildForSstTopRadiusRib0xc26ed38">
     1037       <first ref="SstTopRadiusRibBase0xc271078"/>
     1038       <second ref="SstTopRadiusRibCut00xbf75428"/>
     1039       <position name="SstTopRadiusRibBase-ChildForSstTopRadiusRib0xc26ed38_pos" unit="mm" x="-340" y="0" z="0"/>
     1040     </subtraction>





::

     74 
     75 
     76 a = CSG("trapezoid", param = [0.000,0.000,0.000,0.000],param1 = [0.000,0.000,0.000,0.000])
     77 b = CSG("box3", param = [40.000,40.000,2228.500,0.000],param1 = [0.000,0.000,0.000,0.000])
     78 b.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[-340.000,0.000,0.000,1.000]]
     79 ab = CSG("difference", left=a, right=b)
     80 
     81 c = CSG("box3", param = [691.020,24.000,2238.500,0.000],param1 = [0.000,0.000,0.000,0.000])
     82 c.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[345.510,0.000,0.000,1.000]]
     83 abc = CSG("difference", left=ab, right=c)
     84 










::

    simon:issues blyth$ opticks-;opticks-tbool 66
    opticks-tbool : sourcing /tmp/blyth/opticks/tgltf/extras/66/tbool66.bash
    args: 
    [2017-06-30 20:53:33,769] p17880 {/Users/blyth/opticks/analytic/csg.py:392} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/66 
    288 -rwxr-xr-x  1 blyth  staff  143804 Jun 29 13:25 /usr/local/opticks/lib/OKTest
    proceeding : /usr/local/opticks/lib/OKTest --animtimemax 20 --timemax 20 --geocenter --eye 1,0,0 --dbganalytic --test --testconfig analytic=1_csgpath=/tmp/blyth/opticks/tbool/66_name=66_mode=PyCsgInBox --torch --torchconfig type=sphere_photons=10000_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1000.000,1.000_source=0,0,0_target=0,0,1_time=0.1_radius=100_distance=400_zenithazimuth=0,1,0,1_material=GdDopedLS_wavelength=500 --torchdbg --tag 1 --cat tbool --save
    2017-06-30 20:53:34.033 INFO  [2232690] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    2017-06-30 20:53:34.202 INFO  [2232690] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-30 20:53:34.307 INFO  [2232690] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-06-30 20:53:34.386 INFO  [2232690] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-06-30 20:53:34.386 INFO  [2232690] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-06-30 20:53:34.386 INFO  [2232690] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-06-30 20:53:34.387 INFO  [2232690] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-06-30 20:53:34.392 INFO  [2232690] [GGeo::loadAnalyticPmt@772] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    2017-06-30 20:53:34.401 WARN  [2232690] [GGeoTest::init@54] GGeoTest::init booting from m_ggeo 
    2017-06-30 20:53:34.401 WARN  [2232690] [GMaker::init@171] GMaker::init booting from cache
    2017-06-30 20:53:34.401 INFO  [2232690] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-30 20:53:34.515 INFO  [2232690] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-06-30 20:53:34.519 INFO  [2232690] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-06-30 20:53:34.519 INFO  [2232690] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-06-30 20:53:34.519 INFO  [2232690] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-06-30 20:53:34.520 INFO  [2232690] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-06-30 20:53:34.523 INFO  [2232690] [GGeoTest::loadCSG@212] GGeoTest::loadCSG  csgpath /tmp/blyth/opticks/tbool/66 verbosity 0
    2017-06-30 20:53:34.523 INFO  [2232690] [NCSG::Deserialize@984] NCSG::Deserialize VERBOSITY 0 basedir /tmp/blyth/opticks/tbool/66 txtpath /tmp/blyth/opticks/tbool/66/csg.txt nbnd 2
    Assertion failed: (idx < m_num_planes), function import_planes, file /Users/blyth/opticks/opticksnpy/NCSG.cpp, line 764.
    /Users/blyth/opticks/bin/op.sh: line 619: 18110 Abort trap: 6           /usr/local/opticks/lib/OKTest --animtimemax 20 --timemax 20 --geocenter --eye 1,0,0 --dbganalytic --test --testconfig analytic=1_csgpath=/tmp/blyth/opticks/tbool/66_name=66_mode=PyCsgInBox --torch --torchconfig type=sphere_photons=10000_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1000.000,1.000_source=0,0,0_target=0,0,1_time=0.1_radius=100_distance=400_zenithazimuth=0,1,0,1_material=GdDopedLS_wavelength=500 --torchdbg --tag 1 --cat tbool --save
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:issues blyth$ 








::

    2017-07-01 16:52:32.153 INFO  [2377419] [*GScene::createVolumeTree@353] GScene::createVolumeTree DONE num_nodes: 12230
    2017-07-01 16:52:32.153 INFO  [2377419] [GScene::init@141] GScene::init createVolumeTrue selected_count 2
    2017-07-01 16:52:32.205 INFO  [2377419] [GScene::makeMergedMeshAndInstancedBuffers@647] GScene::makeMergedMeshAndInstancedBuffers num_repeats 1 START 
    Assertion failed: (0 && "plane placement not implemented"), function applyPlacementTransform, file /Users/blyth/opticks/ggeo/GParts.cc, line 531.
    Process 47317 stopped
    * thread #1: tid = 0x2446cb, 0x00007fff9672d866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff9672d866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff9672d866:  jae    0x7fff9672d870            ; __pthread_kill + 20
       0x7fff9672d868:  movq   %rax, %rdi
       0x7fff9672d86b:  jmp    0x7fff9672a175            ; cerror_nocancel
       0x7fff9672d870:  retq   
    (lldb) 



