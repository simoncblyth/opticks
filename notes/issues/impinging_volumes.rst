Impinging Volumes
=====================

Plan
-------

* start by testing each node bbox against its parent bbox 

  * within solid uncoincence done in NCSG::postimport, analogous
    place for volume overlap testing would be NScene/GScene ? 

  * start with NScene::postimport, now in NScene::postimportmesh

  * testing with: tgltf-t 


NEXT
------

* sanity check transforms relative to the G4DAE branch 

  * possibilites : lack of precision in GDML/glTF serialization of transforms
  * i vaguely recall issue with transform serialization precision ?? Check GDML/glTF in this respect

* analyse to see if issues really always are always the same for every lv instance 

  
Insights from RTCD p81
-------------------------

* aa-bbox containment is just a pre-check to avoid expensive object-object testing 

* frame of aabb comparison matters


  * some frames will gave tighter AABB than others
  * one or other local frames will usually be tighter
  * could look for no-overlap in both locals

  * rotations usually make aabb grow

  * comparing in global frame has advantage of only need to compute the positioned bbox
    once, but disadvantage of being much further from origin (than local frame) so 
    more potential precision issues


Float Precision
-----------------

* https://randomascii.wordpress.com/2012/02/13/dont-store-that-in-a-float/
* http://www.tfinley.net/notes/cps104/floating.html
* https://stackoverflow.com/questions/872544/what-range-of-numbers-can-be-represented-in-a-16-32-and-64-bit-ieee-754-syste
* http://www.exploringbinary.com/the-spacing-of-binary-floating-point-numbers/



How to do object-object testing ?
-------------------------------------

Generate Vertices On the Surface of the object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* expand parametric coverage to all primitives in order to generate surface vertices 

  * hmm thats fine for primitives 

  * for composite trees would need to check the parametric vertices of all primitives 
    against the composite root CSG SDF, to see if that primitives verts are actually
    on the surface of the final composite 


===================   =============  ================  =================
primitive              parametric     dec_z1/inc_z2 
===================   =============  ================  ================= 
nbox                    Y              N
ncone                   Y              Y                 kludged parametric endcap/body join
nconvexpolyhedron       N(*)           N                 hmm : defined by planes ? minimally provide single point for each plane
ncylinder               Y              Y                 kludged para 
ndisc                   Y              Y                 kludged para + need flexibility wrt uv steps for different surfs : ie just 1+1 in z for disc
nnode                   -              -
nplane                  -              -
nslab                   -              -
nsphere                 Y              N
nzsphere                Y              Y
===================   =============  ================  ================= 




Self node/sdf check : 8/12230 nodes are outside 1e-3 SDF epsilon band 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:tests blyth$ DBGNODE=3159 NSceneLoadTest 
    2017-06-30 19:59:25.916 INFO  [2206366] [NGLTF::load@35] NGLTF::load path /tmp/blyth/opticks/tgltf-t/sc.gltf
    2017-06-30 19:59:26.428 INFO  [2206366] [NGLTF::load@62] NGLTF::load DONE
    2017-06-30 19:59:26.452 INFO  [2206366] [NScene::init@99] NScene::init START
    2017-06-30 19:59:26.452 INFO  [2206366] [NScene::load_csg_metadata@207] NScene::load_csg_metadata verbosity 1 num_meshes 249
    2017-06-30 19:59:26.867 INFO  [2206366] [NScene::postimportnd@411] NScene::postimportnd numNd 12230 dbgnode 3159 dbgnode_list 1 verbosity 1
    2017-06-30 19:59:28.282 INFO  [2206366] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-06-30 19:59:28.682 INFO  [2206366] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-06-30 19:59:28.724 INFO  [2206366] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-06-30 19:59:30.318 INFO  [2206366] [NScene::postimportmesh@429] NScene::postimportmesh numNd 12230 dbgnode 3159 dbgnode_list 1 verbosity 1
    2017-06-30 19:59:30.318 INFO  [2206366] [NScene::check_surf_containment@506] NScene::check_surf_containment (csc) verbosity 1
    NSc::csp n     1 p     0 n.pv db-rock0xc15d358 
    pp.classify(pp.local) - nsdf:      150(in/su:/ou/er)   0 150   0   0      0.000     0.000 ep 1.000000e-03 [0.000000e+00,0.000000e+00]
    nn.classify(nn.local) - nsdf: EE   114(in/su:/ou/er)   0  75  39  39      0.000     0.062 ep 1.000000e-03 [0.000000e+00,6.250000e-02]
    2017-06-30 19:59:32.072 WARN  [2206366] [NSDF::classify@74]  sd.size ZERO 
    2017-06-30 19:59:32.103 WARN  [2206366] [NSDF::classify@74]  sd.size ZERO 
    2017-06-30 19:59:33.218 WARN  [2206366] [NSDF::classify@74]  sd.size ZERO 
    2017-06-30 19:59:33.249 WARN  [2206366] [NSDF::classify@74]  sd.size ZERO 
    NSc::csp n  9207 p  3150 n.pv lvNearPoolOWS#pvVetoPmtNearOut 
    pp.classify(pp.local) - nsdf:      122(in/su:/ou/er)   0 122   0   0      0.000    -0.000 ep 1.000000e-03 [0.000000e+00,-0.000000e+00]
    nn.classify(nn.local) - nsdf: EE   150(in/su:/ou/er)   0 148   2   2     -0.001     0.001 ep 1.000000e-03 [-9.765625e-04,1.464844e-03]
    NSc::csp n  9241 p  3150 n.pv lvNearPoolOWS#pvVetoPmtNearOut 
    pp.classify(pp.local) - nsdf:      122(in/su:/ou/er)   0 122   0   0      0.000    -0.000 ep 1.000000e-03 [0.000000e+00,-0.000000e+00]
    nn.classify(nn.local) - nsdf: EE   150(in/su:/ou/er)   0 148   2   2     -0.000     0.001 ep 1.000000e-03 [-4.882812e-04,1.464844e-03]
    NSc::csp n  9817 p  3150 n.pv lvNearPoolOWS#pvVetoPmtNearOut 
    pp.classify(pp.local) - nsdf:      122(in/su:/ou/er)   0 122   0   0      0.000    -0.000 ep 1.000000e-03 [0.000000e+00,-0.000000e+00]
    nn.classify(nn.local) - nsdf: EE   150(in/su:/ou/er)   0 147   3   3     -0.000     0.001 ep 1.000000e-03 [-4.882812e-04,1.464844e-03]
    NSc::csp n  9836 p  3150 n.pv lvNearPoolOWS#pvVetoPmtNearOut 
    pp.classify(pp.local) - nsdf:      122(in/su:/ou/er)   0 122   0   0      0.000    -0.000 ep 1.000000e-03 [0.000000e+00,-0.000000e+00]
    nn.classify(nn.local) - nsdf: EE   150(in/su:/ou/er)   0 149   1   1     -0.000     0.001 ep 1.000000e-03 [-4.882812e-04,1.464844e-03]
    NSc::csp n  9885 p  3150 n.pv lvNearPoolOWS#pvVetoPmtNearOut 
    pp.classify(pp.local) - nsdf:      122(in/su:/ou/er)   0 122   0   0      0.000    -0.000 ep 1.000000e-03 [0.000000e+00,-0.000000e+00]
    nn.classify(nn.local) - nsdf: EE   150(in/su:/ou/er)   0 147   3   3     -0.000     0.001 ep 1.000000e-03 [-4.882812e-04,1.464844e-03]
    NSc::csp n 10079 p  3150 n.pv lvNearPoolOWS#pvVetoPmtNearOut 
    pp.classify(pp.local) - nsdf:      122(in/su:/ou/er)   0 122   0   0      0.000    -0.000 ep 1.000000e-03 [0.000000e+00,-0.000000e+00]
    nn.classify(nn.local) - nsdf: EE   150(in/su:/ou/er)   0 145   5   5     -0.001     0.001 ep 1.000000e-03 [-9.765625e-04,1.464844e-03]
    NSc::csp n 10110 p  3150 n.pv lvNearPoolOWS#pvVetoPmtNearOut 
    pp.classify(pp.local) - nsdf:      122(in/su:/ou/er)   0 122   0   0      0.000    -0.000 ep 1.000000e-03 [0.000000e+00,-0.000000e+00]
    nn.classify(nn.local) - nsdf: EE   150(in/su:/ou/er)   0 147   3   3     -0.001     0.001 ep 1.000000e-03 [-9.765625e-04,1.464844e-03]
    2017-06-30 20:00:16.028 INFO  [2206366] [NScene::check_surf_containment@514] NScene::check_surf_containment (csc) verbosity 1 tot 12230 surferr       0       0       0       8
    2017-06-30 20:00:16.029 INFO  [2206366] [NScene::init@135] NScene::init DONE
    simon:tests blyth$ 



prioritization 
~~~~~~~~~~~~~~~~~

* impingements where the materials are the same probably do not matter ...


IavTopHub
~~~~~~~~~~~~

::

      555     <polycone aunit="deg" deltaphi="360" lunit="mm" name="IavTopHub0xc405968" startphi="0">
      556       <zplane rmax="100" rmin="75" z="0"/>
      557       <zplane rmax="100" rmin="75" z="85.5603682281126"/>
      558       <zplane rmax="150" rmin="75" z="85.5603682281126"/>
      559       <zplane rmax="150" rmin="75" z="110.560368228113"/>
      560     </polycone>


    In [6]: so = sc.gdml.find_solids("IavTopHub0x")[0]

    In [7]: so
    Out[7]: [63]             IavTopHub0xc405968  4 z:         [0.0, 85.5603682281126, 110.560368228113] rmax:                     [100.0, 150.0] rmin:              [75.0] 







parent/node impingement test : avoiding precision issue does not resolve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


NScene::check_surf_points::

     624     // cross checking containment of a nodes points inside its parent 
     625     // OR vice versa checking that parents points are outside the child node
     626     // is the raison d'etre of this method
     627     //
     628     // coincidence is a problem, as well as impingement ... but try to 
     629     // see how big the issue is
     630     
     631     pp.classify( nn.local, 1e-3, POINT_INSIDE | POINT_SURFACE );
     632     nn.classify( pp.local, 1e-3, POINT_OUTSIDE | POINT_SURFACE );
     633     err.x = pp.nsdf.tot.w ;
     634     err.y = nn.nsdf.tot.w ;


With above code (ie not treating coincidence as error) see 1836/12230 volumes with impingement



Treating surface zeros as error almost half volumes has issue::

    NSc::csp n 12219 nlv 235 p  3148 n.pv lvNearPoolDead#pvNearADE2DeadLpp(nn.local) - nsdf: EE    75(in:/su/ou/er)  45  30   0  30    -84.000     0.000 ep 1.000000e-03 [-8.400000e+01,0.000000e+00] 
    NSc::csp n 12220 nlv 235 p  3148 n.pv lvNearPoolDead#pvNearADE2DeadLpp(nn.local) - nsdf: EE    75(in:/su/ou/er)  45  30   0  30    -84.000     0.000 ep 1.000000e-03 [-8.400000e+01,0.000000e+00] 
    NSc::csp n 12221 nlv 237 p  3147 n.pv lvNearHallBot#pvNearHallRadSlapp(nn.local) - nsdf: EE   150(in:/su/ou/er) 105  45   0  45   -300.000     0.000 ep 1.000000e-03 [-3.000000e+02,0.000000e+00] 
    NSc::csp n 12223 nlv 239 p  3147 n.pv lvNearHallBot#pvNearHallRadSlapp(nn.local) - nsdf: EE   150(in:/su/ou/er) 105  45   0  45   -300.000     0.000 ep 1.000000e-03 [-3.000000e+02,0.000000e+00] 
    NSc::csp n 12225 nlv 241 p  3147 n.pv lvNearHallBot#pvNearHallRadSlapp(nn.local) - nsdf: EE   150(in:/su/ou/er) 105  45   0  45   -300.000     0.000 ep 1.000000e-03 [-3.000000e+02,0.000000e+00] 
    NSc::csp n 12227 nlv 243 p  3147 n.pv lvNearHallBot#pvNearHallRadSlapp(nn.local) - nsdf: EE   150(in:/su/ou/er) 105  45   0  45   -300.000     0.000 ep 1.000000e-03 [-3.000000e+02,0.000000e+00] 
    NSc::csp n 12229 nlv 245 p  3147 n.pv lvNearHallBot#pvNearHallRadSlapp(nn.local) - nsdf: EE   122(in:/su/ou/er)  49  24  49  73   -150.000   150.000 ep 1.000000e-03 [-1.500000e+02,1.500000e+02] 
    2017-06-30 20:47:46.039 INFO  [2230698] [NScene::check_surf_containment@514] NScene::check_surf_containment (csc) verbosity 1 tot 12230 surferr    5907    4370       0       0
    2017-06-30 20:47:46.039 INFO  [2230698] [NScene::init@135] NScene::init DONE



* notice that problems appear exactly the same for the different instances, so issue comes from lv level 


::


    NSc::csp n 11810 m 225 p  3150 n.pv lvNearPoolOWS#pvNearUnistruts#pp.classify(nn.local) - nsdf: EE   150(in:/su:/ou/er) 117   0  33  33    -39.000     1.000 ep 1.000000e-03 [-3.900000e+01,1.000000e+00] 
    NSc::csp n 11811 m 225 p  3150 n.pv lvNearPoolOWS#pvNearUnistruts#pp.classify(nn.local) - nsdf: EE   150(in:/su:/ou/er) 117   0  33  33    -39.000     1.000 ep 1.000000e-03 [-3.900000e+01,1.000000e+00] 
    NSc::csp n 11812 m 225 p  3150 n.pv lvNearPoolOWS#pvNearUnistruts#pp.classify(nn.local) - nsdf: EE   150(in:/su:/ou/er) 117   0  33  33    -39.000     1.000 ep 1.000000e-03 [-3.900000e+01,1.000000e+00] 
    NSc::csp n 11813 m 225 p  3150 n.pv lvNearPoolOWS#pvNearUnistruts#pp.classify(nn.local) - nsdf: EE   150(in:/su:/ou/er) 117   0  33  33    -39.000     1.000 ep 1.000000e-03 [-3.900000e+01,1.000000e+00] 
    NSc::csp n 12229 m 248 p  3147 n.pv lvNearHallBot#pvNearHallRadSlapp.classify(nn.local) - nsdf: EE   122(in:/su:/ou/er)  49  24  49  49   -150.000   150.000 ep 1.000000e-03 [-1.500000e+02,1.500000e+02] 
    2017-06-30 20:30:02.405 INFO  [2222973] [NScene::check_surf_containment@514] NScene::check_surf_containment (csc) verbosity 1 tot 12230 surferr    1836      83       0       0
    2017-06-30 20:30:02.405 INFO  [2222973] [NScene::init@135] NScene::init DONE


Different instances all with same::

    simon:tests blyth$ DBGNODE=3159 NSceneLoadTest 
    2017-06-30 20:20:26.579 INFO  [2218267] [NGLTF::load@35] NGLTF::load path /tmp/blyth/opticks/tgltf-t/sc.gltf
    2017-06-30 20:20:27.096 INFO  [2218267] [NGLTF::load@62] NGLTF::load DONE
    2017-06-30 20:20:27.121 INFO  [2218267] [NScene::init@99] NScene::init START
    2017-06-30 20:20:27.121 INFO  [2218267] [NScene::load_csg_metadata@207] NScene::load_csg_metadata verbosity 1 num_meshes 249
    2017-06-30 20:20:27.537 INFO  [2218267] [NScene::postimportnd@411] NScene::postimportnd numNd 12230 dbgnode 3159 dbgnode_list 1 verbosity 1
    2017-06-30 20:20:28.966 INFO  [2218267] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-06-30 20:20:29.365 INFO  [2218267] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-06-30 20:20:29.407 INFO  [2218267] [NNodeUncoincide::uncoincide_treewise@340] NNodeUncoincide::uncoincide_tree TRYING root.left UNCOINCIDE_UNCYCO  root union difference cylinder cone  left union cylinder  right cone 
    2017-06-30 20:20:31.009 INFO  [2218267] [NScene::postimportmesh@429] NScene::postimportmesh numNd 12230 dbgnode 3159 dbgnode_list 1 verbosity 1
    2017-06-30 20:20:31.009 INFO  [2218267] [NScene::check_surf_containment@506] NScene::check_surf_containment (csc) verbosity 1
    2017-06-30 20:20:32.728 WARN  [2218267] [NSDF::classify@74]  sd.size ZERO 
    2017-06-30 20:20:32.759 WARN  [2218267] [NSDF::classify@74]  sd.size ZERO 
    NSc::csp n  3201 m  56 p  3200 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3207 m  56 p  3206 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3213 m  56 p  3212 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3219 m  56 p  3218 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3225 m  56 p  3224 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3231 m  56 p  3230 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3237 m  56 p  3236 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3243 m  56 p  3242 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3249 m  56 p  3248 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3255 m  56 p  3254 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3261 m  56 p  3260 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3267 m  56 p  3266 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3273 m  56 p  3272 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3279 m  56 p  3278 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3285 m  56 p  3284 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-04 [-2.923752e+01,3.130188e-01] 


Some crazy big ones::

    NSc::csp n  6007 m  56 p  6006 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp.classify(nn.local) - nsdf: EE   115(in:/su:/ou/er)  60  40  15  15    -29.238     0.313 ep 1.000000e-03 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  6086 m  69 p  6085 n.pv lvTopReflector#pvTopRefGap0xc2pp.classify(nn.local) - nsdf: EE    95(in:/su:/ou/er)  85   0  10  10     -9.995    10.000 ep 1.000000e-03 [-9.995000e+00,1.000000e+01] 
    NSc::csp n  6089 m  72 p  6088 n.pv lvBotReflector#pvBotRefGap0xbfpp.classify(nn.local) - nsdf: EE   155(in:/su:/ou/er) 145   0  10  10     -9.995    10.000 ep 1.000000e-03 [-9.995000e+00,1.000000e+01] 
    NSc::csp n  6108 m  77 p  4815 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp.classify(nn.local) - nsdf: EE   124(in:/su:/ou/er)   0  96  28  28      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  6109 m  77 p  4815 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp.classify(nn.local) - nsdf: EE   124(in:/su:/ou/er)   0  96  28  28      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  6110 m  77 p  4815 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp.classify(nn.local) - nsdf: EE   124(in:/su:/ou/er)   0  96  28  28      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  6111 m  77 p  4815 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp.classify(nn.local) - nsdf: EE   124(in:/su:/ou/er)   0  96  28  28      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  6112 m  77 p  4815 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp.classify(nn.local) - nsdf: EE   124(in:/su:/ou/er)   0  96  28  28      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  6113 m  77 p  4815 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp.classify(nn.local) - nsdf: EE   124(in:/su:/ou/er)   0  96  28  28      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  6114 m  77 p  4815 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp.classify(nn.local) - nsdf: EE   124(in:/su:/ou/er)   0  96  28  28      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  6115 m  77 p  4815 n.pv lvOIL#pvSstTopRadiusRibs#SstBTpp.classify(nn.local) - nsdf: EE   124(in:/su:/ou/er)   0  96  28  28      0.000  1025.250 ep 1.000000e-03 [0.000000e+00,1.025250e+03] 
    NSc::csp n  6133 m  81 p  4815 n.pv lvOIL#pvSstInnVerRibs#SstInnVepp.classify(nn.local) - nsdf: EE   149(in:/su:/ou/er)  71  40  38  38   -120.000     0.031 ep 1.000000e-03 [-1.200000e+02,3.125000e-02] 
    NSc::csp n  6134 m  81 p  4815 n.pv lvOIL#pvSstInnVerRibs#SstInnVepp.classify(nn.local) - nsdf: EE   149(in:/su:/ou/er)  71  40  38  38   -120.000     0.031 ep 1.000000e-03 [-1.200000e+02,3.125000e-02] 





::

    NSc::csp n  3149 nlv 234 p  3148 n.pv lvNearPoolDead#pvNearPoolLinerpp(nn.local) - nsdf: EE   134(in:/su/ou/er) 109  25   0  25    -84.000     0.000 ep 1.000000e-03 [-8.400000e+01,0.000000e+00] 
    NSc::csp n  3150 nlv 232 p  3149 n.pv lvNearPoolLiner#pvNearPoolOWS0pp(nn.local) - nsdf: EE   122(in:/su/ou/er)  97  25   0  25     -8.000     0.000 ep 1.000000e-03 [-8.000000e+00,0.000000e+00] 
    NSc::csp n  3151 nlv 213 p  3150 n.pv lvNearPoolOWS#pvNearPoolCurtaipp(nn.local) - nsdf: EE   122(in:/su/ou/er)  97  25   0  25  -1000.000     0.000 ep 1.000000e-03 [-1.000000e+03,0.000000e+00] 
    NSc::csp n  3152 nlv 211 p  3151 n.pv lvNearPoolCurtain#pvNearPoolIWpp(nn.local) - nsdf: EE   182(in:/su/ou/er) 157  25   0  25     -8.000     0.000 ep 1.000000e-03 [-8.000000e+00,0.000000e+00] 
    NSc::csp n  3157 nlv  37 p  3156 n.pv           lvOAV#pvLSO0xbf8e120pp(nn.local) - nsdf: EE   110(in:/su/ou/er)  85  25   0  25    -18.025     0.000 ep 1.000000e-03 [-1.802490e+01,0.000000e+00] 
    NSc::csp n  3159 nlv  22 p  3158 n.pv           lvIAV#pvGDS0xbf6ab00pp(nn.local) - nsdf: EE   105(in:/su/ou/er)  80  25   0  25    -15.000     0.000 ep 1.000000e-03 [-1.500000e+01,0.000000e+00] DEBUG_NODE 
    2017-06-30 20:47:00.885 WARN  [2230698] [NSDF::classify@74]  sd.size ZERO 
    NSc::csp n  3163 nlv  27 p  3157 n.pv lvLSO#pvCtrGdsOflTfbInLso0xc2cpp(nn.local) - nsdf: EE    55(in:/su/ou/er)  50   5   0   5   -207.000     0.000 ep 1.000000e-03 [-2.070000e+02,0.000000e+00] 
    NSc::csp n  3164 nlv  28 p  3157 n.pv lvLSO#pvCtrGdsOflInLso0xbf7425pp(nn.local) - nsdf: EE    75(in:/su/ou/er)  50  25   0  25   -347.560    -0.000 ep 1.000000e-03 [-3.475604e+02,-2.441406e-04] 
    NSc::csp n  3167 nlv  30 p  3157 n.pv lvLSO#pvOcrGdsTfbInLso0xbfa181pp(nn.local) - nsdf: EE    25(in:/su/ou/er)  10   0  15  15     -9.528    66.378 ep 1.000000e-03 [-9.528076e+00,6.637841e+01] 
    NSc::csp n  3168 nlv  31 p  3157 n.pv   lvLSO#pvOcrGdsInLso0xbf6d280pp(nn.local) - nsdf: EE    35(in:/su/ou/er)   5   0  30  30    -53.925    66.329 ep 1.000000e-03 [-5.392529e+01,6.632938e+01] 
    NSc::csp n  3169 nlv  32 p  3157 n.pv lvLSO#pvOavBotRibs#OavBotRibs#pp(nn.local) - nsdf: EE   150(in:/su/ou/er)  72  40  38  78   -197.000     0.158 ep 1.000000e-03 [-1.970000e+02,1.577148e-01] 
    NSc::csp n  3170 nlv  32 p  3157 n.pv lvLSO#pvOavBotRibs#OavBotRibs:pp(nn.local) - nsdf: EE   150(in:/su/ou/er)  72  40  38  78   -197.000     0.158 ep 1.000000e-03 [-1.970000e+02,1.575928e-01] 
    NSc::csp n  3171 nlv  32 p  3157 n.pv lvLSO#pvOavBotRibs#OavBotRibs:pp(nn.local) - nsdf: EE   150(in:/su/ou/er)  72  40  38  78   -197.000     0.158 ep 1.000000e-03 [-1.970000e+02,1.577148e-01] 
    NSc::csp n  3172 nlv  32 p  3157 n.pv lvLSO#pvOavBotRibs#OavBotRibs:pp(nn.local) - nsdf: EE   150(in:/su/ou/er)  72  40  38  78   -197.000     0.158 ep 1.000000e-03 [-1.970000e+02,1.575928e-01] 
    NSc::csp n  3173 nlv  32 p  3157 n.pv lvLSO#pvOavBotRibs#OavBotRibs:pp(nn.local) - nsdf: EE   150(in:/su/ou/er)  72  40  38  78   -197.000     0.158 ep 1.000000e-03 [-1.970000e+02,1.577148e-01] 
    NSc::csp n  3174 nlv  32 p  3157 n.pv lvLSO#pvOavBotRibs#OavBotRibs:pp(nn.local) - nsdf: EE   150(in:/su/ou/er)  72  40  38  78   -197.000     0.158 ep 1.000000e-03 [-1.970000e+02,1.575928e-01] 
    NSc::csp n  3175 nlv  32 p  3157 n.pv lvLSO#pvOavBotRibs#OavBotRibs:pp(nn.local) - nsdf: EE   150(in:/su/ou/er)  72  40  38  78   -197.000     0.158 ep 1.000000e-03 [-1.970000e+02,1.577148e-01] 
    NSc::csp n  3176 nlv  32 p  3157 n.pv lvLSO#pvOavBotRibs#OavBotRibs:pp(nn.local) - nsdf: EE   150(in:/su/ou/er)  72  40  38  78   -197.000     0.158 ep 1.000000e-03 [-1.970000e+02,1.575928e-01] 
    NSc::csp n  3177 nlv  33 p  3157 n.pv     lvLSO#pvOavBotHub0xbf21f78pp(nn.local) - nsdf: EE    75(in:/su/ou/er)  45  30   0  30   -197.000    -0.000 ep 1.000000e-03 [-1.970000e+02,-0.000000e+00] 
    NSc::csp n  3195 nlv  40 p  3156 n.pv lvOAV#pvOcrGdsLsoInOav0xbfa3dfpp(nn.local) - nsdf: EE     5(in:/su/ou/er)   0   0   5   5      6.302    11.188 ep 1.000000e-03 [6.301849e+00,1.118755e+01] 
    2017-06-30 20:47:00.916 WARN  [2230698] [NSDF::classify@74]  sd.size ZERO 
    NSc::csp n  3199 nlv  47 p  3155 n.pv lvOIL#pvAdPmtArray#pvAdPmtArrapp(nn.local) - nsdf: EE    55(in:/su/ou/er)  25   0  30  30   -294.500     5.858 ep 1.000000e-03 [-2.945000e+02,5.857910e+00] 
    NSc::csp n  3201 nlv  43 p  3200 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-03 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3202 nlv  44 p  3200 n.pv lvPmtHemiVacuum#pvPmtHemiBottopp(nn.local) - nsdf: EE    45(in:/su/ou/er)  25  20   0  20    -31.500     0.000 ep 1.000000e-03 [-3.150000e+01,0.000000e+00] 
    NSc::csp n  3203 nlv  45 p  3200 n.pv lvPmtHemiVacuum#pvPmtHemiDynodpp(nn.local) - nsdf: EE    75(in:/su/ou/er)  45  30   0  30    -31.500    -0.000 ep 1.000000e-03 [-3.150000e+01,-0.000000e+00] 
    NSc::csp n  3205 nlv  47 p  3155 n.pv lvOIL#pvAdPmtArray#pvAdPmtArrapp(nn.local) - nsdf: EE    55(in:/su/ou/er)  25   0  30  30   -294.500     5.858 ep 1.000000e-03 [-2.945000e+02,5.857910e+00] 
    NSc::csp n  3207 nlv  43 p  3206 n.pv lvPmtHemiVacuum#pvPmtHemiCathopp(nn.local) - nsdf: EE   115(in:/su/ou/er)  60  40  15  55    -29.238     0.313 ep 1.000000e-03 [-2.923752e+01,3.130188e-01] 
    NSc::csp n  3208 nlv  44 p  3206 n.pv lvPmtHemiVacuum#pvPmtHemiBottopp(nn.local) - nsdf: EE    45(in:/su/ou/er)  25  20   0  20    -31.500     0.000 ep 1.000000e-03 [-3.150000e+01,0.000000e+00] 




NScene::check_surf_points : this node SDF issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 4 solids have no parametric points
* some PMT param points outside whopper epsilon SDF range ~0.25mm


::

    Sc::cac n  12227 p   3147 mn(n-p) (  10443.004  8369.250   150.000) mx(p-n) (   1794.919  2774.500   150.000) n.pv lvNearHallBot#pvNearHallRadSla err 
    NSc::cac n  12228 p   3147 mn(n-p) (   6288.400 16757.875   150.000) mx(p-n) (   7410.776  1753.500   150.000) n.pv lvNearHallBot#pvNearHallRadSla err 
    NSc::cac n  12229 p   3147 mn(n-p) (    414.836   414.875  -150.000) mx(p-n) (    414.838   414.875 10150.000) n.pv lvNearHallBot#pvNearHallRadSla err ZMIN_OUT 
    2017-06-29 20:37:34.769 INFO  [1977826] [NScene::check_aabb_containment@761] NScene::check_aabb_containment (cac) verbosity 1 tot 12230 err 3491 err/tot       0.29
    2017-06-29 20:37:34.769 INFO  [1977826] [NScene::check_surf_containment@501] NScene::check_surf_containment (csc) verbosity 1
    2017-06-29 20:37:35.919 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:35.919 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:35.919 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:35.919 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    NSc::csp n  3160 p  3158 npt   0 nsd   ??   0(in/su*/ou/er)   0   0   0   0      0.000     0.000 psd   ??   0(in*/su/ou/er)   0   0   0   0      0.000     0.000 n.pv lvIAV#pvOcrGdsInIAV0xbf6b0e0 
    2017-06-29 20:37:35.932 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:35.932 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:35.932 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:35.933 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    NSc::csp n  3198 p  3156 npt   0 nsd   ??   0(in/su*/ou/er)   0   0   0   0      0.000     0.000 psd   ??   0(in*/su/ou/er)   0   0   0   0      0.000     0.000 n.pv lvOAV#pvOcrCalLsoInOav0xbfa3eb 

    NSc::csp n  3293 p  3290 npt  75 nsd **    75(in/su*/ou/er)   0  73   2   2     -0.188     0.250 psd       75(in*/su/ou/er)  75   0   0   0    -31.527    -0.020 n.pv lvPmtHemiVacuum#pvPmtHemiDynod 
    NSc::csp n  3294 p  3155 npt  55 nsd **    55(in/su*/ou/er)   1  54   0   1     -0.250     0.188 psd       55(in*/su/ou/er)  55   0   0   0   -225.863  -210.695 n.pv lvOIL#pvAdPmtArray#pvAdPmtArra 
    NSc::csp n  3437 p  3434 npt  75 nsd **    75(in/su*/ou/er)   0  73   2   2     -0.188     0.250 psd       75(in*/su/ou/er)  75   0   0   0    -31.527    -0.020 n.pv lvPmtHemiVacuum#pvPmtHemiDynod 
    NSc::csp n  3438 p  3155 npt  55 nsd **    55(in/su*/ou/er)   1  54   0   1     -0.250     0.188 psd       55(in*/su/ou/er)  55   0   0   0   -225.863  -210.695 n.pv lvOIL#pvAdPmtArray#pvAdPmtArra 
    NSc::csp n  3581 p  3578 npt  75 nsd **    75(in/su*/ou/er)   0  73   2   2     -0.188     0.250 psd       75(in*/su/ou/er)  75   0   0   0    -31.527    -0.020 n.pv lvPmtHemiVacuum#pvPmtHemiDynod 
    NSc::csp n  3582 p  3155 npt  55 nsd **    55(in/su*/ou/er)   1  54   0   1     -0.250     0.188 psd       55(in*/su/ou/er)  55   0   0   0   -225.863  -210.695 n.pv lvOIL#pvAdPmtArray#pvAdPmtArra 
    NSc::csp n  3725 p  3722 npt  75 nsd **    75(in/su*/ou/er)   0  73   2   2     -0.188     0.250 psd       75(in*/su/ou/er)  75   0   0   0    -31.527    -0.020 n.pv lvPmtHemiVacuum#pvPmtHemiDynod 
    NSc::csp n  3726 p  3155 npt  55 nsd **    55(in/su*/ou/er)   1  54   0   1     -0.250     0.188 psd       55(in*/su/ou/er)  55   0   0   0   -225.863  -210.695 n.pv lvOIL#pvAdPmtArray#pvAdPmtArra 
    NSc::csp n  3869 p  3866 npt  75 nsd **    75(in/su*/ou/er)   0  73   2   2     -0.188     0.250 psd       75(in*/su/ou/er)  75   0   0   0    -31.527    -0.020 n.pv lvPmtHemiVacuum#pvPmtHemiDynod 
    NSc::csp n  3870 p  3155 npt  55 nsd **    55(in/su*/ou/er)   1  54   0   1     -0.250     0.188 psd       55(in*/su/ou/er)  55   0   0   0   -225.863  -210.695 n.pv lvOIL#pvAdPmtArray#pvAdPmtArra 
    NSc::csp n  4013 p  4010 npt  75 nsd **    75(in/su*/ou/er)   0  73   2   2     -0.188     0.250 psd       75(in*/su/ou/er)  75   0   0   0    -31.527    -0.020 n.pv lvPmtHemiVacuum#pvPmtHemiDynod 
    NSc::csp n  4014 p  3155 npt  55 nsd **    55(in/su*/ou/er)   1  54   0   1     -0.250     0.188 psd       55(in*/su/ou/er)  55   0   0   0   -225.863  -210.695 n.pv lvOIL#pvAdPmtArray#pvAdPmtArra 
    NSc::csp n  4157 p  4154 npt  75 nsd **    75(in/su*/ou/er)   0  73   2   2     -0.188     0.250 psd       75(in*/su/ou/er)  75   0   0   0    -31.527    -0.020 n.pv lvPmtHemiVacuum#pvPmtHemiDynod 
    NSc::csp n  4158 p  3155 npt  55 nsd **    55(in/su*/ou/er)   1  54   0   1     -0.250     0.188 psd       55(in*/su/ou/er)  55   0   0   0   -225.863  -210.695 n.pv lvOIL#pvAdPmtArray#pvAdPmtArra 
    NSc::csp n  4301 p  4298 npt  75 nsd **    75(in/su*/ou/er)   0  73   2   2     -0.188     0.250 psd       75(in*/su/ou/er)  75   0   0   0    -31.527    -0.020 n.pv lvPmtHemiVacuum#pvPmtHemiDynod 
    NSc::csp n  4302 p  3155 npt  55 nsd **    55(in/su*/ou/er)   1  54   0   1     -0.250     0.188 psd       55(in*/su/ou/er)  55   0   0   0   -225.863  -210.695 n.pv lvOIL#pvAdPmtArray#pvAdPmtArra 

    2017-06-29 20:37:36.359 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:36.359 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:36.359 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:36.359 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    NSc::csp n  4820 p  4818 npt   0 nsd   ??   0(in/su*/ou/er)   0   0   0   0      0.000     0.000 psd   ??   0(in*/su/ou/er)   0   0   0   0      0.000     0.000 n.pv lvIAV#pvOcrGdsInIAV0xbf6b0e0 
    2017-06-29 20:37:36.372 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:36.372 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:36.372 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    2017-06-29 20:37:36.372 WARN  [1977826] [NSDF::apply@594]  sd.size ZERO 
    NSc::csp n  4858 p  4816 npt   0 nsd   ??   0(in/su*/ou/er)   0   0   0   0      0.000     0.000 psd   ??   0(in*/su/ou/er)   0   0   0   0      0.000     0.000 n.pv lvOAV#pvOcrCalLsoInOav0xbfa3eb 
    2017-06-29 20:37:40.609 INFO  [1977826] [NScene::check_surf_containment@509] NScene::check_surf_containment (csc) verbosity 1 tot 12230 surferr    5690      16   10647    1129
    Assertion failed: (0 && "NScene::postimportmesh HARIKARI"), function postimportmesh, file /Users/blyth/opticks/opticksnpy/NScene.cpp, line 437.





NScene::check_surf_points : parent node SDF issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* a large fraction of nodes have small parent node SDF impingement/coicidence issues

  * suspect lack of precision issue from doing comparisons in global frame ... a long way from origin
  * need to arrange comparisons to be done closer to origin somehow : by picking an appropriate 
    frame in which to compare

Deviations are tending to land on particular values ::

    In [3]: np.arange(1,10,dtype=np.float32)*0.031
    Out[3]: array([ 0.031,  0.062,  0.093,  0.124,  0.155,  0.186,  0.217,  0.248,  0.279], dtype=float32)


::

    017-06-29 20:46:01.457 INFO  [1981075] [NScene::check_aabb_containment@760] NScene::check_aabb_containment (cac) verbosity 1 tot 12230 err 3491 err/tot       0.29
    2017-06-29 20:46:01.457 INFO  [1981075] [NScene::check_surf_containment@501] NScene::check_surf_containment (csc) verbosity 1
    NSc::csp n     0 p     0 npt 150 nsd      150(in/su*/ou/er)   0 150   0   0      0.000     0.000 psd **   150(in*/su/ou/er)   0 150   0 150      0.000     0.000 n.pv top 
    NSc::csp n     2 p     1 npt 300 nsd      300(in/su*/ou/er)   0 300   0   0     -0.031     0.062 psd **   300(in*/su/ou/er) 255   0  45  45 -17500.000  3000.062 n.pv lvNearSiteRock#pvNearHallTop0x 
    NSc::csp n     3 p     2 npt 122 nsd      122(in/su*/ou/er)   0 122   0   0     -0.031     0.031 psd **   122(in*/su/ou/er)  97  25   0  25    -44.000     0.000 n.pv lvNearHallTop#pvNearTopCover0x 
    NSc::csp n     8 p     7 npt 150 nsd      150(in/su*/ou/er)   0 150   0   0     -0.062     0.062 psd **   150(in*/su/ou/er)  24  60  66 126     -1.000     0.062 n.pv lvRPCGasgap14#pvStrip14Array#p 
    NSc::csp n     9 p     7 npt 150 nsd      150(in/su*/ou/er)   0 150   0   0     -0.031     0.062 psd **   150(in*/su/ou/er)  39  97  14 111     -1.000     0.031 n.pv lvRPCGasgap14#pvStrip14Array#p 
    NSc::csp n    10 p     7 npt 150 nsd      150(in/su*/ou/er)   0 150   0   0     -0.062     0.062 psd **   150(in*/su/ou/er)  39  78  33 111     -1.000     0.031 n.pv lvRPCGasgap14#pvStrip14Array#p 
    NSc::csp n    11 p     7 npt 150 nsd      150(in/su*/ou/er)   0 150   0   0     -0.062     0.062 psd **   150(in*/su/ou/er)  36  69  45 114     -1.000     0.062 n.pv lvRPCGasgap14#pvStrip14Array#p 
    NSc::csp n    12 p     7 npt 150 nsd      150(in/su*/ou/er)   0 150   0   0      0.000     0.062 psd **   150(in*/su/ou/er)  24  93  33 126     -1.000     0.031 n.pv lvRPCGasgap14#pvStrip14Array#p 
    NSc::csp n    13 p     7 npt 150 nsd      150(in/su*/ou/er)   0 150   0   0     -0.031     0.062 psd **   150(in*/su/ou/er)  33  72  45 117     -1.000     0.031 n.pv lvRPCGasgap14#pvStrip14Array#p 
    NSc::csp n    14 p     7 npt 150 nsd      150(in/su*/ou/er)   0 150   0   0     -0.062     0.062 psd **   150(in*/su/ou/er)  39  66  45 111     -1.000     0.062 n.pv lvRPCGasgap14#pvStrip14Array#p 


Some big impingements too::

    Sc::csp n  4444 p  3155 npt  33 nsd       33(in/su*/ou/er)   0  33   0   0     -0.031     0.031 psd **    33(in*/su/ou/er)  27   6   0   6   -430.000    -0.000 n.pv lvOIL#pvSstBotCirRib#SstBotCir 
    NSc::csp n  4445 p  3155 npt  33 nsd       33(in/su*/ou/er)   0  33   0   0     -0.125     0.125 psd **    33(in*/su/ou/er)  27   6   0   6   -430.000    -0.000 n.pv lvOIL#pvSstBotCirRib#SstBotCir 
    NSc::csp n  4446 p  3155 npt  33 nsd       33(in/su*/ou/er)   0  33   0   0     -0.031     0.031 psd **    33(in*/su/ou/er)  27   6   0   6   -430.000    -0.000 n.pv lvOIL#pvSstBotCirRib#SstBotCir 
    NSc::csp n  4447 p  3155 npt  33 nsd       33(in/su*/ou/er)   0  33   0   0     -0.062     0.062 psd **    33(in*/su/ou/er)  27   6   0   6   -430.000    -0.000 n.pv lvOIL#pvSstBotCirRib#SstBotCir 
    NSc::csp n  4448 p  3155 npt 149 nsd      149(in/su*/ou/er)   0 149   0   0     -0.062    -0.000 psd **   149(in*/su/ou/er)  25  96  28 124   -212.002  1025.188 n.pv lvOIL#pvSstTopRadiusRibs#SstBT 
    NSc::csp n  4449 p  3155 npt 149 nsd      149(in/su*/ou/er)   0 149   0   0     -0.047     0.062 psd **   149(in*/su/ou/er)  25  96  28 124   -212.002  1025.238 n.pv lvOIL#pvSstTopRadiusRibs#SstBT 
    NSc::csp n  4450 p  3155 npt 149 nsd      149(in/su*/ou/er)   0 149   0   0     -0.062     0.031 psd **   149(in*/su/ou/er)  25  96  28 124   -212.002  1025.188 n.pv lvOIL#pvSstTopRadiusRibs#SstBT 
    NSc::csp n  4451 p  3155 npt 149 nsd      149(in/su*/ou/er)   0 149   0   0     -0.062     0.062 psd **   149(in*/su/ou/er)  25  96  28 124   -212.002  1025.238 n.pv lvOIL#pvSstTopRadiusRibs#SstBT 
    NSc::csp n  4452 p  3155 npt 149 nsd      149(in/su*/ou/er)   0 149   0   0     -0.062     0.031 psd **   149(in*/su/ou/er)  25  96  28 124   -212.002  1025.188 n.pv lvOIL#pvSstTopRadiusRibs#SstBT 
    NSc::csp n  4453 p  3155 npt 149 nsd      149(in/su*/ou/er)   0 149   0   0     -0.047     0.047 psd **   149(in*/su/ou/er)  25  96  28 124   -212.002  1025.282 n.pv lvOIL#pvSstTopRadiusRibs#SstBT 
    NSc::csp n  4454 p  3155 npt 149 nsd      149(in/su*/ou/er)   0 149   0   0     -0.000     0.031 psd **   149(in*/su/ou/er)  25  96  28 124   -212.002  1025.250 n.pv lvOIL#pvSstTopRadiusRibs#SstBT 
    NSc::csp n  4455 p  3155 npt 149 nsd      149(in/su*/ou/er)   0 149   0   0     -0.062     0.062 psd **   149(in*/su/ou/er)  25  96  28 124   -212.002  1025.260 n.pv lvOIL#pvSstTopRadiusRibs#SstBT 
    NSc::csp n  4464 p  3155 npt 100 nsd      100(in/su*/ou/er)   0 100   0   0     -0.062     0.000 psd **   100(in*/su/ou/er)  75  25   0  25   -340.000     0.000 n.pv lvOIL#pvSstTopHub0xc2476b8 
    NSc::csp n  4473 p  3155 npt 149 nsd      149(in/su*/ou/er)   0 149   0   0     -0.062     0.062 psd **   149(in*/su/ou/er)  71  33  45  78   -120.000     0.062 n.pv lvOIL#pvSstInnVerRibs#SstInnVe 
    NSc::csp n  4474 p  3155 npt 149 nsd      149(in/su*/ou/er)   0 149   0   0     -0.125     0.125 psd **   149(in*/su/ou/er)  81  37  31  68   -120.024     0.009 n.pv lvOIL#pvSstInnVerRibs#SstInnVe 



Parametric Convex Polyhedron ?
---------------------------------

*nconvexpolyhedron* is defined by a set of planes, 
unclear how to parametrize, as it is so general 
eg could be a tetrahedron, cube, prism, trapezoid, octahedron, dodecahedron, ...

* need intersections points of planes to define the shapes of the faces, 
  which could then be subdivided


* https://mathoverflow.net/questions/138494/finding-the-vertices-of-a-convex-polyhedron-from-a-set-of-planes
* http://cgm.cs.mcgill.ca/~avis/doc/avis/AF92b.pdf
* https://www.inf.ethz.ch/personal/fukudak/soft/soft.html
* https://www.inf.ethz.ch/personal/fukudak/polyfaq/polyfaq.html

* https://www.inf.ethz.ch/personal/fukudak/polyfaq/node41.html

* http://www.cs.mcgill.ca/~fukuda/software/cdd_home/cdd.html




Check SDF values of one object for surface verts of other object 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* this is an approximation (as only checking a subset of the surface) : 
  but a very good one depending on how good the parametrizations are 

* for example for containment the parent SDF values of child object should
  all be negative : indicating are inside the parent volume 



Checking Composite Surface Point Generation
---------------------------------------------

::

     .

      20         +-------*--------+
                 |              B |
                 |                |
      10 +-------*3------+        *
         |       |       |        |   
         |       |       |        |
       0 *0      +-------*1-------+     
         |               |
         | A             | 
     -10 +-------*2------+
       -10       0       10      20


        X
        |
        +-- Z


::

    Process 65773 launched: '/usr/local/opticks/lib/NNodeTest' (x86_64)
    2017-06-28 19:54:42.041 INFO  [1739966] [test_getSurfacePointsAll_Composite@299] test_getSurfacePointsAll_Composite
    nnode::dump [ 0:di] OPER  v:0
             L [ 0:bo] PRIM  v:0 bb  mi  ( -10.00  -10.00  -10.00)  mx  (  10.00   10.00   10.00)  si  (  20.00   20.00   20.00) 
             R [ 0:bo] PRIM  v:0 bb  mi  (   0.00  -10.00    0.00)  mx  (  20.00   10.00   20.00)  si  (  20.00   20.00   20.00) 
    2017-06-28 19:54:42.041 INFO  [1739966] [nnode::dumpPointsSDF@989] nnode::dumpPointsSDF points 12
     i    0 p (      0.000     0.000   -10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    1 p (      0.000     0.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    2 p (    -10.000     0.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    3 p (     10.000     0.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    4 p (      0.000   -10.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    5 p (      0.000    10.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    6 p (     10.000     0.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    7 p (     10.000     0.000    20.000) sd(fx4)    10.0000 sd(sci) 1.0000e+01 sd(def)         10
     i    8 p (      0.000     0.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    9 p (     20.000     0.000    10.000) sd(fx4)    10.0000 sd(sci) 1.0000e+01 sd(def)         10
     i   10 p (     10.000   -10.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i   11 p (     10.000    10.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
    2017-06-28 19:54:42.042 INFO  [1739966] [nnode::dumpPointsSDF@1021] nnode::dumpPointsSDF points     12 epsilon 1.000000e-05 num_inside      0 num_surface     10 num_outside      2
    nnode::dump [ 0:un] OPER  v:0
             L [ 0:bo] PRIM  v:0 bb  mi  ( -10.00  -10.00  -10.00)  mx  (  10.00   10.00   10.00)  si  (  20.00   20.00   20.00) 
             R [ 0:bo] PRIM  v:0 bb  mi  (   0.00  -10.00    0.00)  mx  (  20.00   10.00   20.00)  si  (  20.00   20.00   20.00) 
    2017-06-28 19:54:42.042 INFO  [1739966] [nnode::dumpPointsSDF@989] nnode::dumpPointsSDF points 12
     i    0 p (      0.000     0.000   -10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    1 p (      0.000     0.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    2 p (    -10.000     0.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    3 p (     10.000     0.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    4 p (      0.000   -10.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    5 p (      0.000    10.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    6 p (     10.000     0.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    7 p (     10.000     0.000    20.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    8 p (      0.000     0.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    9 p (     20.000     0.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i   10 p (     10.000   -10.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i   11 p (     10.000    10.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
    2017-06-28 19:54:42.042 INFO  [1739966] [nnode::dumpPointsSDF@1021] nnode::dumpPointsSDF points     12 epsilon 1.000000e-05 num_inside      0 num_surface     12 num_outside      0
    nnode::dump [ 0:in] OPER  v:0
             L [ 0:bo] PRIM  v:0 bb  mi  ( -10.00  -10.00  -10.00)  mx  (  10.00   10.00   10.00)  si  (  20.00   20.00   20.00) 
             R [ 0:bo] PRIM  v:0 bb  mi  (   0.00  -10.00    0.00)  mx  (  20.00   10.00   20.00)  si  (  20.00   20.00   20.00) 
    2017-06-28 19:54:42.042 INFO  [1739966] [nnode::dumpPointsSDF@989] nnode::dumpPointsSDF points 12
     i    0 p (      0.000     0.000   -10.000) sd(fx4)    10.0000 sd(sci) 1.0000e+01 sd(def)         10
     i    1 p (      0.000     0.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    2 p (    -10.000     0.000     0.000) sd(fx4)    10.0000 sd(sci) 1.0000e+01 sd(def)         10
     i    3 p (     10.000     0.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    4 p (      0.000   -10.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    5 p (      0.000    10.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    6 p (     10.000     0.000     0.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    7 p (     10.000     0.000    20.000) sd(fx4)    10.0000 sd(sci) 1.0000e+01 sd(def)         10
     i    8 p (      0.000     0.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i    9 p (     20.000     0.000    10.000) sd(fx4)    10.0000 sd(sci) 1.0000e+01 sd(def)         10
     i   10 p (     10.000   -10.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
     i   11 p (     10.000    10.000    10.000) sd(fx4)     0.0000 sd(sci) 0.0000e+00 sd(def)          0
    2017-06-28 19:54:42.042 INFO  [1739966] [nnode::dumpPointsSDF@1021] nnode::dumpPointsSDF points     12 epsilon 1.000000e-05 num_inside      0 num_surface      8 num_outside      4
    Process 65773 exited with status = 0 (0x00000000) 





tgltf-t : Look at gds example
----------------------------------

::

    tgltf-;tgltf-t  ## with OPTICKS_QUERY selection to pick two volumes only, and manual dumping


Comparing gds and parent nd volumes in NScene::postimportmesh find that they have coincident bbox in Z.

* this is highly likely to explain the tachyon behaviour


Whats the appropriate fix ?
----------------------------

* nudging CSG (eg a few epsilon decrease_z2 or increase_z1) 
  would apply to all instances, so that might not be appropriate 

  * need to check if all lv are similarly coincident

* otherwise would need to apply a nudge transform to the node ? 


Are there missing transforms ?
----------------------------------

* TODO: examine full structural transform tree, for node and its parent, to look for bugs

::

    Hmm : is there 2.5mm of z translation missing in the parent (iav) gtransform ?

             -7101.5
             -7100.0


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




NScene::check_containment checking bbox containment of all node/parent pairs
----------------------------------------------------------------------------------

* 30% of volumes have bbox containment issues, including PMT volumes

  * bbox impingement doesnt mean solid impingement : it just provides a fast 
    selection of possible collisions for more expensive object-object testing
 
  * perhaps a missing transform bug ? perhaps but first check obj-obj collisions

  * all the mn and mx in (mm) in the below table 
    should be +ve, they are zero with coincidence and -ve with protrusion  

  * TODO: check the instanced are correctly treated here


Are checking containment by comparing the globally transformed axis aligned bbox 
of a node and its parent.

* is there a better way to check containment ? YES : obj-obj

* rotational transforms change box dimensions (as bbox stays axis aligned), 

* perhaps should transform into parent frame to make comparison ?


::

     514 void NScene::check_containment_r(nd* node)
     515 {
     516     nd* parent = node->parent ;
     517     if(!parent) parent = node ;   // only root should not have parent
     518 
     519     nbbox  nbb = get_bbox( node->idx ) ;
     520     nbbox  pbb = get_bbox( parent->idx ) ;
     521 
     522     float epsilon = 1e-5 ;
     523 
     524     unsigned errmask = nbb.classify_containment( pbb, epsilon );
     525 
     526     node->containment = errmask ;
     527 
     528     if(errmask) m_containment_err++ ;
     529 
     530     //if(m_verbosity > 2 || ( errmask && m_verbosity > 0))
     531     {
     532         glm::vec3 dmin( nbb.min.x - pbb.min.x,
     533                         nbb.min.y - pbb.min.y,
     534                         nbb.min.z - pbb.min.z );
     535 
     536         glm::vec3 dmax( pbb.max.x - nbb.max.x,
     537                         pbb.max.y - nbb.max.y,
     538                         pbb.max.z - nbb.max.z );



     442 nbbox NScene::calc_bbox(const nd* node, bool global) const
     443 {
     444     unsigned mesh_idx = node->mesh ;
     445 
     446     NCSG* csg = getCSG(mesh_idx);
     447     assert(csg);
     448 
     449     nnode* root = csg->getRoot();
     450     assert(root);
     451 
     452     assert( node->gtransform );
     453     const glm::mat4& node_t  = node->gtransform->t ;
     454 
     455     nbbox bb  = root->bbox();
     456 
     457     nbbox gbb = bb.transform(node_t) ;
     458 
     459     if(m_verbosity > 2)
     460     std::cout
     461         << " get_bbox "
     462         << " verbosity " << m_verbosity
     463         << " mesh_idx "  << mesh_idx
     464         << " root "  << root->tag()
     465         << std::endl
     466         << gpresent("node_t", node_t)
     467         << std::endl
     468         << " bb  " <<  bb.desc() << std::endl
     469         << " gbb " <<  gbb.desc() << std::endl
     470         ;
     471 
     472     return global ? gbb : bb ;
     473 }



::

    2017-06-27 20:45:11.089 INFO  [1538289] [NScene::postimportmesh@420] NScene::postimportmesh numNd 12230 dbgnode 3159 verbosity 1
    2017-06-27 20:45:11.116 INFO  [1538289] [NScene::check_containment@498] NScene::check_containment verbosity 1
    NSc::ccr n      0 p      0 mn(n-p) (      0.000     0.000     0.000) mx(p-n) (      0.000     0.000     0.000) pv                            top err XMIN_CO YMIN_CO ZMIN_CO XMAX_CO YMAX_CO ZMAX_CO 
    NSc::ccr n      1 p      0 mn(n-p) ( 2348910.2501563320.1252372890.000) mx(p-n) ( 2381950.2503167540.0002377110.000) pv               db-rock0xc15d358 err 
    NSc::ccr n      2 p      1 mn(n-p) (  20001.729  7258.312 25000.000) mx(p-n) (  12644.018 16790.562 10000.000) pv lvNearSiteRock#pvNearHallTop0x err 
    NSc::ccr n      3 p      2 mn(n-p) (   6024.635 17878.750     0.000) mx(p-n) (  13382.347  8346.500 14956.000) pv lvNearHallTop#pvNearTopCover0x err ZMIN_CO 
    NSc::ccr n      4 p      2 mn(n-p) (  17966.039 28909.250  2754.903) mx(p-n) (  15508.528 13171.500 12167.097) pv lvNearHallTop#pvNearTeleRpc#pv err 
    NSc::ccr n      5 p      4 mn(n-p) (     55.189    38.312     1.500) mx(p-n) (     52.945    60.562     1.500) pv    lvRPCMod#pvRPCFoam0xbf1a820 err 
    NSc::ccr n      6 p      5 mn(n-p) (      6.899     6.875    20.500) mx(p-n) (      6.899     6.875    48.500) pv lvRPCFoam#pvBarCham14Array#pvB err 
    NSc::ccr n      7 p      6 mn(n-p) (     13.797    13.812     2.000) mx(p-n) (     13.797    13.812     2.000) pv lvRPCBarCham14#pvRPCGasgap140x err 
    NSc::ccr n      8 p      7 mn(n-p) (    973.189     0.000     0.000) mx(p-n) (      0.000  1538.000     0.000) pv lvRPCGasgap14#pvStrip14Array#p err YMIN_CO ZMIN_CO XMAX_CO ZMAX_CO 
    NSc::ccr n      9 p      7 mn(n-p) (    834.162   219.750     0.000) mx(p-n) (    139.027  1318.250     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     10 p      7 mn(n-p) (    695.136   439.438     0.000) mx(p-n) (    278.054  1098.562     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     11 p      7 mn(n-p) (    556.108   659.125     0.000) mx(p-n) (    417.081   878.875     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     12 p      7 mn(n-p) (    417.081   878.875     0.000) mx(p-n) (    556.108   659.125     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     13 p      7 mn(n-p) (    278.054  1098.562     0.000) mx(p-n) (    695.136   439.438     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     14 p      7 mn(n-p) (    139.027  1318.250     0.000) mx(p-n) (    834.162   219.750     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     15 p      7 mn(n-p) (      0.000  1538.000     0.000) mx(p-n) (    973.189     0.000     0.000) pv lvRPCGasgap14#pvStrip14Array#p err XMIN_CO ZMIN_CO YMAX_CO ZMAX_CO 
    NSc::ccr n     16 p      5 mn(n-p) (      6.899     6.875    58.500) mx(p-n) (      6.899     6.875    10.500) pv lvRPCFoam#pvBarCham14Array#pvB err 
    NSc::ccr n     17 p     16 mn(n-p) (     13.797    13.812     2.000) mx(p-n) (     13.797    13.812     2.000) pv lvRPCBarCham14#pvRPCGasgap140x err 
    NSc::ccr n     18 p     17 mn(n-p) (    973.189     0.000     0.000) mx(p-n) (      0.000  1538.000     0.000) pv lvRPCGasgap14#pvStrip14Array#p err YMIN_CO ZMIN_CO XMAX_CO ZMAX_CO 
    ...
    NSc::ccr n   3142 p   2968 mn(n-p) (   6025.996  5863.750    42.000) mx(p-n) (   6148.171  3832.000    42.000) pv lvNearHbeamBigUnit#pvNearRight err 
    NSc::ccr n   3143 p   2968 mn(n-p) (   5132.042  5358.812    42.000) mx(p-n) (   6968.165  4428.938    42.000) pv lvNearHbeamBigUnit#pvNearRight err 
    NSc::ccr n   3144 p   2968 mn(n-p) (   4675.837  5417.750    42.000) mx(p-n) (   7424.370  4370.000    42.000) pv lvNearHbeamBigUnit#pvNearRight err 
    NSc::ccr n   3145 p   2968 mn(n-p) (   1851.244  3537.688    42.000) mx(p-n) (  10322.922  6158.062    42.000) pv lvNearHbeamBigUnit#pvNearRight err 
    NSc::ccr n   3146 p   2968 mn(n-p) (   1710.129  3099.875    42.000) mx(p-n) (  10464.037  6595.875    42.000) pv lvNearHbeamBigUnit#pvNearRight err 
    NSc::ccr n   3147 p      1 mn(n-p) (  25611.527 24722.188 14700.000) mx(p-n) (  25611.527 24722.188 25000.000) pv lvNearSiteRock#pvNearHallBot0x err 
    NSc::ccr n   3148 p   3147 mn(n-p) (    414.836   414.875   300.000) mx(p-n) (    414.838   414.875     0.000) pv lvNearHallBot#pvNearPoolDead0x err ZMAX_CO 
    NSc::ccr n   3149 p   3148 mn(n-p) (    116.156   116.125    84.000) mx(p-n) (    116.155   116.125     0.000) pv lvNearPoolDead#pvNearPoolLiner err ZMAX_CO 
    NSc::ccr n   3150 p   3149 mn(n-p) (      0.000     0.000     4.000) mx(p-n) (      0.000     0.000     0.000) pv lvNearPoolLiner#pvNearPoolOWS0 err XMIN_CO YMIN_CO XMAX_CO YMAX_CO ZMAX_CO 
    NSc::ccr n   3151 p   3150 mn(n-p) (   1388.324  1388.312  1000.000) mx(p-n) (   1388.325  1388.312     0.000) pv lvNearPoolOWS#pvNearPoolCurtai err ZMAX_CO 
    NSc::ccr n   3152 p   3151 mn(n-p) (      0.000     0.000     4.000) mx(p-n) (      0.000     0.000     0.000) pv lvNearPoolCurtain#pvNearPoolIW err XMIN_CO YMIN_CO XMAX_CO YMAX_CO ZMAX_CO 
    NSc::ccr n   3153 p   3152 mn(n-p) (   1676.879  6536.250  1408.000) mx(p-n) (   4795.783  1715.125  1490.000) pv lvNearPoolIWS#pvNearADE10xc2cf err 
    NSc::ccr n   3154 p   3153 mn(n-p) (    345.697   345.688    10.000) mx(p-n) (    345.698   345.688  1000.000) pv           lvADE#pvSST0xc128d90 err 
    NSc::ccr n   3155 p   3154 mn(n-p) (     16.594    16.625    30.000) mx(p-n) (     16.594    16.625    15.000) pv           lvSST#pvOIL0xc241510 err 
    NSc::ccr n   3156 p   3155 mn(n-p) (    619.492   619.500   460.000) mx(p-n) (    619.492   619.500   400.379) pv           lvOIL#pvOAV0xbf8f638 err 
    NSc::ccr n   3157 p   3156 mn(n-p) (     80.201    80.188    18.000) mx(p-n) (     80.202    80.188     0.000) pv           lvOAV#pvLSO0xbf8e120 err ZMAX_CO 
    NSc::ccr n   3158 p   3157 mn(n-p) (    576.625   576.625   442.000) mx(p-n) (    576.625   576.625   460.182) pv           lvLSO#pvIAV0xc2d0348 err 
    NSc::ccr n   3159 p   3158 mn(n-p) (     20.742    20.750    15.000) mx(p-n) (     20.742    20.750     0.000) pv           lvIAV#pvGDS0xbf6ab00 err ZMAX_CO 
    NSc::ccr n   3160 p   3158 mn(n-p) (   1353.928  1009.250  3129.720) mx(p-n) (   2887.104  3231.750   -44.720) pv   lvIAV#pvOcrGdsInIAV0xbf6b0e0 err ZMAX_OUT 
    NSc::ccr n   3161 p   3157 mn(n-p) (   2533.279  2533.250  3616.439) mx(p-n) (   2533.278  2533.250   349.621) pv     lvLSO#pvIavTopHub0xc34e6e8 err 
    NSc::ccr n   3162 p   3157 mn(n-p) (   2533.279  2533.250  3727.000) mx(p-n) (   2533.278  2533.250   319.621) pv lvLSO#pvCtrGdsOflBotClp0xc2ce2 err 
    NSc::ccr n   3163 p   3157 mn(n-p) (   2695.758  2695.750  3757.000) mx(p-n) (   2695.757  2695.750     0.000) pv lvLSO#pvCtrGdsOflTfbInLso0xc2c err ZMAX_CO 
    NSc::ccr n   3164 p   3157 mn(n-p) (   2697.141  2697.125  3616.440) mx(p-n) (   2697.140  2697.125     0.000) pv lvLSO#pvCtrGdsOflInLso0xbf7425 err 
    NSc::ccr n   3165 p   3157 mn(n-p) (   1766.689  1422.000  3542.000) mx(p-n) (   3299.868  3644.500   349.621) pv     lvLSO#pvOcrGdsPrt0xbf6d0d0 err 
    NSc::ccr n   3166 p   3157 mn(n-p) (   1766.689  1422.000  3727.000) mx(p-n) (   3299.868  3644.500   319.621) pv  lvLSO#pvOcrGdsBotClp0xbfa1610 err 
    NSc::ccr n   3167 p   3157 mn(n-p) (   1666.207  1584.500  3907.798) mx(p-n) (   2442.429  2740.688    18.025) pv lvLSO#pvOcrGdsTfbInLso0xbfa181 err 
    NSc::ccr n   3168 p   3157 mn(n-p) (   1930.553  1585.875  3800.298) mx(p-n) (   3463.729  3808.375    18.025) pv   lvLSO#pvOcrGdsInLso0xbf6d280 err 
    NSc::ccr n   3169 p   3157 mn(n-p) (   2774.027  1062.938     0.000) mx(p-n) (   1643.136  2811.062  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs# err ZMIN_CO 
    NSc::ccr n   3170 p   3157 mn(n-p) (   2833.238  2300.812     0.000) mx(p-n) (    797.491  2737.188  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3171 p   3157 mn(n-p) (   2811.082  2774.000     0.000) mx(p-n) (   1062.991  1643.125  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3172 p   3157 mn(n-p) (   2737.217  2833.250     0.000) mx(p-n) (   2300.790   797.500  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3173 p   3157 mn(n-p) (   1643.137  2811.062     0.000) mx(p-n) (   2774.026  1062.938  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3174 p   3157 mn(n-p) (    797.492  2737.188     0.000) mx(p-n) (   2833.237  2300.812  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3175 p   3157 mn(n-p) (   1062.992  1643.125     0.000) mx(p-n) (   2811.081  2774.000  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3176 p   3157 mn(n-p) (   2300.791   797.500     0.000) mx(p-n) (   2737.216  2833.250  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3177 p   3157 mn(n-p) (   2602.420  2602.438     0.000) mx(p-n) (   2602.419  2602.438  3879.621) pv     lvLSO#pvOavBotHub0xbf21f78 err ZMIN_CO 
    NSc::ccr n   3178 p   3157 mn(n-p) (   2774.025  1322.438   242.000) mx(p-n) (   1810.978  2811.062  3634.621) pv lvLSO#pvIavBotRibs#IavBotRibs# err 
    NSc::ccr n   3179 p   3157 mn(n-p) (   2833.236  2365.562   242.000) mx(p-n) (   1099.626  2737.188  3634.621) pv lvLSO#pvIavBotRibs#IavBotRibs: err 
    NSc::ccr n   3180 p   3157 mn(n-p) (   2811.082  2774.000   242.000) mx(p-n) (   1322.437  1811.000  3634.621) pv lvLSO#pvIavBotRibs#IavBotRibs: err 
    ...
    NSc::ccr n   3192 p   3157 mn(n-p) (   1248.844  2737.188  3542.000) mx(p-n) (   2833.237  2397.562   425.621) pv lvLSO#pvIavTopRibs#IavRibs:5#I err 
    NSc::ccr n   3193 p   3157 mn(n-p) (   1450.566  1893.875  3542.000) mx(p-n) (   2811.081  2774.000   425.621) pv lvLSO#pvIavTopRibs#IavRibs:6#I err 
    NSc::ccr n   3194 p   3157 mn(n-p) (   2397.553  1248.812  3542.000) mx(p-n) (   2737.216  2833.188   425.621) pv lvLSO#pvIavTopRibs#IavRibs:7#I err 
    NSc::ccr n   3195 p   3156 mn(n-p) (   1985.172  1640.500  3993.311) mx(p-n) (   3518.350  3863.000    -5.000) pv lvOAV#pvOcrGdsLsoInOav0xbfa3df err ZMAX_OUT 
    NSc::ccr n   3196 p   3195 mn(n-p) (     24.199    24.188     0.000) mx(p-n) (     24.199    24.188     0.000) pv lvOcrGdsLsoInOav#pvOcrGdsTfbIn err ZMIN_CO ZMAX_CO 
    NSc::ccr n   3197 p   3196 mn(n-p) (      1.383     1.375     0.000) mx(p-n) (      1.383     1.375     0.000) pv lvOcrGdsTfbInOav#pvOcrGdsInOav err ZMIN_CO ZMAX_CO 
    NSc::ccr n   3198 p   3156 mn(n-p) (   3758.264  4210.750  3993.311) mx(p-n) (   1745.258  1292.750    -5.000) pv lvOAV#pvOcrCalLsoInOav0xbfa3eb err ZMAX_OUT 
    NSc::ccr n   3199 p   3155 mn(n-p) (   4784.367  1475.375   625.500) mx(p-n) (   1746.629  5044.688  4125.500) pv lvOIL#pvAdPmtArray#pvAdPmtArra err 
    NSc::ccr n   3200 p   3199 mn(n-p) (      4.229     5.375     3.000) mx(p-n) (      5.201     4.250     3.000) pv lvPmtHemi#pvPmtHemiVacuum0xc13 err 
    NSc::ccr n   3201 p   3200 mn(n-p) (    -22.084   106.500   -29.000) mx(p-n) (     84.531   -18.812   -29.000) pv lvPmtHemiVacuum#pvPmtHemiCatho err XMIN_OUT ZMIN_OUT YMAX_OUT ZMAX_OUT 
    NSc::ccr n   3202 p   3200 mn(n-p) (     38.238   102.438     0.000) mx(p-n) (     87.172    44.875     0.000) pv lvPmtHemiVacuum#pvPmtHemiBotto err ZMIN_CO ZMAX_CO 
    NSc::ccr n   3203 p   3200 mn(n-p) (    136.438    46.375    71.500) mx(p-n) (     54.449   142.688    71.500) pv lvPmtHemiVacuum#pvPmtHemiDynod err 
    NSc::ccr n   3204 p   3155 mn(n-p) (   4825.814  1639.250   621.500) mx(p-n) (   1885.295  5094.375  4121.500) pv lvOIL#pvAdPmtArray#pvAdPmtArra err 
    NSc::ccr n   3205 p   3155 mn(n-p) (   5188.022  1940.500   625.500) mx(p-n) (   1329.981  4601.938  4125.500) pv lvOIL#pvAdPmtArray#pvAdPmtArra err 
    NSc::ccr n   3206 p   3205 mn(n-p) (      4.173     5.062     3.000) mx(p-n) (      5.408     4.188     3.000) pv lvPmtHemi#pvPmtHemiVacuum0xc13 err 
    NSc::ccr n   3207 p   3206 mn(n-p) (    -16.468    69.500   -29.000) mx(p-n) (    118.938   -23.875   -29.000) pv lvPmtHemiVacuum#pvPmtHemiCatho err XMIN_OUT ZMIN_OUT YMAX_OUT ZMAX_OUT 
    NSc::ccr n   3208 p   3206 mn(n-p) (     48.564    76.375     0.000) mx(p-n) (    110.712    33.500     0.000) pv lvPmtHemiVacuum#pvPmtHemiBotto err ZMIN_CO ZMAX_CO 
    NSc::ccr n   3209 p   3206 mn(n-p) (    144.729    58.875    71.500) mx(p-n) (     40.601   130.688    71.500) pv lvPmtHemiVacuum#pvPmtHemiDynod err 
    NSc::ccr n   3210 p   3155 mn(n-p) (   5242.260  2061.375   621.500) mx(p-n) (   1507.689  4637.625  4121.500) pv lvOIL#pvAdPmtArray#pvAdPmtArra err 
    ...
    NSc::ccr n  12225 p   3147 mn(n-p) (  11628.265  1794.938   150.000) mx(p-n) (   2774.523 15480.688   150.000) pv lvNearHallBot#pvNearHallRadSla err 
    NSc::ccr n  12226 p   3147 mn(n-p) (  14979.191  4151.750   150.000) mx(p-n) (   1753.470 11326.125   150.000) pv lvNearHallBot#pvNearHallRadSla err 
    NSc::ccr n  12227 p   3147 mn(n-p) (  10443.004  8369.250   150.000) mx(p-n) (   1794.919  2774.500   150.000) pv lvNearHallBot#pvNearHallRadSla err 
    NSc::ccr n  12228 p   3147 mn(n-p) (   6288.400 16757.875   150.000) mx(p-n) (   7410.776  1753.500   150.000) pv lvNearHallBot#pvNearHallRadSla err 
    NSc::ccr n  12229 p   3147 mn(n-p) (    414.836   414.875  -150.000) mx(p-n) (    414.838   414.875 10150.000) pv lvNearHallBot#pvNearHallRadSla err ZMIN_OUT 
    2017-06-27 20:45:11.361 INFO  [1538289] [NScene::check_containment@506] NScene::check_containment verbosity 1 tot 12230 err 3491 err/tot       0.29



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






