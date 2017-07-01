
lvid 66 : unbelievable big impingement  : trapezoid/convexpolyhedron machinery bug
======================================================================================


Parametric surface points from nodes using lvid 66 are impinging 1025mm into parent volume...

* parametric bug ?
* bbox bug ?

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



