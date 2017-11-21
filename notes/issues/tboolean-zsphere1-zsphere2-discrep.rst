tboolean-zsphere1-zsphere2-discrep
=====================================

::

    tboolean-;tboolean-zsphere1 --okg4 
    tboolean-;tboolean-zsphere2 --okg4 


Observations

* replacing torch with emitconfig, still discrep

* replacing zsphere with sphere OR box get agreement : so not "1st order" material or torch issue 
    
* eyballing suggests G4Sphere with theta range not giving the flat endcaps I expect,
  
* make start at getting G4RayTracer to help...

  * :doc:`G4TheRayTracer.rst` 


* anaEvent has SDF checking ... how to apply to Opticks G4 evt ?

* **SMOKING GUN** from perusal of G4Sphere.cc


Lessons/TODO
------------------

Automate anaEvent SDF checking, to find such issues quicker, 
for single primitive in container the recipe is:

1. use perfectSpecularSurface on the obj (to easily distinguish from container with perfectAbsorbSurface)
2. switch off scattering/absorption/reemission "--nosc --noab --nore"
3. emitconfig from all sheets of box container:  sheetmask (0b11-1111 = 0x3f) 
4. SDF checking of rpost_(1) of "TO SR SA"   
5. as using emitconfig input photons with sc/ab/re switched off 
   can go one step further and check equality of intersect positions
     


SMOKING GUN : G4Sphere with THETA RANGE IS CONICAL SECTOR NOT SEGMENT 
------------------------------------------------------------------------

* i was expecting "segment" to mean segment with flat endcaps, not conical sector 

* https://en.wikipedia.org/wiki/Spherical_segment
* https://en.wikipedia.org/wiki/Spherical_sector

::

     g4-;g4-cls G4Sphere


     37 //   A G4Sphere is, in the general case, a section of a spherical shell,
     38 //   between specified phi and theta angles
     39 //
     40 //   The phi and theta segments are described by a starting angle,
     41 //   and the +ve delta angle for the shape.
     42 //   If the delta angle is >=2*pi, or >=pi the shape is treated as
     43 //   continuous in phi or theta respectively.
     44 //
     45 //   Theta must lie between 0-pi (incl).
     46 //
     47 //   Member Data:
     48 //
     49 //   fRmin  inner radius
     50 //   fRmax  outer radius
     51 //
     52 //   fSPhi  starting angle of the segment in radians
     53 //   fDPhi  delta angle of the segment in radians
     54 //
     55 //   fSTheta  starting angle of the segment in radians
     56 //   fDTheta  delta angle of the segment in radians
     57 //
     58 //     
     59 //   Note:
     60 //      Internally fSPhi & fDPhi are adjusted so that fDPhi<=2PI,
     61 //      and fDPhi+fSPhi<=2PI. This enables simpler comparisons to be
     62 //      made with (say) Phi of a point.


::

    g4-;g4-cls G4Sphere

    1354   // Theta segment intersection
    1355 
    1356   if ( !fFullThetaSphere )
    1357   {
    1358 
    1359     // Intersection with theta surfaces
    1360     // Known failure cases:
    1361     // o  Inside tolerance of stheta surface, skim
    1362     //    ~parallel to cone and Hit & enter etheta surface [& visa versa]
    1363     //
    1364     //    To solve: Check 2nd root of etheta surface in addition to stheta
    1365     //
    1366     // o  start/end theta is exactly pi/2 
    1367     // Intersections with cones
    1368     //
    1369     // Cone equation: x^2+y^2=z^2tan^2(t)
    1370     //
    1371     // => (px+svx)^2+(py+svy)^2=(pz+svz)^2tan^2(t)
    1372     //
    1373     // => (px^2+py^2-pz^2tan^2(t))+2sd(pxvx+pyvy-pzvztan^2(t))
    1374     //       + sd^2(vx^2+vy^2-vz^2tan^2(t)) = 0
    1375     //
    1376     // => sd^2(1-vz^2(1+tan^2(t))+2sd(pdotv2d-pzvztan^2(t))+(rho2-pz^2tan^2(t))=0
    1377 
    1378     if (fSTheta)
    1379     {
    1380       dist2STheta = rho2 - p.z()*p.z()*tanSTheta2 ;
    1381     }
    1382     else
    1383     {
    1384       dist2STheta = kInfinity ;
    1385     }
    1386     if ( eTheta < pi )
    1387     {







anaEvent SDF checking intersectcs
----------------------------------



Eyeballing
-------------

Eyeball the simulations:

* orthographic (d-key), point photons (p-key several times), mat1 coloring (m-key several times) gives a precise view of whats happening 

::

   tboolean-;tboolean-zsphere1 --load 
       # endcaps as intersected appear in expected place 
       # emitconfig photons have disc of decreased density, prior to intersect ? 

   tboolean-;tboolean-zsphere1 --load --vizg4
       # endcaps as intersected appear as back to back cones, touching at apex 







Simplify : give zsphere perfectSpecularSurface and switch off absorb/scatter
-------------------------------------------------------------------------------

::

    #testobj.boundary = "Vacuum///GlassSchottF2" 
    testobj.boundary = "Vacuum/perfectSpecularSurface//GlassSchottF2"     

    tboolean-;tboolean-zsphere1 --okg4 --noab --nosc 

Constrains photons to two possible histories, which match (ignore the 1/100000)::

    [2017-11-21 12:16:13,572] p2716 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,torch,tboolean-zsphere1)  None 0 
    A tboolean-zsphere1/torch/  1 :  20171121-1214 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/1/fdom.npy () 
    B tboolean-zsphere1/torch/ -1 :  20171121-1214 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum/perfectSpecularSurface//GlassSchottF2
    /tmp/blyth/opticks/tboolean-zsphere1--
    .                seqhis_ana  1:tboolean-zsphere1   -1:tboolean-zsphere1        c2        ab        ba 
    .                             100000    100000         0.00/1 =  0.00  (pval:0.996 prob:0.004)  
    0000               8d     80300     80300             0.00        1.000 +- 0.004        1.000 +- 0.004  [2 ] TO SA
    0001              8ad     19700     19699             0.00        1.000 +- 0.007        1.000 +- 0.007  [3 ] TO SR SA
    0002            8caad         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO SR SR BT SA
    .                             100000    100000         0.00/1 =  0.00  (pval:0.996 prob:0.004)  
    .                pflags_ana  1:tboolean-zsphere1   -1:tboolean-zsphere1        c2        ab        ba 
    .                             100000    100000         0.00/1 =  0.00  (pval:0.996 prob:0.004)  
    0000             1080     80300     80300             0.00        1.000 +- 0.004        1.000 +- 0.004  [2 ] TO|SA
    0001             1280     19700     19699             0.00        1.000 +- 0.007        1.000 +- 0.007  [3 ] TO|SR|SA
    0002             1a80         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|SR|SA
    .                             100000    100000         0.00/1 =  0.00  (pval:0.996 prob:0.004)  
    .                seqmat_ana  1:tboolean-zsphere1   -1:tboolean-zsphere1        c2        ab        ba 
    .                             100000    100000         0.00/1 =  0.00  (pval:0.996 prob:0.004)  
    0000               12     80300     80300             0.00        1.000 +- 0.004        1.000 +- 0.004  [2 ] Vm Rk
    0001              122     19700     19699             0.00        1.000 +- 0.007        1.000 +- 0.007  [3 ] Vm Vm Rk
    0002            12322         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Vm Vm F2 Vm Rk
    .                             100000    100000         0.00/1 =  0.00  (pval:0.996 prob:0.004)  
                /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/1 09d00c198cb3c30093ab00a545f367dc 11dd613deda41f648eadbb48358231d9  100000    -1.0000 INTEROP_MODE 
    {u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons=100000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'resolution': u'40', u'emit': -1}
    [2017-11-21 12:16:13,578] p2716 {/Users/blyth/opticks/ana/tboolean.py:25} INFO - early exit as non-interactive
    simon:ana blyth$ 


Although histories match, vizg4 positions do not : there is a single cone (apex at origin) 

::

   tboolean-;tboolean-zsphere1 --load --vizg4



Check intersect positions in ipython
---------------------------------------

::


   tboolean-;tboolean-zsphere1 --okg4 --noab --nosc   # only 2 histories, misses the obj or reflects off it 


Extreme level of history agreement is because the photons are input/emitconfig photons
which are exactly the same for both simulations

::

    tboolean-;tboolean-zsphere1-ip 


    A tboolean-zsphere1/torch/  1 :  20171121-1430 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/1/fdom.npy () 
    B tboolean-zsphere1/torch/ -1 :  20171121-1430 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum/perfectSpecularSurface//GlassSchottF2
    /tmp/blyth/opticks/tboolean-zsphere1--
    .                seqhis_ana  1:tboolean-zsphere1   -1:tboolean-zsphere1        c2        ab        ba 
    .                             100000    100000         0.00/1 =  0.00  (pval:0.996 prob:0.004)  
    0000      80300     80300             0.00  TO SA
    0001      19700     19699             0.00  TO SR SA
    0002          0         1             0.00  TO SR SR BT SA



    In [5]: ab.a.rpost_(slice(0,2))
    Out[5]: 
    A()sliced
    A([[[-965.6548, -777.1673, -999.9002,    0.2002],
            [-965.6548, -777.1673,  999.9919,    6.8709]],

           [[ -37.2393, -655.3683, -999.9002,    0.2002],
            [ -37.2393, -655.3683,  999.9919,    6.8709]],

           [[ 833.0414, -503.3563, -999.9002,    0.2002],
            [ 833.0414, -503.3563,  999.9919,    6.8709]],

           ..., 
           [[-772.2489,  876.085 , -999.9002,    0.2002],
            [-772.2489,  876.085 ,  999.9919,    6.8709]],

           [[ -84.7125, -246.1641, -999.9002,    0.2002],
            [ -84.7125, -246.1641, -200.0045,    2.8681]],

           [[-221.175 ,  762.2593, -999.9002,    0.2002],
            [-221.175 ,  762.2593,  999.9919,    6.8709]]])


    In [6]: ab.sel = "TO SR SA"

    In [7]: ab.his
    Out[7]: 
    .                seqhis_ana  1:tboolean-zsphere1   -1:tboolean-zsphere1        c2        ab        ba 
    .                              19700     19699         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000      19700     19699             0.00  TO SR SA
    .                              19700     19699         0.00/0 =  0.00  (pval:nan prob:nan)  

    In [8]: 


    In [9]: ab.a.rpost_(slice(0,2))
    Out[9]: 
    A()sliced
    A([[[-205.9615, -428.0224, -999.9002,    0.2002],
            [-205.9615, -428.0224, -156.075 ,    3.0146]],

           [[ 224.8408,  -55.8131, -999.9002,    0.2002],
            [ 224.8408,  -55.8131, -200.0045,    2.8681]],

           [[ 309.034 , -117.5221, -999.9002,    0.2002],
            [ 309.034 , -117.5221, -200.0045,    2.8681]],

           ..., 
           [[ 133.8353,  478.6422, -999.9002,    0.2002],
            [ 133.8353,  478.6422,  -54.6522,    3.3528]],

           [[-336.6197, -109.7932, -999.9002,    0.2002],
            [-336.6197, -109.7932, -200.0045,    2.8681]],

           [[ -84.7125, -246.1641, -999.9002,    0.2002],
            [ -84.7125, -246.1641, -200.0045,    2.8681]]])

    In [10]: ab.b.rpost_(slice(0,2))
    Out[10]: 
    A()sliced
    A([[[-205.9615, -428.0224, -999.9002,    0.2002],
            [-205.9615, -428.0224, -156.075 ,    3.0146]],

           [[ 224.8408,  -55.8131, -999.9002,    0.2002],
            [ 224.8408,  -55.8131, -101.1173,    3.1983]],

           [[ 309.034 , -117.5221, -999.9002,    0.2002],
            [ 309.034 , -117.5221, -144.2831,    3.0543]],

           ..., 
           [[ 133.8353,  478.6422, -999.9002,    0.2002],
            [ 133.8353,  478.6422,  -54.6522,    3.3528]],

           [[-336.6197, -109.7932, -999.9002,    0.2002],
            [-336.6197, -109.7932, -154.5475,    3.0201]],

           [[ -84.7125, -246.1641, -999.9002,    0.2002],
            [ -84.7125, -246.1641, -113.6118,    3.1562]]])



Stark difference for G4Sphere intersect::

    In [14]: ab.a.rpost_(1)[:20]
    Out[14]: 
    A()sliced
    A([[-205.9615, -428.0224, -156.075 ,    3.0146],
           [ 224.8408,  -55.8131, -200.0045,    2.8681],
           [ 309.034 , -117.5221, -200.0045,    2.8681],
           [ -86.3316,  359.7454, -200.0045,    2.8681],
           [-165.6368,    0.4888, -200.0045,    2.8681],
           [ -64.5501,  415.4668, -200.0045,    2.8681],
           [-233.5779, -230.2175, -200.0045,    2.8681],
           [ 300.358 ,  337.1696, -200.0045,    2.8681],
           [-344.257 , -179.9338, -200.0045,    2.8681],
           [  12.6167, -398.3593, -200.0045,    2.8681],
           [-254.6872,  110.4347, -200.0045,    2.8681],
           [  57.035 , -229.4537, -200.0045,    2.8681],
           [ 376.2113, -253.0681, -200.0045,    2.8681],
           [-244.6366,  423.6539, -103.3474,    3.1904],
           [ 390.2027,   75.7921, -200.0045,    2.8681],
           [  -6.232 , -435.2931, -200.0045,    2.8681],
           [-116.9722, -176.6039, -200.0045,    2.8681],
           [ 150.7595,  -57.1878, -200.0045,    2.8681],
           [-290.6129, -342.363 , -200.0045,    2.8681],
           [-160.3518, -399.8868, -200.0045,    2.8681]])

    In [15]: ab.b.rpost_(1)[:20]
    Out[15]: 
    A()sliced
    A([[-205.9615, -428.0224, -156.075 ,    3.0146],
           [ 224.8408,  -55.8131, -101.1173,    3.1983],
           [ 309.034 , -117.5221, -144.2831,    3.0543],
           [ -86.3316,  359.7454, -161.4516,    2.9969],
           [-165.6368,    0.4888,  -72.279 ,    3.2942],
           [ -64.5501,  415.4668, -183.508 ,    2.9231],
           [-233.5779, -230.2175, -143.1222,    3.058 ],
           [ 300.358 ,  337.1696, -197.0718,    2.8779],
           [-344.257 , -179.9338, -169.5471,    2.9701],
           [  12.6167, -398.3593, -173.9462,    2.9548],
           [-254.6872,  110.4347, -121.1574,    3.1312],
           [  57.035 , -229.4537, -103.1946,    3.191 ],
           [ 376.2113, -253.0681, -197.8661,    2.8755],
           [-244.6366,  423.6539, -103.3474,    3.1904],
           [ 390.2027,   75.7921, -173.4879,    2.9566],
           [  -6.232 , -435.2931, -189.9844,    2.9017],
           [-116.9722, -176.6039,  -92.4414,    3.227 ],
           [ 150.7595,  -57.1878,  -70.385 ,    3.3003],
           [-290.6129, -342.363 , -196.0026,    2.8816],
           [-160.3518, -399.8868, -188.0293,    2.9078]])








review CSG_ZSPHERE
--------------------

csg.py 
    collecting/serializing param

NCSG::import_primitive
    new nzsphere(make_zsphere(p0,p1,p2))

CMaker::ConvertPrimitive
    new G4Sphere( name, innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);



nzsphere
~~~~~~~~~~~

::

    simon:opticksnpy blyth$ NZSphereTest 
    2017-11-21 11:37:26.885 INFO  [6067821] [test_deltaTheta@112] test_deltaTheta radius 10 z1 -5 z2 5 startTheta 1.0472 endTheta 2.0944 deltaTheta 1.0472
    2017-11-21 11:37:26.886 INFO  [6067821] [test_deltaTheta@112] test_deltaTheta radius 500 z1 -200 z2 200 startTheta 1.15928 endTheta 1.98231 deltaTheta 0.823034
    simon:opticksnpy blyth$ 



history difference
----------------------

Looks like reflection difference with the symmetrical z1:z2 -200:200::

    simon:opticksnpy blyth$ tboolean-;tboolean-zsphere1--

    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  
    args = opticks_main(csgpath="/tmp/blyth/opticks/tboolean-zsphere1--")
    CSG.kwa = dict(poly="IM", resolution="40", verbosity="0", ctrl="0" )

    container = CSG("box", param=[0,0,0,1000], boundary="Rock//perfectAbsorbSurface/Vacuum", poly="MC", nx="20" )

    zsphere = CSG("zsphere", param=[0,0,0,500], param1=[-200,200,0,0],param2=[0,0,0,0],  boundary="Vacuum///GlassSchottF2" )

    CSG.Serialize([container, zsphere], args.csgpath )


    [2017-11-20 21:00:37,547] p90143 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,torch,tboolean-zsphere1)  None 0 
    A tboolean-zsphere1/torch/  1 :  20171120-2059 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/1/fdom.npy () 
    B tboolean-zsphere1/torch/ -1 :  20171120-2059 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-zsphere1--
    .                seqhis_ana  1:tboolean-zsphere1   -1:tboolean-zsphere1        c2        ab        ba 
    .                             100000    100000      3442.65/12 = 286.89  (pval:0.000 prob:1.000)  
    0000             8ccd     88627     82520           217.91        1.074 +- 0.004        0.931 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      5685      5776             0.72        0.984 +- 0.013        1.016 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5162      8007           614.63        0.645 +- 0.009        1.551 +- 0.017  [5 ] TO BT BR BT SA
    0003           8cbbcd       301      2193          1435.31        0.137 +- 0.008        7.286 +- 0.156  [6 ] TO BT BR BR BT SA
    0004            86ccd        61        69             0.49        0.884 +- 0.113        1.131 +- 0.136  [5 ] TO BT BT SC SA
    0005              86d        33        35             0.06        0.943 +- 0.164        1.061 +- 0.179  [3 ] TO SC SA
    0006              4cd        32        18             3.92        1.778 +- 0.314        0.562 +- 0.133  [3 ] TO BT AB
    0007            8c6cd        17         8             0.00        2.125 +- 0.515        0.471 +- 0.166  [5 ] TO BT SC BT SA
    0008          8cbbbcd        12       938           902.61        0.013 +- 0.004       78.167 +- 2.552  [7 ] TO BT BR BR BR BT SA
    0009          8cc6ccd        10         7             0.00        1.429 +- 0.452        0.700 +- 0.265  [7 ] TO BT BT SC BT BT SA
    0010          8cbb6cd         5         3             0.00        1.667 +- 0.745        0.600 +- 0.346  [7 ] TO BT SC BR BR BT SA
    0011           8cb6cd         5         9             0.00        0.556 +- 0.248        1.800 +- 0.600  [6 ] TO BT SC BR BT SA
    0012             4ccd         5         9             0.00        0.556 +- 0.248        1.800 +- 0.600  [4 ] TO BT BT AB
    0013           86cbcd         4        10             0.00        0.400 +- 0.200        2.500 +- 0.791  [6 ] TO BT BR BT SC SA
    0014            8cc6d         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [5 ] TO SC BT BT SA
    0015           8b6ccd         4         1             0.00        4.000 +- 2.000        0.250 +- 0.250  [6 ] TO BT BT SC BR SA
    0016               4d         4         2             0.00        2.000 +- 1.000        0.500 +- 0.354  [2 ] TO AB
    0017           8c6bcd         3         1             0.00        3.000 +- 1.732        0.333 +- 0.333  [6 ] TO BT BR SC BT SA
    0018         8cbbb6cd         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [8 ] TO BT SC BR BR BR BT SA
    0019       8cbbbbb6cd         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT SC BR BR BR BR BR BT SA
    .                             100000    100000      3442.65/12 = 286.89  (pval:0.000 prob:1.000)  




But with offset z1:z2 100:200 get agreement::

    simon:opticksnpy blyth$ tboolean-;tboolean-zsphere2--

    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  
    args = opticks_main(csgpath="/tmp/blyth/opticks/tboolean-zsphere2--")
    CSG.kwa = dict(poly="IM", resolution="40", verbosity="0", ctrl="0" )

    container = CSG("box", param=[0,0,0,1000], boundary="Rock//perfectAbsorbSurface/Vacuum", poly="MC", nx="20" )

    zsphere = CSG("zsphere", param=[0,0,0,500], param1=[100,200,0,0],param2=[0,0,0,0],  boundary="Vacuum///GlassSchottF2" )

    CSG.Serialize([container, zsphere], args.csgpath )

    simon:opticksnpy blyth$ 
    simon:opticksnpy blyth$ 

    [2017-11-20 21:02:58,439] p90174 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,torch,tboolean-zsphere2)  None 0 
    A tboolean-zsphere2/torch/  1 :  20171120-2100 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere2/torch/1/fdom.npy () 
    B tboolean-zsphere2/torch/ -1 :  20171120-2100 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-zsphere2/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-zsphere2--
    .                seqhis_ana  1:tboolean-zsphere2   -1:tboolean-zsphere2        c2        ab        ba 
    .                             100000    100000         6.70/6 =  1.12  (pval:0.349 prob:0.651)  
    0000             8ccd     88645     88772             0.09        0.999 +- 0.003        1.001 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      5685      5709             0.05        0.996 +- 0.013        1.004 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5168      5008             2.52        1.032 +- 0.014        0.969 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       301       301             0.00        1.000 +- 0.058        1.000 +- 0.058  [6 ] TO BT BR BR BT SA
    0004            86ccd        86        69             1.86        1.246 +- 0.134        0.802 +- 0.097  [5 ] TO BT BT SC SA
    0005              86d        33        27             0.60        1.222 +- 0.213        0.818 +- 0.157  [3 ] TO SC SA
    0006          8cc6ccd        14         7             0.00        2.000 +- 0.535        0.500 +- 0.189  [7 ] TO BT BT SC BT BT SA
    0007          8cbbbcd        12        19             1.58        0.632 +- 0.182        1.583 +- 0.363  [7 ] TO BT BR BR BR BT SA
    0008             4ccd         8        15             0.00        0.533 +- 0.189        1.875 +- 0.484  [4 ] TO BT BT AB
    0009              4cd         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [3 ] TO BT AB
    0010            8cc6d         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [5 ] TO SC BT BT SA
    0011           86cbcd         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [6 ] TO BT BR BT SC SA
    0012           8b6ccd         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [6 ] TO BT BT SC BR SA
    0013       bbbbbc6ccd         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT SC BT BR BR BR BR BR
    0014               4d         4         8             0.00        0.500 +- 0.250        2.000 +- 0.707  [2 ] TO AB




