
tlens-concave-ignored-due-to-cylinder-axial-photon-intersect-failure
=======================================================================

issue
-------

tlens-convex appears as expected, focussing the parallel beam, 
but photons are ignoring tlens-concave : they sail thru the lens without intersect.


FIXED : thru-going cylinder axial intersects 
-----------------------------------------------------

* cylinder axial intersects was failing to pick the other intersect 
  (via tmin cutting) for thru going axial photons 



tlens-convex : CSG intersection of two spheres
------------------------------------------------

::

    simon:optixrap blyth$ tlens-convex--

    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  

    args = opticks_main(csgpath="/tmp/blyth/opticks/tlens-convex--", testobject="Vacuum///GlassSchottF2", container="Rock//perfectAbsorbSurface/Vacuum" )

    container = CSG("box", param=[-1,1,0,700], boundary=args.container, poly="MC", nx="20" )

    CSG.boundary = args.testobject
    CSG.kwa = dict(poly="IM", resolution="40", verbosity="0", ctrl="0" )

    al = CSG("sphere", param=[0,0,-600,641.2])   
    ar = CSG("sphere", param=[0,0, 600,641.2])
    lens = CSG("intersection", left=al, right=ar )

    CSG.Serialize([container, lens ], args.csgpath )


tlens-concave-- : CSG difference of a cylinder and two spheres
-----------------------------------------------------------------

* subtracting two big spheres from the cyclinder

::

    import logging 
    log = logging.getLogger(__name__)
    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  

    args = opticks_main(csgpath="/tmp/blyth/opticks/tlens-concave--", testobject="Vacuum///GlassSchottF2", container="Rock//perfectAbsorbSurface/Vacuum" )

    cr = 300.
    cz = 100.

    sz = (cz*cz + cr*cr)/(2.*cz )
    sr = sz

    log.info( " cr %s cz %s sr %s sz %s " % (cr,cz,sr,sz ))


    container = CSG("box", param=[0,0,0,sz], boundary=args.container, poly="MC", nx="20" )
    log.info(" container.boundary : %s " % container.boundary )

    CSG.boundary = args.testobject
    CSG.kwa = dict(poly="IM", resolution="50", verbosity="0", ctrl="0" )

    cy = CSG("cylinder", param=[0,0,0,cr], param1=[-cz,cz,0,0])   
    ar = CSG("sphere", param=[0,0, sz,sr], complement=False)
    al = CSG("sphere", param=[0,0,-sz,sr], complement=False)

    lens = cy - ar - al 
    #lens = cy - al 
    #lens = cy  
    #lens = al  


    log.info(" lens.boundary : %s " % lens.boundary )


    CSG.Serialize([container, lens ], args.csgpath )

    """

              (-cz,cr)      (+cz,cr)
                   +---------+ 
                   |    |    |
                   |    |    |
                   |    |    |
                   |    |    |                                
          ---------|----0----|-----------+------------------   --> Z
                   |    |    |         (sz
                   |    |    |
                   |    |    |
                   |    |    |
                   +---------+ 

        Find parameters of sphere that goes thru points (0,0) and (cz,cr)

           sz = sr 

          (sz - cz)^2 + cr^2 = sr^2

            sz^2 - 2 sz cz + cz^2 + cr^2 = sz^2
     
                    sz = (cz^2 + cr^2)/(2*cz)


    """



possible causes
------------------

* failure to label geometry with boundary ? hmm but convex manages

* incorrect bounds : calculating the bounds of a CSG shape is non-trivial
  suspect this goes wrong with CSG differences resulting in overly large
  CSG bounds 

  * tried replacing the subtraction of big spheres with subtraction of zspheres
    to prevent : seemed to make no difference

  
* huh : replacing the cylinder with a box3 to make a square lens behaving 
  as expected : smth funny with cylinder endcaps ?


* huh : back to cylinder see more normal behaviour with off axis photons 
  
  * so something is wrong with axial photon intersects onto cylinder endcaps 
    but only when cylinder within CSG tree : red herring actually 
    no need for CSG, just the issue is easily hidden by other non-axial photon intersects

  * tboolean-cy appears to work as expected : so CSG tree somehow involved :
   

* to see the issue have to shoot only axial photons otherwise
  lack of intersects is hidden, but CSG tree is implicated because 
  for a cylinder with sphere chopped out of endcap : only the axial 
  fail to intersect with the sphere 
  

 



Issue reproduced with single cylinder and axial photons
---------------------------------------------------------

::

   tboolean-;tboolean-cyd


::

    2344 
    2345 
    2346 #tboolean-cyd-torch-(){ tboolean-torchconfig-disc 1,1,599 ; }  ## non-axial works
    2347 tboolean-cyd-torch-(){ tboolean-torchconfig-disc 0,0,599 90 ; }  ## axial fails to intersect
    2348 tboolean-cyd(){ TESTCONFIG=$($FUNCNAME-) TORCHCONFIG=$($FUNCNAME-torch-) tboolean-- $* ; }
    2349 tboolean-cyd-(){  $FUNCNAME- | python $* ; }
    2350 tboolean-cyd--(){ cat << EOP 
    2351 import numpy as np
    2352 from opticks.ana.base import opticks_main
    2353 from opticks.analytic.csg import CSG  
    2354 args = opticks_main(csgpath="$TMP/$FUNCNAME")
    2355 
    2356 CSG.boundary = args.testobject
    2357 CSG.kwa = dict(verbosity="1", poly="IM", resolution="4" )
    2358 
    2359 container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="IM", resolution="4", verbosity="0" )
    2360 
    2361 ra = 200 
    2362 z1 = -100
    2363 z2 = 100
    2364 delta = 0.1
    2365 
    2366 a = CSG("cylinder", param=[0,0,0,ra], param1=[z1,z2,0,0] )
    2367 b = CSG("sphere", param=[0,0,z2,ra/2]  )
    2368 
    2369 obj = a - b 
    2370 
    2371 CSG.Serialize([container, obj], args.csgpath )
    2372 
    2373 EOP
    2374 }
    2375 




Arrange a pencil of 10 photons that all miss CSG intersect::

    2017-10-26 14:56:13.296 INFO  [981246] [OPropagator::prelaunch@160] 1 : (0;10,1) prelaunch_times vali,comp,prel,lnch  0.0001 3.4517 0.1311 0.0000
    // csg_intersect_cylinder tmin     0.1000 abc (    0.0000     0.0000 -1360169472.0000) ori (   71.7140    29.2040   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin     0.1000 abc (    0.0000     0.0000 -1301596672.0000) ori (   83.7070    21.2881   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin     0.1000 abc (    0.0000     0.0000 -1587358208.0000) ori (    0.0240    17.7782   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin     0.1000 abc (    0.0000     0.0000 -1286056448.0000) ori (   88.5440     2.9255   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin     0.1000 abc (    0.0000     0.0000 -1300255232.0000) ori (  -81.8082    28.3031   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin     0.1000 abc (    0.0000     0.0000 -1455383040.0000) ori (   31.4812    51.2286   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin     0.1000 abc (    0.0000     0.0000 -1383789056.0000) ori (   58.5882    44.4151   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin     0.1000 abc (    0.0000     0.0000 -1564381696.0000) ori (  -21.0230   -21.1772   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin     0.1000 abc (    0.0000     0.0000 -1447647744.0000) ori (   61.3282     6.9006   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin     0.1000 abc (    0.0000     0.0000 -1433725440.0000) ori (  -62.1522    17.1455   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin   499.1000 abc (    0.0000     0.0000 -1360169472.0000) ori (   71.7140    29.2040   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin   499.1000 abc (    0.0000     0.0000 -1301596672.0000) ori (   83.7070    21.2881   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin   499.1000 abc (    0.0000     0.0000 -1587358208.0000) ori (    0.0240    17.7782   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin   499.1000 abc (    0.0000     0.0000 -1286056448.0000) ori (   88.5440     2.9255   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin   499.1000 abc (    0.0000     0.0000 -1300255232.0000) ori (  -81.8082    28.3031   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin   499.1000 abc (    0.0000     0.0000 -1455383040.0000) ori (   31.4812    51.2286   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin   499.1000 abc (    0.0000     0.0000 -1383789056.0000) ori (   58.5882    44.4151   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin   499.1000 abc (    0.0000     0.0000 -1564381696.0000) ori (  -21.0230   -21.1772   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin   499.1000 abc (    0.0000     0.0000 -1447647744.0000) ori (   61.3282     6.9006   599.0000) dir (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder tmin   499.1000 abc (    0.0000     0.0000 -1433725440.0000) ori (  -62.1522    17.1455   599.0000) dir (    0.0000     0.0000    -1.0000)  
    2017-10-26 14:56:13.310 INFO  [981246] [OContext::launch@322] OContext::launch LAUNCH time: 0.014029

::

    2017-10-26 15:01:31.894 INFO  [982826] [OPropagator::prelaunch@160] 1 : (0;10,1) prelaunch_times vali,comp,prel,lnch  0.0001 3.4215 0.1315 0.0000
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1360169472.0000)  m (   71.7140    29.2040   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1301596672.0000)  m (   83.7070    21.2881   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1587358208.0000)  m (    0.0240    17.7782   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1286056448.0000)  m (   88.5440     2.9255   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1300255232.0000)  m (  -81.8082    28.3031   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1455383040.0000)  m (   31.4812    51.2286   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1383789056.0000)  m (   58.5882    44.4151   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1564381696.0000)  m (  -21.0230   -21.1772   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1447647744.0000)  m (   61.3282     6.9006   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1433725440.0000)  m (  -62.1522    17.1455   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin   499.1000 abc (    0.0000     0.0000 -1360169472.0000)  m (   71.7140    29.2040   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin   499.1000 abc (    0.0000     0.0000 -1301596672.0000)  m (   83.7070    21.2881   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin   499.1000 abc (    0.0000     0.0000 -1587358208.0000)  m (    0.0240    17.7782   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin   499.1000 abc (    0.0000     0.0000 -1286056448.0000)  m (   88.5440     2.9255   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin   499.1000 abc (    0.0000     0.0000 -1300255232.0000)  m (  -81.8082    28.3031   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin   499.1000 abc (    0.0000     0.0000 -1455383040.0000)  m (   31.4812    51.2286   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin   499.1000 abc (    0.0000     0.0000 -1383789056.0000)  m (   58.5882    44.4151   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin   499.1000 abc (    0.0000     0.0000 -1564381696.0000)  m (  -21.0230   -21.1772   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin   499.1000 abc (    0.0000     0.0000 -1447647744.0000)  m (   61.3282     6.9006   699.0000)  n (    0.0000     0.0000    -1.0000)  
    // csg_intersect_cylinder  tmin   499.1000 abc (    0.0000     0.0000 -1433725440.0000)  m (  -62.1522    17.1455   699.0000)  n (    0.0000     0.0000    -1.0000)  
    2017-10-26 15:01:31.908 INFO  [982826] [OContext::launch@322] OContext::launch LAUNCH time: 0.014078


::

    2017-10-26 15:08:52.081 INFO  [985131] [OPropagator::prelaunch@160] 1 : (0;10,1) prelaunch_times vali,comp,prel,lnch  0.0001 3.5851 0.1323 0.0000
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1360167936.0000)  m (   71.7140    29.2040   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1301597184.0000)  m (   83.7070    21.2881   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1587356672.0000)  m (    0.0240    17.7782   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1286056960.0000)  m (   88.5440     2.9255   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1300255744.0000)  m (  -81.8082    28.3031   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1455381504.0000)  m (   31.4812    51.2286   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1383789568.0000)  m (   58.5882    44.4151   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1564382208.0000)  m (  -21.0230   -21.1772   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1447648256.0000)  m (   61.3282     6.9006   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin     0.1000 abc (    0.0000     0.0000 -1433725952.0000)  m (  -62.1522    17.1455   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin     0.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin     0.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin     0.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin     0.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin     0.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin     0.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin     0.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin     0.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin     0.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin     0.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin   500.1000 abc (    0.0000     0.0000 -1360167936.0000)  m (   71.7140    29.2040   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin   500.1000 abc (    0.0000     0.0000 -1301597184.0000)  m (   83.7070    21.2881   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin   500.1000 abc (    0.0000     0.0000 -1587356672.0000)  m (    0.0240    17.7782   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin   500.1000 abc (    0.0000     0.0000 -1286056960.0000)  m (   88.5440     2.9255   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin   500.1000 abc (    0.0000     0.0000 -1300255744.0000)  m (  -81.8082    28.3031   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin   500.1000 abc (    0.0000     0.0000 -1455381504.0000)  m (   31.4812    51.2286   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin   500.1000 abc (    0.0000     0.0000 -1383789568.0000)  m (   58.5882    44.4151   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin   500.1000 abc (    0.0000     0.0000 -1564382208.0000)  m (  -21.0230   -21.1772   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin   500.1000 abc (    0.0000     0.0000 -1447648256.0000)  m (   61.3282     6.9006   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin   500.1000 abc (    0.0000     0.0000 -1433725952.0000)  m (  -62.1522    17.1455   700.0000)  d (    0.0000     0.0000   200.0000)  
    // csg_intersect_cylinder  tmin   500.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin   500.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin   500.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin   500.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin   500.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin   500.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin   500.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin   500.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin   500.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    // csg_intersect_cylinder  tmin   500.1000 tcan   500.0000  md 140000.0000 t_pcap_ax   700.0000 t_qcap_ax   500.0000 
    2017-10-26 15:08:52.096 INFO  [985131] [OContext::launch@322] OContext::launch LAUNCH time: 0.014978




