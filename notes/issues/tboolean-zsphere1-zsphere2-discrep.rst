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
     


testauto shakedown
--------------------

::

    (lldb) p csg->m_meta
    (NParameters *) $2 = 0x000000010c624850
    (lldb) p *(csg->m_meta)
    (NParameters) $3 = {
      m_parameters = size=6 {
        [0] = (first = "ctrl", second = "0")
        [1] = (first = "emitconfig", second = "-1")
        [2] = (first = "verbosity", second = "0")
        [3] = (first = "resolution", second = "40")
        [4] = (first = "emit", second = "-1")
        [5] = (first = "poly", second = "IM")
      }
      m_lines = size=0 {}
    }
    (lldb) bt
    * thread #1: tid = 0x6452a0, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000100375211 libBoostRap.dylib`BConfig::parse(this=0x000000010c63b850) + 497 at BConfig.cc:47
        frame #5: 0x00000001009c6268 libNPY.dylib`NEmitConfig::NEmitConfig(this=0x000000010c63b810, cfg=0x000000010c63b540) + 1032 at NEmitConfig.cpp:34
        frame #6: 0x00000001009c62ad libNPY.dylib`NEmitConfig::NEmitConfig(this=0x000000010c63b810, cfg=0x000000010c63b540) + 29 at NEmitConfig.cpp:35
      * frame #7: 0x00000001009c6a57 libNPY.dylib`NEmitPhotonsNPY::NEmitPhotonsNPY(this=0x000000010c63b7d0, csg=0x000000010c623850, gencode=262144, emitdbg=false) + 135 at NEmitPhotonsNPY.cpp:23
        frame #8: 0x00000001009c76d2 libNPY.dylib`NEmitPhotonsNPY::NEmitPhotonsNPY(this=0x000000010c63b7d0, csg=0x000000010c623850, gencode=262144, emitdbg=false) + 50 at NEmitPhotonsNPY.cpp:31
        frame #9: 0x00000001023380df libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x000000010c63b760, hub=0x000000010a00f8e0) + 255 at OpticksGen.cc:37
        frame #10: 0x000000010233820d libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x000000010c63b760, hub=0x000000010a00f8e0) + 29 at OpticksGen.cc:43
        frame #11: 0x00000001023322f2 libOpticksGeometry.dylib`OpticksHub::init(this=0x000000010a00f8e0) + 242 at OpticksHub.cc:188
        frame #12: 0x0000000102332150 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x000000010a00f8e0, ok=0x00000001097315a0) + 464 at OpticksHub.cc:164
        frame #13: 0x00000001023323ad libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x000000010a00f8e0, ok=0x00000001097315a0) + 29 at OpticksHub.cc:166
        frame #14: 0x0000000104514bbb libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe1c0, argc=28, argv=0x00007fff5fbfe2a0) + 283 at OKG4Mgr.cc:30
        frame #15: 0x0000000104514f53 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe1c0, argc=28, argv=0x00007fff5fbfe2a0) + 35 at OKG4Mgr.cc:41
        frame #16: 0x00000001000132ee OKG4Test`main(argc=28, argv=0x00007fff5fbfe2a0) + 1486 at OKG4Test.cc:56
        frame #17: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 



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

::

    simon:opticks blyth$ opticks-find ::anaEvent\(
    ./ggeo/GGeo.cc:void GGeo::anaEvent(OpticksEvent* evt)
    ./ggeo/GGeoTest.cc:void GGeoTest::anaEvent(OpticksEvent* evt)
    ./ggeo/GScene.cc:void GScene::anaEvent(OpticksEvent* evt)
    ./optickscore/OpticksRun.cc:void OpticksRun::anaEvent()
    ./opticksgeo/OpticksHub.cc:void OpticksHub::anaEvent()
    simon:opticks blyth$ 


OpticksHub::anaEvent is called after propagation if **save** is active::

    simon:opticks blyth$ opticks-find \>anaEvent
    ./ok/OKMgr.cc:                if(!production) m_hub->anaEvent();
    ./okg4/OKG4Mgr.cc:                m_hub->anaEvent();
    ./okop/OpMgr.cc:                if(!production) m_hub->anaEvent();
    ./okop/OpMgr.cc:                if(!production) m_hub->anaEvent();
    ./opticksgeo/OpticksHub.cc:        m_geotest->anaEvent( evt );  
    ./opticksgeo/OpticksHub.cc:        m_gscene->anaEvent( evt ); 
    ./opticksgeo/OpticksHub.cc:        m_ggeo->anaEvent( evt ); 
    ./opticksgeo/OpticksHub.cc:    m_run->anaEvent();
    simon:opticks blyth$ 


::

    636 void GGeoTest::anaEvent(OpticksEvent* evt)
    637 {
    638     int dbgnode = m_ok->getDbgNode();
    639     //NCSG* csg = getTree(dbgnode);
    640 
    641     LOG(info) << "GGeoTest::anaEvent "
    642               << " dbgnode " << dbgnode
    643               << " numTrees " << getNumTrees()
    644               << " evt " << evt
    645               ;
    646 
    647     assert( m_csglist ) ;
    648 
    649     OpticksEventAna ana(m_ok, evt, m_csglist);
    650     ana.dump("GGeoTest::anaEvent");
    651 }


::

        
    tboolean-;tboolean-zsphere1 --okg4 

    2017-11-21 16:34:05.950 INFO  [6188285] [*OpticksEventStat::CreateRecordsNPY@32] OpticksEventStat::CreateRecordsNPY  shape 100000,10,2,4
    2017-11-21 16:34:05.954 INFO  [6188285] [OpticksEventAna::countExcursions@81] OpticksEventAna::countExcursions pho_num 100000 epsilon 0.1 dbgseqhis 0 dbgseqhis                                                 
    2017-11-21 16:34:06.002 INFO  [6188285] [OpticksEventAna::countExcursions@136] OpticksEventAna::countExcursions pho_num 100000 dbgseqhis 0 dbgseqhis                                                  count 0
    2017-11-21 16:34:06.040 INFO  [6188285] [OpticksEventAna::countExcursions@136] OpticksEventAna::countExcursions pho_num 100000 dbgseqhis 0 dbgseqhis                                                  count 0
    2017-11-21 16:34:06.040 INFO  [6188285] [OpticksEventAna::dump@57] GGeoTest::anaEvent OpticksEventAna pho 100000,4,4 seq 100000,1,2
    2017-11-21 16:34:06.040 INFO  [6188285] [OpticksEventStat::dump@89] per-seqhis per-tree counts on NCSG tree surface evt Evt /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/1 20171121_163402 /usr/local/opticks/lib/OKG4Test totmin 2
     seqhis               8d                 TO SA                                            tot  80131 cat (   80131      0 )  frac (   1.000  0.000 ) 
     seqhis              8ad                 TO SR SA                                         tot  19664 cat (   19664      0 )  frac (   1.000  0.000 ) 
     seqhis              86d                 TO SC SA                                         tot    165 cat (     165      0 )  frac (   1.000  0.000 ) 
     seqhis               4d                 TO AB                                            tot     19 cat (       0      0 )  frac (   0.000  0.000 ) 
     seqhis             86ad                 TO SR SC SA                                      tot     13 cat (      13      0 )  frac (   1.000  0.000 ) 
     seqhis             8a6d                 TO SC SR SA                                      tot      5 cat (       5      0 )  frac (   1.000  0.000 ) 
     seqhis            8a6ad                 TO SR SC SR SA                                   tot      2 cat (       2      0 )  frac (   1.000  0.000 ) 
    2017-11-21 16:34:06.042 INFO  [6188285] [OpticksEventAna::dumpStepByStepCSGExcursions@168] OpticksEventAna::dumpStepByStepCSGExcursions m_dbgseqhis                0 count 0 dumpmax 10


    tboolean-;tboolean-zsphere1 --okg4 --nore --noab --nosc 

    2017-11-21 16:37:11.190 INFO  [6189200] [OpticksEventAna::countExcursions@81] OpticksEventAna::countExcursions pho_num 100000 epsilon 0.1 dbgseqhis 0 dbgseqhis                                                 
    2017-11-21 16:37:11.238 INFO  [6189200] [OpticksEventAna::countExcursions@136] OpticksEventAna::countExcursions pho_num 100000 dbgseqhis 0 dbgseqhis                                                  count 0
    2017-11-21 16:37:11.280 INFO  [6189200] [OpticksEventAna::countExcursions@136] OpticksEventAna::countExcursions pho_num 100000 dbgseqhis 0 dbgseqhis                                                  count 0
    2017-11-21 16:37:11.280 INFO  [6189200] [OpticksEventAna::dump@57] GGeoTest::anaEvent OpticksEventAna pho 100000,4,4 seq 100000,1,2
    2017-11-21 16:37:11.280 INFO  [6189200] [OpticksEventStat::dump@89] per-seqhis per-tree counts on NCSG tree surface evt Evt /tmp/blyth/opticks/evt/tboolean-zsphere1/torch/1 20171121_163707 /usr/local/opticks/lib/OKG4Test totmin 2
     seqhis               8d                 TO SA                                            tot  80300 cat (   80300      0 )  frac (   1.000  0.000 ) 
     seqhis              8ad                 TO SR SA                                         tot  19700 cat (   19700      0 )  frac (   1.000  0.000 ) 
    2017-11-21 16:37:11.282 INFO  [6189200] [OpticksEventAna::dumpStepByStepCSGExcursions@168] OpticksEventAna::dumpStepByStepCSGExcursions m_dbgseqhis                0 count 0 dumpmax 10
    2017-11-21 16:37:11.282 INFO  [6189200] [OpticksAna::run@50] OpticksAna::run
    2017-11-21 16:37:11.282 INFO  [6189200] [OpticksAna::run@53] OpticksAna::run anakey 
    --tag 1 --tagoffset 0 --det tboolean-zsphere1 --src torch
    2017-11-21 16:37:11.284 INFO  [6189200] [SSys::run@46] echo --tag 1 --tagoffset 0 --det tboolean-zsphere1 --src torch   rc_raw : 0 rc : 0
    2017-11-21 16:37:11.284 INFO  [6189200] [OpticksAna::run@59] OpticksAna::run anakey  cmdline echo --tag 1 --tagoffset 0 --det tboolean-zsphere1 --src torch   rc 0





CSG Excursions for each tree at each photon step for each photon
--------------------------------------------------------------------

how to collectivize ? 
~~~~~~~~~~~~~~~~~~~~~~~~

* assume single seqhis input, so have fixed number of points (recs)
* delta min/max/avg for each position of each tree...  so for 3 points and 2 trees just have 6 triplets 
* make an NCSGIntersect class to hold the sdf an collect point excursions 

::

    2017-11-21 20:37:59.509 INFO  [6280871] [OpticksEventAna::dumpPointExcursions@84] ok dbgseqhis 8ad dbgseqhis TO SR SA                                        
    min/max/avg signed-distance(mm) and time(ns) of each photon step point from each NCSG tree
    [p: 0](  19700)(     -0.100    -0.100    -0.100       0.000) mm[p: 0](  19700)(    799.900   799.900   799.980       0.000) mm
    [p: 1](  19700)(   -799.995  -501.165  -686.365       0.000) mm[p: 1](  19700)(     -0.022     0.023     0.004       0.000) mm
    [p: 2](  19700)(     -0.008    -0.008    -0.008       0.000) mm[p: 2](  19700)(    545.533  1220.413   802.941       0.000) mm
    [p: 0](  19700)(      0.200     0.200     0.200       0.000) ns[p: 0](  19700)(      0.200     0.200     0.200       0.000) ns
    [p: 1](  19700)(      2.868     3.522     2.903       0.000) ns[p: 1](  19700)(      2.868     3.522     2.903       0.000) ns
    [p: 2](  19700)(      5.337     7.950     5.722       0.000) ns[p: 2](  19700)(      5.337     7.950     5.722       0.000) ns
    2017-11-21 20:37:59.509 INFO  [6280871] [OpticksEventAna::dumpPointExcursions@84] g4 dbgseqhis 8ad dbgseqhis TO SR SA                                        
    min/max/avg signed-distance(mm) and time(ns) of each photon step point from each NCSG tree
    [p: 0](  19699)(     -0.100    -0.100    -0.100       0.000) mm[p: 0](  19699)(    799.900   799.900   799.980       0.000) mm
    [p: 1](  19699)(   -996.915  -501.165  -699.756       0.000) mm[p: 1](  19699)(   -198.534     0.023   -56.019       0.000) mm
    [p: 2](  19699)(     -0.008    -0.008    -0.008       0.000) mm[p: 2](  19699)(    545.533  1220.413   809.020       0.000) mm
    [p: 0](  19699)(      0.200     0.200     0.200       0.000) ns[p: 0](  19699)(      0.200     0.200     0.200       0.000) ns
    [p: 1](  19699)(      2.868     3.530     3.090       0.000) ns[p: 1](  19699)(      2.868     3.530     3.090       0.000) ns
    [p: 2](  19699)(      5.337     8.428     7.239       0.000) ns[p: 2](  19699)(      5.337     8.428     7.239       0.000) ns
    simon:opticksnpy blyth$ 


how to assert ? 
~~~~~~~~~~~~~~~~~

* dbgseqhis eg 0x8ad "TO SR SA" is an input 
* perhaps dbgseqtree eg 0x121 1-based tree index expected for each point : this will work for very simple test geometry 
* more generally dbgxseqtree="0,1,0" string list of expected trees for each point, that assumes there is an expected
  tree for every point

* xseqtree="TO:0 SC:- SR:1 SA:0" provides both the dbgseqhis and expected trees for each point 

 


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




