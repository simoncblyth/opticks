tboolean-proxy-g4evt-immediate-absorption
===============================================

Context
---------

Following major event paths refactor (and before than test geometry proxying implementation
and container resizing), g4 photon propagation is broken.

* :doc:`opticks-event-paths`
* :doc:`tboolean-with-proxylv-bringing-in-basis-solids`
* :doc:`review-test-geometry`


To create the events
-----------------------

::

   PROXYLV=17 tboolean.sh             ## compute mode
   PROXYLV=17 tboolean.sh --interop   ## propagate and visualize it 


Load the events into ipython
---------------------------------

::

    tboolean-;PROXYLV=17 tboolean-proxy-ip



Check extent of issue
-----------------------

tboolean-box is not effected::

   tboolean.sh box


Issue : all g4 photons are immediately absorbed without going anywhere
----------------------------------------------------------------------------


tboolean-;PROXYLV=17 tboolean-proxy-ip::

    A tboolean-proxy-17/tboolean-proxy-17/torch/  1 :  20190610-2223 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-proxy-17/evt/tboolean-proxy-17/torch/1/fdom.npy () 
    B tboolean-proxy-17/tboolean-proxy-17/torch/ -1 :  20190610-2223 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-proxy-17/evt/tboolean-proxy-17/torch/-1/fdom.npy (recstp) 
    tboolean-proxy-17
    .                seqhis_ana  1:tboolean-proxy-17:tboolean-proxy-17   -1:tboolean-proxy-17:tboolean-proxy-17        c2        ab        ba 
    .                              10000     10000     19786.00/5 = 3957.20  (pval:0.000 prob:1.000)  
    0000           8ccccd      7728         0          7728.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT BT BT SA
    0001              8bd       580         0           580.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO BR SA
    0002            8cbcd       564         0           564.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BR BT SA
    0003          8ccbccd       491         0           491.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT BR BT BT SA
    0004        8cccbcccd       423         0           423.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT BT BR BT BT BT SA
    0005       8cccbcbccd        29         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BT BR BT BT BT SA
    0006         8ccbbccd        28         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT BR BR BT BT SA
    0007       ccbccbcccd        26         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BT BR BT BT
    0008         8cccbbcd        26         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BR BR BT BT BT SA
    0009         8cbbcccd        24         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT BT BR BR BT SA
    0010       8ccbcbcccd        20         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BR BT BT SA
    0011              86d         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO SC SA
    0012          8cbbbcd         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BR BR BR BT SA
    0013        8cbbcbccd         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT BR BT BR BR BT SA
    0014       ccbbcbcccd         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BR BR BT BT
    0015       bcbccbcccd         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BT BR BT BR
    0016          8cc6ccd         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT SC BT BT SA
    0017       cbbccbcccd         3         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BT BR BR BT
    0018       bbccbcbccd         3         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BT BR BT BT BR BR
    0019          86ccccd         3         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT BT BT SC SA
    .                              10000     10000     19786.00/5 = 3957.20  (pval:0.000 prob:1.000)  


    ## adjust the slice to find the g4 photons, they are all under "TO AB"

    In [7]: ab.his[35:50]
    Out[7]: 
    .                seqhis_ana  1:tboolean-proxy-17:tboolean-proxy-17   -1:tboolean-proxy-17:tboolean-proxy-17        c2        ab        ba 
    .                              10000     10000     19786.00/5 = 3957.20  (pval:0.000 prob:1.000)  
    0035       ccbcbcbccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BT BR BT BR BT BT
    0036       cbbcbccc6d         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SC BT BT BT BR BT BR BR BT
    0037       cbcbcbbccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BR BT BR BT BR BT
    0038       ccbcbbbccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BR BR BT BR BT BT
    0039       bcbcbbbccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BR BR BT BR BT BR
    0040       bbbbbcbccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BT BR BR BR BR BR
    0041               4d         0     10000         10000.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO AB
    .                              10000     10000     19786.00/5 = 3957.20  (pval:0.000 prob:1.000)  




    In [12]: a.seqhis_ls[:5]
    Out[12]: 
    TO BT BT BT BT SA
    TO BT BT BR BT BT SA
    TO BR SA
    TO BT BT BT BT SA
    TO BT BT BT BT SA

    In [13]: b.seqhis_ls[:5]
    Out[13]: 
    TO AB
    TO AB
    TO AB
    TO AB
    TO AB

    In [14]: b.seqhis
    Out[14]: 
    A()sliced
    A([77, 77, 77, ..., 77, 77, 77], dtype=uint64)

    In [15]: np.unique(b.seqhis)
    Out[15]: 
    A()sliced
    A([77], dtype=uint64)


All B are two steps going nowhere::

    In [13]: b.rpostn(2).shape
    Out[13]: (10000, 2, 4)

    In [14]: a.rpostn(2).shape
    Out[14]: (0, 2, 4)

    In [15]: b.rpostn(2)
    Out[15]: 
    A()sliced
    A([[[  20.6922,  -63.5134, -825.8752,    0.    ],
        [  20.6922,  -63.5134, -825.8752,    0.    ]],

       [[ -48.9204,   -0.5293, -825.8752,    0.    ],
        [ -48.9204,   -0.5293, -825.8752,    0.    ]],

       [[ -74.351 ,   17.9955, -825.8752,    0.    ],
        [ -74.351 ,   17.9955, -825.8752,    0.    ]],

       ...,

       [[ -18.8272,   74.0233, -825.8752,    0.    ],
        [ -18.8272,   74.0233, -825.8752,    0.    ]],

       [[ -16.0548,   36.1925, -825.8752,    0.    ],
        [ -16.0548,   36.1925, -825.8752,    0.    ]],

       [[  28.7322,   56.8848, -825.8752,    0.    ],
        [  28.7322,   56.8848, -825.8752,    0.    ]]])




    In [2]: x = b.rpostn(2)

    In [3]: x.shape
    Out[3]: (10000, 2, 4)

    In [5]: x[:,0,2]
    Out[5]: 
    A([-825.8752, -825.8752, -825.8752, ..., -825.8752, -825.8752, -825.8752])

    In [6]: np.unique(x[:,0,2])    ## all same
    Out[6]: 
    A([-825.8752])




Hmm unexplained z-difference, perhaps a start delta to avoid being stuck on boundary ?

* hmm that might explain the peculiar photon behaviour observed in :doc:`tboolean-with-proxylv-bringing-in-basis-solids`
  with large extent proxies if the start delta was not big enough  

::

    [blyth@localhost issues]$ np.py $TMP/cg4/primary.npy -v --sli 0:10
    a :                          /tmp/blyth/location/cg4/primary.npy :        (10000, 4, 4) : f1520b5be97926aff24f10f576f0a725 : 20190610-2223 
    (10000, 4, 4)
    f32
    [[[  20.6971  -63.5045 -903.7001    0.    ]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.        0.    ]
      [   0.        0.        0.        0.    ]]

     [[ -48.9207   -0.5178 -903.7001    0.    ]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.        0.    ]
      [   0.        0.        0.        0.    ]]

     [[ -74.3543   17.9927 -903.7001    0.    ]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.        0.    ]
      [   0.        0.        0.        0.    ]]




First Thing : switch on some g4 debug 
------------------------------------------

::

    PROXYLV=17 tboolean.sh --dbgrec              # this fairly useless, machinery debug 

    PROXYLV=17 tboolean.sh --dbgseqhis 0x4d      # this looks useful, dumping just "TO AB" photons which is all of them  

    PROXYLV=17 tboolean.sh --dbgseqhis 0x4d --generateoverride 5       ## restrict to 1st 5 photons


* hmm need to look into UNIVERSE_PV 


According to g4 the photons are starting in Rock and immediately get absorbed::

    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump@159] CDebug::postTrack
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@162] CDebug::dump record_id 1  origin[ -48.921-0.518-903.700]   Ori[ -48.921-0.518-903.700] 
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@168]  nstp 1
    ( 0)  TO/AB     Und   PRE_SAVE POST_SAVE POST_DONE LAST_POST STEP_START 
    [   0](Stp ;opticalphoton stepNum    1(tk ;opticalphoton tid 2 pid 0 nm    380 mm  ori[  -48.921  -0.518-903.700]  pos[    0.000   0.000   0.002]  )
      pre               UNIVERSE_PV            Rock          noProc           Undefined pos[      0.000     0.000     0.000]  dir[   -0.000  -0.000   1.000]  pol[    0.000  -1.000   0.000]  ns  0.000 nm 380.000 mm/ns 299.792
     post               UNIVERSE_PV            Rock    OpAbsorption    PostStepDoItProc pos[      0.000     0.000     0.002]  dir[   -0.000  -0.000   1.000]  pol[    0.000  -1.000   0.000]  ns  0.000 nm 380.000 mm/ns 299.792
     )
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@172]  npoi 0
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@176] CRecorder::dump_brief m_ctx._record_id        1 m_photon._badflag     0 --dbgseqhis  sas: PRE_SAVE POST_SAVE POST_DONE LAST_POST STEP_START 
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@185]  seqhis               4d    TO AB                                           
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@190]  mskhis             1008    AB|TO
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@195]  seqmat               33    Rock Rock - - - - - - - - - - - - - - 
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_sequence@203] CDebug::dump_sequence
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_points@229] CDeug::dump_points
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump@159] CDebug::postTrack
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@162] CDebug::dump record_id 0  origin[ 20.697-63.504-903.700]   Ori[ 20.697-63.504-903.700] 
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@168]  nstp 1
    ( 0)  TO/AB     Und   PRE_SAVE POST_SAVE POST_DONE LAST_POST STEP_START 
    [   0](Stp ;opticalphoton stepNum    1(tk ;opticalphoton tid 1 pid 0 nm    380 mm  ori[   20.697 -63.504-903.700]  pos[    0.000   0.000   0.003]  )
      pre               UNIVERSE_PV            Rock          noProc           Undefined pos[      0.000     0.000     0.000]  dir[   -0.000  -0.000   1.000]  pol[    0.000  -1.000   0.000]  ns  0.000 nm 380.000 mm/ns 299.792
     post               UNIVERSE_PV            Rock    OpAbsorption    PostStepDoItProc pos[      0.000     0.000     0.003]  dir[   -0.000  -0.000   1.000]  pol[    0.000  -1.000   0.000]  ns  0.000 nm 380.000 mm/ns 299.792
     )
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@172]  npoi 0
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@176] CRecorder::dump_brief m_ctx._record_id        0 m_photon._badflag     0 --dbgseqhis  sas: PRE_SAVE POST_SAVE POST_DONE LAST_POST STEP_START 
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@185]  seqhis               4d    TO AB                                           
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@190]  mskhis             1008    AB|TO
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@195]  seqmat               33    Rock Rock - - - - - - - - - - - - - - 
    2019-0




G4 OK Geometry mismatch : likely source container auto resizing : CONFIRMED by adding containerautosize control  
------------------------------------------------------------------------------------------------------------------

* emitter is also a container and containers gets auto-resized when proxying 
  in base solids : thats a likely cause, try switching off auto-resizing

::

    tboolean-proxy-- () 
    { 
        cat  <<EOP
    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.main import opticks_main
    from opticks.analytic.csg import CSG  

    autoemitconfig="photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0"
    args = opticks_main(csgpath="$(tboolean-proxy-name)", autoemitconfig=autoemitconfig)

    # 0x3f is all 6 
    # 0x1 is -Z
    # 0x2 is +Z   havent succeed to get this to work yet 
    
    emitconfig = "photons:10000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x2,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55" 

    CSG.kwa = dict(poly="IM",resolution="20", verbosity="0", ctrl=0, containerscale=3.0, emitconfig=emitconfig  )

    container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container=1 )  # no param, container="1" switches on auto-sizing

    box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2", proxylv=$(tboolean-proxy-lvidx) )

    CSG.Serialize([container, box], args )
    EOP

    }




With containerautosize=1 see discrepancy between uncompressed ox and domain compressed rx 
--------------------------------------------------------------------------------------------------------

::

    In [18]: b.ox[:,0]
    Out[18]: 
    A()sliced
    A([[  20.6971,  -63.5045, -903.7   ,    0.    ],
       [ -48.9207,   -0.5178, -903.6999,    0.    ],
       [ -74.3543,   17.9927, -903.6993,    0.    ],
       ...,
       [ -18.8286,   74.0311, -903.699 ,    0.    ],
       [ -16.0536,   36.2017, -903.6989,    0.    ],
       [  28.7337,   56.8787, -903.7001,    0.    ]], dtype=float32)

    In [19]: b.ox[:,0,2]
    Out[19]: 
    A()sliced
    A([-903.7   , -903.6999, -903.6993, ..., -903.699 , -903.6989, -903.7001], dtype=float32)

    In [20]: b.ox[:,0,2].min()
    Out[20]: 
    A()sliced
    A(-903.7001, dtype=float32)

    In [21]: b.ox[:,0,2].max()
    Out[21]: 
    A()sliced
    A(-903.6913, dtype=float32)


    In [33]: b.rpostn(2)[:,1,2].min()    # z of the 2nd position (AB) of g4 photons 
    Out[33]: 
    A(-825.8752)

    In [34]: b.rpostn(2)[:,1,2].max()
    Out[34]: 
    A(-825.8752)


* suggests fdom not accounting for resizing ?

::

    In [37]: a.fdom[0]
    Out[37]: 
    A()sliced
    A([[  0.  ,   0.  ,   0.  , 825.85]], dtype=float32)

    In [38]: b.fdom[0]
    Out[38]: 
    A()sliced
    A([[  0.  ,   0.  ,   0.  , 825.85]], dtype=float32)



Review OpticksDomain, add header docs
-------------------------------------------

Canonical m_domain instance is a resident of OpticksEvent and
is instancianted by OpticksEvent::init. The domains are 
critically important for record domain compression.

* OpticksEvent getters and setters defer to OpticksDomain.
* Note the vec and buffers duplication

  1. local glm::vec4/glm::ivec4 
  2. fdom/idom NPY buffers 

* copies both ways by updateBuffer() and importBuffer()

* domains are setup by Opticks::makeEvent on creating an OpticksEvent
  using results of Opticks getters such as Opticks::getSpaceDomain

* domain information comes from Opticks::setSpaceDomain which 
  triggers Opticks::postgeometry Opticks::configureDomains 

::

    [blyth@localhost opticks]$ opticks-f m_ok-\>setSpaceDomain
    ./cfg4/CGeometry.cc:    m_ok->setSpaceDomain(ce); // triggers Opticks::configureDomains
    ./opticksgeo/OpticksAim.cc:    m_ok->setSpaceDomain( ce0 );
    ./okop/OpIndexerApp.cc:    m_ok->setSpaceDomain(0.f,0.f,0.f,1000.f);  // this is required before can create an evt 


* OpticksAim::registerGeometry invokes Opticks::setSpaceDomain with 
  geometry information from mm0 the first GMergedMesh 

* OpticksHub::registerGeometry invokes OpticksAim::registerGeometry
  in the tail of OpticksHub::loadGeometry

* CGeometry::hookup also invokes Opticks::setSpaceDomain, which happens at CG4::CG4 

* Q: why twice ?  



Possible Cause
------------------

* test geometry is making its own resized mesh and not putting it
  in the standard GGeoLib : so maybe the registerGeometry is not seeing the resized mm0 ?



OpticksAim::registerGeometry --dbgaim
-------------------------------------------

Inconsitent space_domain::

    2019-06-11 13:05:17.642 INFO  [43238] [OpticksHub::loadGeometry@508] --test modifying geometry
    2019-06-11 13:05:17.642 INFO  [43238] [OpticksHub::createTestGeometry@560] [
    2019-06-11 13:05:17.642 INFO  [43238] [NCSGList::load@181]  VERBOSITY 0 basedir tboolean-proxy-17 txtpath tboolean-proxy-17/csg.txt nbnd 2
    2019-06-11 13:05:17.644 ERROR [43238] [NCSGList::add@114]  add tree, boundary: Rock//perfectAbsorbSurface/Vacuum
    2019-06-11 13:05:17.644 INFO  [43238] [NCSG::postload@301]  proxylv 17
    2019-06-11 13:05:17.645 ERROR [43238] [NCSGList::add@114]  add tree, boundary: Vacuum///GlassSchottF2
    2019-06-11 13:05:17.645 INFO  [43238] [NCSGList::adjustContainerSize@155]  m_bbox  mi (   -450.000  -450.000  -450.000) mx (    450.000   450.000   450.000) si (    900.000   900.000   900.000)
    2019-06-11 13:05:17.677 FATAL [43238] [GGeoTest::adjustContainer@352]  containerautosize ENABLED by metadata on container CSG 1
    2019-06-11 13:05:17.677 INFO  [43238] [NCSGList::adjustContainerSize@155]  m_bbox  mi (   -824.850  -824.850  -903.800) mx (    824.850   824.850   745.900) si (   1649.700  1649.700  1649.700)
    2019-06-11 13:05:17.679 INFO  [43238] [OpticksHub::createTestGeometry@564] ]
    2019-06-11 13:05:17.679 FATAL [43238] [Opticks::setSpaceDomain@1926]  --dbgaim : m_space_domain 0.0000,0.0000,-78.9500,824.8500
    2019-06-11 13:05:17.679 FATAL [43238] [OpticksAim::registerGeometry@43]  setting SpaceDomain :  ce0 0.0000,0.0000,-78.9500,824.8500
    2019-06-11 13:05:17.681 INFO  [43238] [OpticksHub::loadGeometry@534] ]
    ...
    2019-06-11 13:05:17.748 INFO  [43238] [CDetector::attachSurfaces@340] ]
    2019-06-11 13:05:17.748 ERROR [43238] [CDetector::hookupSD@129]  NOT INVOKING SetSensitiveDetector ON ANY VOLUMES AS nlvsd is zero or m_sd NULL  nlvsd 0 m_sd 0x63cda90 sdname SD0
    2019-06-11 13:05:17.748 FATAL [43238] [CGeometry::hookup@93]  center_extent 0.0000,0.0000,0.0000,825.8500
    2019-06-11 13:05:17.748 FATAL [43238] [Opticks::setSpaceDomain@1926]  --dbgaim : m_space_domain 0.0000,0.0000,0.0000,825.8500
    2019-06-11 13:05:17.748 FATAL [43238] [CGenerator::initSource@52]  code 262144 SourceType EMITSOURCE m_source_type EMITSOURCE
    2019-06-11 13:05:17.748 INFO  [43238] [CGenerator::initInputPhotonSource@179] CGenerator::initInputPhotonSource 

* G4 : huh 1 mm larger extent, and symmetric
* and the 2nd G4 one is the one that gets persisted::

    In [37]: a.fdom[0]
    Out[37]: 
    A([[  0.  ,   0.  ,   0.  , 825.85]], dtype=float32)

    In [38]: b.fdom[0]
    Out[38]: 
    A([[  0.  ,   0.  ,   0.  , 825.85]], dtype=float32)


::

    In [1]: 745.9-903.8
    Out[1]: -157.89999999999998

    In [2]: (745.9-903.8)/2.
    Out[2]: -78.94999999999999

    In [3]: 824.8500-78.9500
    Out[3]: 745.9

    In [4]: -824.8500-78.9500
    Out[4]: -903.8000000000001




NCSGList::createUniverse
----------------------------

Universe wrapper looks implicated, what was that added for ?

* :doc:`okg4-material-drastic-difference`

   Universe wrapper is there to reconcile Opticks surface model and G4 volume model

* :doc:`surface_review_test_geometry`



::

    202 /**
    203 NCSGList::getUniverse
    204 -----------------------
    205 
    206 No longer create universe by default, 
    207 as with full geomrtries NCSGLoadTest and NScanTest 
    208 when reading /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/csg.txt
    209 takes exception to the content of "extras/248" not being a bnd
    210 
    211 **/
    212 
    213 NCSG* NCSGList::getUniverse()
    214 {
    215     float scale = 1.f ;
    216     float delta = 1.f ;
    217 
    218     if(m_universe == NULL) m_universe = createUniverse(scale, delta);
    219     return m_universe ;
    220 }
    221 
    222 /**
    223 NCSGList::createUniverse
    224 -------------------------
    225 
    226 "cheat" clone (via 2nd load) of outer volume 
    227 then increase size a little 
    228 this is only used for the Geant4 geometry
    229 
    230 **/
    231 
    232 NCSG* NCSGList::createUniverse(float scale, float delta) const
    233 {
    234     const char* bnd0 = getBoundary(0);
    235     const char* ubnd = BBnd::DuplicateOuterMaterial( bnd0 );
    236 
    237     LOG(info)
    238         << " bnd0 " << bnd0
    239         << " ubnd " << ubnd
    240         << " scale " << scale
    241         << " delta " << delta
    242         ;
    243 
    244     NCSG* universe = loadTree(0) ;
    245     universe->setBoundary(ubnd);
    246 
    247     if( universe->isContainer() )
    248     {
    249         LOG(info)
    250             << " outer volume isContainer (ie auto scaled) "
    251             << " universe will be scaled/delted a bit from there "
    252             ;
    253     }
    254 
    255     universe->adjustToFit( m_bbox, scale, delta );
    256     /// huh : not re-exported : this means different geometry on CPU and GPU ??
    257     return universe ;
    258 }



::

    1100 /**
    1101 NCSG::adjustToFit
    1102 ------------------
    1103 
    1104 Changes extent of analytic geometry to be that of the container argument
    1105 with scale and delta applied.
    1106 Only implemented for CSG_BOX, CSG_BOX3 and CSG_SPHERE.
    1107 
    1108 **/
    1109 
    1110 void NCSG::adjustToFit( const nbbox& container, float scale, float delta ) const
    1111 {
    1112     LOG(debug) << "NCSG::adjustToFit START " ;
    1113 
    1114     nnode* root = getRoot();
    1115 
    1116     nbbox root_bb = root->bbox();
    1117 
    1118     nnode::AdjustToFit(root, container, scale, delta );
    1119 
    1120     LOG(debug) << "NCSG::updateContainer DONE"
    1121               << " root_bb " << root_bb.desc()
    1122               << " container " << container.desc()
    1123               ;
    1124 }
    1125 

::

    383     else if(node->type == CSG_BOX || node->type == CSG_BOX3)
    384     {
    385         // BOX can have an offset, BOX3 cannot it being always origin centered. 
    386         // Hence treating them as equivalent will loose the offset for BOX.
    387         
    388         nbox* n = (nbox*)node ;
    389         glm::vec3 halfside = n->halfside();
    390         
    391         G4Box* bx = new G4Box( name, halfside.x, halfside.y, halfside.z );
    392         result = bx ; 



Try to avoid loosing the box offset with CTestDetector::boxCenteringFix
------------------------------------------------------------------------

::

    099 /**
    100 CTestDetector::boxCenteringFix
    101 --------------------------------
    102 
    103 See notes/issues/tboolean-proxy-g4evt-immediate-absorption.rst
    104 
    105 **/
    106 
    107 void CTestDetector::boxCenteringFix( glm::vec3& placement, nnode* root  )
    108 {
    109     assert( root->type == CSG_BOX ) ;
    110     nbox* box = (nbox*)root ;
    111     if( !box->is_centered() )
    112     {
    113         glm::vec3 center = box->center();
    114         LOG(fatal) << " box.center " << gformat(center) ;
    115         placement = center ;
    116         box->set_centered() ;
    117     }
    118     assert( box->is_centered() );
    119 }
    120 


BUT Geant4 takes exception to a non-centered universe::


    2019-06-11 15:45:19.391 FATAL [337650] [NCSGList::createUniverse@237]  bnd0 Rock//perfectAbsorbSurface/Vacuum ubnd Rock///Rock scale 1 delta 1
    2019-06-11 15:45:19.391 FATAL [337650] [NCSGList::createUniverse@244]  m_bbox  mi (   -824.850  -824.850  -903.800) mx (    824.850   824.850   745.900) si (   1649.700  1649.700  1649.700)
    2019-06-11 15:45:19.392 FATAL [337650] [NCSGList::createUniverse@253]  universe.get_root_csgname box
    2019-06-11 15:45:19.392 INFO  [337650] [NCSGList::createUniverse@258]  outer volume isContainer (ie auto scaled)  universe will be scaled/delted a bit from there 
    2019-06-11 15:45:19.395 FATAL [337650] [CTestDetector::boxCenteringFix@114]  box.center 0.0000,0.0000,-78.9500
    2019-06-11 15:45:19.396 FATAL [337650] [CTestDetector::makeChildVolume@166]  csg.spec Rock///Rock csg.get_root_csgname box boundary 2 mother - lv UNIVERSE_LV pv UNIVERSE_PV mat Rock
    2019-06-11 15:45:19.396 INFO  [337650] [CTestDetector::makeDetector_NCSG@228]    0 spec Rock//perfectAbsorbSurface/Vacuum
    2019-06-11 15:45:19.396 FATAL [337650] [CTestDetector::boxCenteringFix@114]  box.center 0.0000,0.0000,-78.9500
    2019-06-11 15:45:19.396 FATAL [337650] [CTestDetector::makeChildVolume@166]  csg.spec Rock//perfectAbsorbSurface/Vacuum csg.get_root_csgname box boundary 0 mother UNIVERSE_LV lv box_lv0_ pv box_pv0_ mat Vacuum
    2019-06-11 15:45:19.396 INFO  [337650] [CTestDetector::makeDetector_NCSG@228]    1 spec Vacuum///GlassSchottF2
    2019-06-11 15:45:19.396 INFO  [337650] [nnode::reconstruct_ellipsoid@1905]  sx 1.34694 sy 1.34694 sz 1 radius 196
    2019-06-11 15:45:19.396 ERROR [337650] [CMaker::MakeSolid_r@134]  non-identity left transform on sphere (an ellipsoid perhaps) 
    2019-06-11 15:45:19.397 INFO  [337650] [nnode::reconstruct_ellipsoid@1905]  sx 1.37634 sy 1.37634 sz 1 radius 186
    2019-06-11 15:45:19.397 ERROR [337650] [CMaker::MakeSolid_r@134]  non-identity left transform on sphere (an ellipsoid perhaps) 
    2019-06-11 15:45:19.397 FATAL [337650] [CTestDetector::makeChildVolume@166]  csg.spec Vacuum///GlassSchottF2 csg.get_root_csgname difference boundary 1 mother box_lv0_ lv difference_lv0_ pv difference_pv0_ mat GlassSchottF2
    2019-06-11 15:45:19.397 INFO  [337650] [CDetector::setTop@94] .
    2019-06-11 15:45:19.397 INFO  [337650] [CTraverser::Summary@106] CDetector::traverse numMaterials 3 numMaterialsWithoutMPT 0
    2019-06-11 15:45:19.397 INFO  [337650] [CDetector::attachSurfaces@323] [ num_bs 0 num_sk 0
    2019-06-11 15:45:19.397 ERROR [337650] [CDetector::attachSurfaces@335]  no surfaces found : try to convert some from Opticks model 
    2019-06-11 15:45:19.397 INFO  [337650] [CSurfaceLib::convert@81] .
    2019-06-11 15:45:19.397 INFO  [337650] [CSurfaceLib::convert@93] . num_surf 1
    2019-06-11 15:45:19.397 INFO  [337650] [CTraverser::getPV@317] CTraverser::getPV name box_pv0_ index 1 num_indices 1
    2019-06-11 15:45:19.397 INFO  [337650] [CTraverser::getPV@317] CTraverser::getPV name UNIVERSE_PV index 0 num_indices 1
    2019-06-11 15:45:19.397 INFO  [337650] [CSurfaceLib::convert@136] CSurfaceLib  numBorderSurface 1 numSkinSurface 0
    2019-06-11 15:45:19.397 INFO  [337650] [CDetector::attachSurfaces@340] ]
    2019-06-11 15:45:19.397 ERROR [337650] [CDetector::hookupSD@129]  NOT INVOKING SetSensitiveDetector ON ANY VOLUMES AS nlvsd is zero or m_sd NULL  nlvsd 0 m_sd 0x60cac60 sdname SD0
    2019-06-11 15:45:19.397 FATAL [337650] [CGeometry::hookup@93]  center_extent 0.0000,0.0000,-117.9250,864.8251
    2019-06-11 15:45:19.397 FATAL [337650] [Opticks::setSpaceDomain@1926]  --dbgaim : m_space_domain 0.0000,0.0000,-117.9250,864.8251
    2019-06-11 15:45:19.397 FATAL [337650] [CGenerator::initSource@52]  code 262144 SourceType EMITSOURCE m_source_type EMITSOURCE
    2019-06-11 15:45:19.397 INFO  [337650] [CGenerator::initInputPhotonSource@179] CGenerator::initInputPhotonSource 
    2019-06-11 15:45:19.398 FATAL [337650] [CGenerator::initSource@79]  code 262144 type EMITSOURCE STATIC
    2019-06-11 15:45:19.398 FATAL [337650] [CWriter::CWriter@50]  STATIC
    2019-06-11 15:45:19.398 FATAL [337650] [CRecorder::CRecorder@77]  STATIC
    2019-06-11 15:45:19.398 INFO  [337650] [CRunAction::CRunAction@10] CRunAction::CRunAction count 0
    2019-06-11 15:45:19.398 INFO  [337650] [CG4::init@150] CG4::init ctx  record_id -1 event_id -1 track_id -1 photon_id -1 parent_id -1 primary_id -1 reemtrack 0
    2019-06-11 15:45:19.398 INFO  [337650] [CG4::initialize@169] [

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : GeomNav0002
          issued by : G4Navigator::SetWorldVolume()
    Volume must be centered on the origin.
    *** Fatal Exception *** core dump ***
     **** Track information is not available at this moment
     **** Step information is not available at this moment

    -------- EEEE -------- G4Exception-END --------- EEEE -------


    *** G4Exception: Aborting execution ***

    Program received signal SIGABRT, Aborted.
    0x00007fffe2020207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2020207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20218f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe7e35f8b in G4Exception (originOfException=0x7fffed003663 "G4Navigator::SetWorldVolume()", exceptionCode=0x7fffed003657 "GeomNav0002", severity=FatalException, description=0x7fffed003630 "Volume must be centered on the origin.")
            at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/global/management/src/G4Exception.cc:100
    #3  0x00007fffecfe1418 in G4Navigator::SetWorldVolume (this=0x6066040, pWorld=0x610b590) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/navigation/include/G4Navigator.icc:96
    #4  0x00007fffec6c0109 in G4TransportationManager::SetWorldForTracking (this=0x6065fd0, theWorld=0x610b590) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/navigation/include/G4TransportationManager.icc:59
    #5  0x00007fffec6bdbd3 in G4RunManagerKernel::DefineWorldVolume (this=0x5eeab50, worldVol=0x610b590, topologyIsChanged=false) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManagerKernel.cc:497
    #6  0x00007fffec6b0305 in G4RunManager::InitializeGeometry (this=0x5eeaa30) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:588
    #7  0x00007fffec6b01cb in G4RunManager::Initialize (this=0x5eeaa30) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:566
    #8  0x00007fffefdec17a in CG4::initialize (this=0x5eeac10) at /home/blyth/opticks/cfg4/CG4.cc:179
    #9  0x00007fffefdebeb4 in CG4::init (this=0x5eeac10) at /home/blyth/opticks/cfg4/CG4.cc:151
    #10 0x00007fffefdebc54 in CG4::CG4 (this=0x5eeac10, hub=0x6b8e50) at /home/blyth/opticks/cfg4/CG4.cc:143
    #11 0x00007ffff7bd5256 in OKG4Mgr::OKG4Mgr (this=0x7fffffffcc10, argc=34, argv=0x7fffffffcf48) at /home/blyth/opticks/okg4/OKG4Mgr.cc:76
    #12 0x0000000000403998 in main (argc=34, argv=0x7fffffffcf48) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
        (gdb) f 8
    #8  0x00007fffefdec17a in CG4::initialize (this=0x5eeac10) at /home/blyth/opticks/cfg4/CG4.cc:179
        179     m_runManager->Initialize();
        (gdb) f 7
    #7  0x00007fffec6b01cb in G4RunManager::Initialize (this=0x5eeaa30) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:566
        566   if(!geometryInitialized) InitializeGeometry();
        (gdb) f 6
    #6  0x00007fffec6b0305 in G4RunManager::InitializeGeometry (this=0x5eeaa30) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:588
        588   kernel->DefineWorldVolume(userDetector->Construct(),false);
        (gdb) 



Hmm : maybe just center the standard volume that comes in via the proxy to avoid this ?
--------------------------------------------------------------------------------------------

::

    [blyth@localhost tests]$ GMeshLibTest --envkey 
    2019-06-11 16:22:23.005 INFO  [402242] [Opticks::init@308] INTEROP_MODE
    2019-06-11 16:22:23.006 FATAL [402242] [Opticks::configure@1719]  --interop mode with no cvd specified, adopting OPTICKS_DEFAULT_INTEROP_CVD hinted by envvar [1]
    2019-06-11 16:22:23.006 INFO  [402242] [Opticks::configure@1726]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
    2019-06-11 16:22:23.012 INFO  [402242] [BOpticksResource::setupViaKey@544] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
                     exename  : OKX4Test
             current_exename  : GMeshLibTest
                       class  : X4PhysicalVolume
                     volname  : lWorld0x4bc2710_PV
                      digest  : f6cc352e44243f8fa536ab483ad390ce
                      idname  : OKX4Test_lWorld0x4bc2710_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2019-06-11 16:22:23.012 ERROR [402242] [OpticksResource::initRunResultsDir@260] /home/blyth/local/opticks/results/GMeshLibTest/R0_cvd_1/20190611_162223
    2019-06-11 16:22:23.077 ERROR [402242] [NNodeNudger::init@61] NNodeNudger::brief root.treeidx  22 num_prim  1 num_coincidence  0 num_nudge  0 ##LISTED
    2019-06-11 16:22:23.079 ERROR [402242] [NNodeNudger::init@61] NNodeNudger::brief root.treeidx  24 num_prim  1 num_coincidence  0 num_nudge  0 ##LISTED
    2019-06-11 16:22:23.106 ERROR [402242] [NNodeNudger::init@61] NNodeNudger::brief root.treeidx  37 num_prim  1 num_coincidence  0 num_nudge  0 ##LISTED
    2019-06-11 16:22:23.111 INFO  [402242] [GMeshLib::loadMeshes@342]  loaded  meshes 40 solids 40
    2019-06-11 16:22:23.111 INFO  [402242] [main@56]  num_mesh 40
     0                       Upper_LS_tube0x5b2e9f0 bba  mi (   -400.000  -400.000 -1750.000) mx (    400.000   400.000  1750.000) si (    800.000   800.000  3500.000) ce  (   0.00    0.00    0.00 1750.00)   0
     1                    Upper_Steel_tube0x5b2eb10 bba  mi (   -407.000  -407.000 -1750.000) mx (    407.000   407.000  1750.000) si (    814.000   814.000  3500.000) ce  (   0.00    0.00    0.00 1750.00)   1
     2                    Upper_Tyvek_tube0x5b2ec30 bba  mi (   -402.000  -402.000 -1750.000) mx (    402.000   402.000  1750.000) si (    804.000   804.000  3500.000) ce  (   0.00    0.00    0.00 1750.00)   2
     3                       Upper_Chimney0x5b2e8e0 bba  mi (   -412.000  -412.000 -1750.000) mx (    412.000   412.000  1750.000) si (    824.000   824.000  3500.000) ce  (   0.00    0.00    0.00 1750.00)   3
     4                                sBar0x5b34ab0 bba  mi (  -3430.000   -13.000    -5.000) mx (   3430.000    13.000     5.000) si (   6860.000    26.000    10.000) ce  (   0.00    0.00    0.00 3430.00)   4
     5                                sBar0x5b34920 bba  mi (  -3430.000   -13.150    -5.150) mx (   3430.000    13.150     5.150) si (   6860.000    26.300    10.300) ce  (   0.00    0.00    0.00 3430.00)   5
     6                         sModuleTape0x5b34790 bba  mi (  -3430.000  -845.650    -6.050) mx (   3430.000   845.650     6.050) si (   6860.000  1691.300    12.100) ce  (   0.00    0.00    0.00 3430.00)   6
     7                             sModule0x5b34600 bba  mi (  -3430.600  -846.250    -6.650) mx (   3430.600   846.250     6.650) si (   6861.200  1692.500    13.300) ce  (   0.00    0.00    0.00 3430.60)   7
     8                              sPlane0x5b34470 bba  mi (  -3430.600 -3385.150    -6.650) mx (   3430.600  3385.150     6.650) si (   6861.200  6770.300    13.300) ce  (   0.00    0.00    0.00 3430.60)   8
     9                               sWall0x5b342e0 bba  mi (  -3430.600 -3430.600   -13.800) mx (   3430.600  3430.600    13.800) si (   6861.200  6861.200    27.600) ce  (   0.00    0.00    0.00 3430.60)   9
    10                              sAirTT0x5b34000 bba  mi ( -24000.000-24000.000 -2500.000) mx (  24000.000 24000.000  2500.000) si (  48000.000 48000.000  5000.000) ce  (   0.00    0.00    0.00 24000.00)  10
    11                            sExpHall0x4bcd390 bba  mi ( -24000.000-24000.000 -9300.000) mx (  24000.000 24000.000  9300.000) si (  48000.000 48000.000 18600.000) ce  (   0.00    0.00    0.00 24000.00)  11
    12                            sTopRock0x4bccfc0 bba  mi ( -27000.000-27000.000-10800.000) mx (  27000.000 27000.000 10800.000) si (  54000.000 54000.000 21600.000) ce  (   0.00    0.00    0.00 27000.00)  12
    13                             sTarget0x4bd4340 bba  mi ( -17700.000-17700.000-17700.000) mx (  17700.000 17700.000 17820.000) si (  35400.000 35400.000 35520.000) ce  (   0.00    0.00   60.00 17760.00)  13
    14                            sAcrylic0x4bd3cd0 bba  mi ( -17820.000-17820.000-17820.000) mx (  17820.000 17820.000 17820.000) si (  35640.000 35640.000 35640.000) ce  (   0.00    0.00    0.00 17820.00)  14
    15                              sStrut0x4bd4b80 bba  mi (    -51.000   -51.000  -600.000) mx (     51.000    51.000   600.000) si (    102.000   102.000  1200.000) ce  (   0.00    0.00    0.00  600.00)  15
    16                          sFasteners0x4c01080 bba  mi (   -150.000  -150.000  -190.000) mx (    150.000   150.000     5.000) si (    300.000   300.000   195.000) ce  (   0.00    0.00  -92.50  150.00)  16
    17                               sMask0x4ca38d0 bba  mi (   -264.000  -264.000  -353.900) mx (    264.000   264.000   196.000) si (    528.000   528.000   549.900) ce  (   0.00    0.00  -78.95  274.95)  17
    18             PMT_20inch_inner1_solid0x4cb3610 bba  mi (   -249.000  -249.000     0.000) mx (    249.000   249.000   179.000) si (    498.000   498.000   179.000) ce  (   0.00    0.00   89.50  249.00)  18
    19             PMT_20inch_inner2_solid0x4cb3870 bba  mi (   -249.000  -249.000  -334.010) mx (    249.000   249.000     0.000) si (    498.000   498.000   334.010) ce  (   0.00    0.00 -167.01  249.00)  19
    20               PMT_20inch_body_solid0x4c90e50 bba  mi (   -254.000  -254.000  -339.010) mx (    254.000   254.000   184.000) si (    508.000   508.000   523.010) ce  (   0.00    0.00  -77.51  261.51)  20
    21                PMT_20inch_pmt_solid0x4c81b40 bba  mi (   -254.001  -254.001  -339.011) mx (    254.001   254.001   184.001) si (    508.002   508.002   523.012) ce  (   0.00    0.00  -77.50  261.51)  21
    22                       sMask_virtual0x4c36e10 bba  mi (   -264.050  -264.050  -354.050) mx (    264.050   264.050   196.050) si (    528.100   528.100   550.100) ce  (   0.00    0.00  -79.00  275.05)  22
    23   PMT_3inch_inner1_solid_ell_helper0x510ae30 bba  mi (    -38.000   -38.000     7.043) mx (     38.000    38.000    22.000) si (     76.000    76.000    14.957) ce  (   0.00    0.00   14.52   38.00)  23
    24   PMT_3inch_inner2_solid_ell_helper0x510af10 bba  mi (    -38.000   -38.000   -15.875) mx (     38.000    38.000     7.043) si (     76.000    76.000    22.918) ce  (   0.00    0.00   -4.42   38.00)  24
    25 PMT_3inch_body_solid_ell_ell_helper0x510ada0 bba  mi (    -40.000   -40.000   -15.875) mx (     40.000    40.000    24.000) si (     80.000    80.000    39.875) ce  (   0.00    0.00    4.06   40.00)  25
    26                PMT_3inch_cntr_solid0x510afa0 bba  mi (    -29.999   -29.999   -75.874) mx (     29.999    29.999   -15.875) si (     59.998    59.998    59.999) ce  (   0.00    0.00  -45.87   30.00)  26
    27                 PMT_3inch_pmt_solid0x510aae0 bba  mi (    -40.001   -40.001   -75.876) mx (     40.001    40.001    40.001) si (     80.002    80.002   115.877) ce  (   0.00    0.00  -17.94   57.94)  27
    28                     sChimneyAcrylic0x5b310c0 bba  mi (   -520.000  -520.000  -300.000) mx (    520.000   520.000   300.000) si (   1040.000  1040.000   600.000) ce  (   0.00    0.00    0.00  520.00)  28
    29                          sChimneyLS0x5b312e0 bba  mi (   -400.000  -400.000 -1965.000) mx (    400.000   400.000  1965.000) si (    800.000   800.000  3930.000) ce  (   0.00    0.00    0.00 1965.00)  29
    30                       sChimneySteel0x5b314f0 bba  mi (   -405.000  -405.000 -1665.000) mx (    405.000   405.000  1665.000) si (    810.000   810.000  3330.000) ce  (   0.00    0.00    0.00 1665.00)  30
    31                          sWaterTube0x5b30eb0 bba  mi (   -520.000  -520.000 -1965.000) mx (    520.000   520.000  1965.000) si (   1040.000  1040.000  3930.000) ce  (   0.00    0.00    0.00 1965.00)  31
    32                        svacSurftube0x5b3bf50 bba  mi (     -4.000    -4.000    -4.000) mx (      4.000     4.000     4.000) si (      8.000     8.000     8.000) ce  (   0.00    0.00    0.00    4.00)  32
    33                           sSurftube0x5b3ab80 bba  mi (     -5.000    -5.000    -5.000) mx (      5.000     5.000     5.000) si (     10.000    10.000    10.000) ce  (   0.00    0.00    0.00    5.00)  33
    34                         sInnerWater0x4bd3660 bba  mi ( -20050.000-20050.000-20050.000) mx (  20050.000 20050.000 21750.000) si (  40100.000 40100.000 41800.000) ce  (   0.00    0.00  850.00 20900.00)  34
    35                      sReflectorInCD0x4bd3040 bba  mi ( -20052.000-20052.000-20052.000) mx (  20052.000 20052.000 21750.000) si (  40104.000 40104.000 41802.000) ce  (   0.00    0.00  849.00 20901.00)  35
    36                     sOuterWaterPool0x4bd2960 bba  mi ( -21750.000-21750.000-21750.000) mx (  21750.000 21750.000 21750.000) si (  43500.000 43500.000 43500.000) ce  (   0.00    0.00    0.00 21750.00)  36
    37                         sPoolLining0x4bd1eb0 bba  mi ( -21753.000-21753.000-21753.000) mx (  21753.000 21753.000 21750.000) si (  43506.000 43506.000 43503.000) ce  (   0.00    0.00   -1.50 21753.00)  37
    38                         sBottomRock0x4bcd770 bba  mi ( -24750.000-24750.000-24750.000) mx (  24750.000 24750.000 21750.000) si (  49500.000 49500.000 46500.000) ce  (   0.00    0.00 -1500.00 24750.00)  38
    39                              sWorld0x4bc2350 bba  mi ( -60000.000-60000.000-60000.000) mx (  60000.000 60000.000 60000.000) si ( 120000.000120000.000120000.000) ce  (   0.00    0.00    0.00 60000.00)  39





mergeVolume will transform vertices and applyPlacementTransform just need to set transform on the volume
----------------------------------------------------------------------------------------------------------------

* GMergedMesh::mergeVolume called by GMergedMesh::combine will transform GMesh verts and applyPlacementTransform
  to GParts analytic according to the transform on the GVolume

* so this means can setup universe appropriate for an origin centered proxied in solid 
  and then in the combine apply the requisite transform to make it so 

::

     362 void GMergedMesh::mergeVolume( GVolume* volume, bool selected, unsigned verbosity )
     363 {
     364     GNode* node = static_cast<GNode*>(volume);
     365     GNode* base = getCurrentBase();
     366     unsigned ridx = volume->getRepeatIndex() ;
     367 
     368     GMatrixF* transform = base ? volume->getRelativeTransform(base) : volume->getTransform() ;     // base or root relative global transform
     369 


Need to come up with the centering transform for the proxied in volume::

    836 GMergedMesh* GGeoTest::combineVolumes(GMergedMesh* mm0)
    837 {
    838     std::vector<GVolume*>& volumes = m_nodelib->getVolumes();
    839 
    840     LOG(LEVEL) << "[" ;
    841 
    842     GMergedMesh* tri = GMergedMesh::combine( 0, mm0, volumes, m_verbosity );
    843 
    844     unsigned nelem = volumes.size() ;
    845     GTransforms* txf = GTransforms::make(nelem); // identities
    846     GIds*        aii = GIds::make(nelem);        // placeholder (n,4) of zeros
    847 


* those are instance transforms, not them : the ones on the volumes




Am I picking up the right mm0 ?
--------------------------------------

::

    078 GMergedMesh* OpticksHub::getMergedMesh( unsigned index )
     79 {
     80     GGeoBase* ggb = getGGeoBase();  // 3-way   m_geotest/m_ggeo/m_gscene
     81     return ggb->getMergedMesh(index);
     82 }


    600 void OpticksHub::registerGeometry()
    601 {
    602     LOG(LEVEL) << "[" ;
    603 
    604     const char* ggb = getIdentifier();
    605     LOG(fatal) << " ggb " << ggb ; 
    606     GMergedMesh* mm0 = getMergedMesh(0);
    607 
    608     assert(mm0);
    609     m_aim->registerGeometry( mm0 );
    610     LOG(LEVEL) << "]" ; 
    611 }   


    080 GMergedMesh*      GGeoTest::getMergedMesh(unsigned index) const { return m_geolib->getMergedMesh(index) ; }

    101 GGeoTest::GGeoTest(Opticks* ok, GGeoBase* basis)
    102     :
    103     m_ok(ok),
    104     m_dbggeotest(ok->isDbgGeoTest()),    // --dbggeotest
    105     m_config_(ok->getTestConfig()),
    106     m_config(new NGeoTestConfig(m_config_)),
    107     m_verbosity(m_ok->isDbgGeoTest() ? 10 : m_config->getVerbosity()),
    108     m_resource(ok->getResource()),
    109     m_dbgbnd(m_ok->isDbgBnd()),
    110     m_dbganalytic(m_ok->isDbgAnalytic()),
    111     m_lodconfig(ok->getLODConfig()),
    112     m_lod(ok->getLOD()),
    113     m_analytic(m_config->getAnalytic()),
    114     m_csgpath(m_config->getCSGPath()),
    115     m_test(true),
    116     m_basis(basis),
    117     m_pmtlib(basis->getPmtLib()),
    118     m_meshlib(basis->getMeshLib()),
    119     m_mlib(new GMaterialLib(m_ok, basis->getMaterialLib())),
    120     m_slib(new GSurfaceLib(m_ok, basis->getSurfaceLib())),
    121     m_bndlib(new GBndLib(m_ok, m_mlib, m_slib)),
    122     m_geolib(new GGeoLib(m_ok,m_analytic,m_bndlib)),
    123     m_nodelib(new GNodeLib(m_ok, m_analytic, m_test, basis->getNodeLib() )),
    124     m_maker(new GMaker(m_ok, m_bndlib, m_meshlib)),
    125     m_csglist(m_csgpath ? NCSGList::Load(m_csgpath, m_verbosity ) : NULL),
    126     m_err(0)
    127 {

