QSimTest_shakedown_following_QPMT_extension
============================================

Context
---------

* next :doc:`review_sensor_id` (FIXED sensor info) 


FIXED lack of metadata in new bnd
-----------------------------------

::

   ./QSimTest.sh 


    2023-07-11 02:48:49.305 INFO  [53727] [QSim::UploadComponents@130] QOptical optical NP  dtype <i4(52, 4, 4, ) size 832 uifc i ebyte 4 shape.size 3 data.size 3328 meta.size 0 names.size 0
    2023-07-11 02:48:49.327 FATAL [53727] [QBnd::MakeBoundaryTex@135]  buf_has_meta FAIL : domain metadata is required to create texture  buf.desc NP  dtype <f4(52, 4, 2, 761, 4, ) size 1266304 uifc f ebyte 4 shape.size 5 data.size 5065216 meta.size 0 names.size 52
    QSimTest: /data/blyth/junotop/opticks/qudarap/QBnd.cc:136: static QTex<float4>* QBnd::MakeBoundaryTex(const NP*): Assertion `buf_has_meta' failed.
    ./QSimTest.sh: line 146: 53727 Aborted                 (core dumped) $bin
    === eprd.sh : run error



::

    #3  0x00007ffff545c252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff77cc663 in QBnd::MakeBoundaryTex (buf=0x8ade400) at /data/blyth/junotop/opticks/qudarap/QBnd.cc:136
    #5  0x00007ffff77cc1dd in QBnd::QBnd (this=0x8add4b0, buf=0x29dbe80) at /data/blyth/junotop/opticks/qudarap/QBnd.cc:73
    #6  0x00007ffff7774a4a in QSim::UploadComponents (ssim=0x773b40) at /data/blyth/junotop/opticks/qudarap/QSim.cc:132
    #7  0x000000000041a95f in main (argc=1, argv=0x7fffffff5588) at /data/blyth/junotop/opticks/qudarap/tests/QSimTest.cc:688
    (gdb) f 5
    #5  0x00007ffff77cc1dd in QBnd::QBnd (this=0x8add4b0, buf=0x29dbe80) at /data/blyth/junotop/opticks/qudarap/QBnd.cc:73
    73	    tex(MakeBoundaryTex(src)),
    (gdb) f 4
    #4  0x00007ffff77cc663 in QBnd::MakeBoundaryTex (buf=0x8ade400) at /data/blyth/junotop/opticks/qudarap/QBnd.cc:136
    136	    assert( buf_has_meta ); 
    (gdb) list
    131	
    132	    QTex<float4>* btex = new QTex<float4>(nx, ny, values, filterMode, normalizedCoords ) ; 
    133	
    134	    bool buf_has_meta = buf->has_meta() ;
    135	    LOG_IF(fatal, !buf_has_meta) << " buf_has_meta FAIL : domain metadata is required to create texture  buf.desc " << buf->desc() ;  
    136	    assert( buf_has_meta ); 
    137	
    138	    quad domainX ; 
    139	    domainX.f.x = buf->get_meta<float>("domain_low",   0.f ); 
    140	    domainX.f.y = buf->get_meta<float>("domain_high",  0.f ); 
    (gdb) 


New bnd misses some metadata::

    1082     bool fine = nl == sdomain::FINE_DOMAIN_LENGTH ;
    1083     glm::vec4 dom = fine ? Opticks::GetFineDomainSpec() : Opticks::GetCoarseDomainSpec()  ;
    1084 
    ...
    1102     NPY<double>* wav = NPY<double>::make( ni, nj, nk, nl, nm) ;
    1103     wav->fill( GSurfaceLib::SURFACE_UNSET );
    1104 
    1105     double domain_low = dom.x ;
    1106     double domain_high = dom.y ;
    1107     double domain_step = dom.z ;
    1108     double domain_range = dom.w ;
    1109 
    1110     wav->setMeta("domain_low",   domain_low );
    1111     wav->setMeta("domain_high",  domain_high );

::

     180 glm::vec4 Opticks::GetFineDomainSpec()
     181 {
     182     glm::vec4 bd ;
     183 
     184     bd.x = sdomain::DOMAIN_LOW ;
     185     bd.y = sdomain::DOMAIN_HIGH ;
     186     bd.z = sdomain::FINE_DOMAIN_STEP  ;
     187     bd.w = sdomain::DOMAIN_HIGH - sdomain::DOMAIN_LOW ;
     188 
     189     return bd ;
     190 }




FIXED : Outdated bnd name in QPrd defaults
------------------------------------------------

After fixing that::


    (gdb) bt
    #0  0x00007ffff5463387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5464a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff545c1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff545c252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff77cf547 in SBnd::getBoundaryIndices (this=0x8b1ada0, bnd_idx=..., 
        bnd_sequence=0x7ffff78622e8 "Acrylic///LS,Water///Acrylic,Water///Pyrex,Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum", delim=44 ',')
        at /data/blyth/junotop/ExternalLibs/opticks/head/include/SysRap/SBnd.h:165
    #5  0x00007ffff77ce802 in QPrd::init (this=0x8b28e30) at /data/blyth/junotop/opticks/qudarap/QPrd.cc:51
    #6  0x00007ffff77ce3c1 in QPrd::QPrd (this=0x8b28e30) at /data/blyth/junotop/opticks/qudarap/QPrd.cc:28
    #7  0x0000000000419e7c in QSimTest::EventConfig (type=35) at /data/blyth/junotop/opticks/qudarap/tests/QSimTest.cc:567
    #8  0x000000000041a969 in main (argc=1, argv=0x7fffffff5168) at /data/blyth/junotop/opticks/qudarap/tests/QSimTest.cc:690
    (gdb) 



The hardcoded mock boundaries need updating for current geom::

    (gdb) f 8
    #8  0x000000000041a969 in main (argc=1, argv=0x7fffffff5168) at /data/blyth/junotop/opticks/qudarap/tests/QSimTest.cc:690
    690	    QSimTest::EventConfig(type)  ;  // must be after QBnd instanciation and before SEvt instanciation
    (gdb) f 7
    #7  0x0000000000419e7c in QSimTest::EventConfig (type=35) at /data/blyth/junotop/opticks/qudarap/tests/QSimTest.cc:567
    567	        QPrd* prd = new QPrd ; 
    (gdb) f 6
    #6  0x00007ffff77ce3c1 in QPrd::QPrd (this=0x8b28e30) at /data/blyth/junotop/opticks/qudarap/QPrd.cc:28
    28	    init(); 
    (gdb) f 5
    #5  0x00007ffff77ce802 in QPrd::init (this=0x8b28e30) at /data/blyth/junotop/opticks/qudarap/QPrd.cc:51
    51	    sbn->getBoundaryIndices( bnd_idx, bnd_sequence, ',' ); 
    (gdb) f 4
    #4  0x00007ffff77cf547 in SBnd::getBoundaryIndices (this=0x8b1ada0, bnd_idx=..., 
        bnd_sequence=0x7ffff78622e8 "Acrylic///LS,Water///Acrylic,Water///Pyrex,Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum", delim=44 ',')
        at /data/blyth/junotop/ExternalLibs/opticks/head/include/SysRap/SBnd.h:165
    165	        assert( bidx != MISSING ); 
    (gdb) 



FIXED : Changes to input photon setup were not handled by mock_propagate
------------------------------------------------------------------------------


Hmm, input photon issue maybe::

    (gdb) bt
    #0  0x00007ffff5462387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5463a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff545b1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff545b252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7777caa in QSim::mock_propagate (this=0x90f74b0, prd=0x90f8380, type=35)
        at /data/blyth/junotop/opticks/qudarap/QSim.cc:931
    #5  0x00000000004196ec in QSimTest::mock_propagate (this=0x7fffffff4bc0)
        at /data/blyth/junotop/opticks/qudarap/tests/QSimTest.cc:485
    #6  0x000000000041a69b in QSimTest::main (this=0x7fffffff4bc0) at /data/blyth/junotop/opticks/qudarap/tests/QSimTest.cc:670
    #7  0x000000000041a99e in main (argc=1, argv=0x7fffffff5588) at /data/blyth/junotop/opticks/qudarap/tests/QSimTest.cc:696
    (gdb) f 6 
    #6  0x000000000041a69b in QSimTest::main (this=0x7fffffff4bc0) at /data/blyth/junotop/opticks/qudarap/tests/QSimTest.cc:670
    670	                                                mock_propagate()              ; break ; 
    (gdb) f 5
    #5  0x00000000004196ec in QSimTest::mock_propagate (this=0x7fffffff4bc0)
        at /data/blyth/junotop/opticks/qudarap/tests/QSimTest.cc:485
    485	    qs->mock_propagate( prd, type ); 
    (gdb) f 4
    #4  0x00007ffff7777caa in QSim::mock_propagate (this=0x90f74b0, prd=0x90f8380, type=35)
        at /data/blyth/junotop/opticks/qudarap/QSim.cc:931
    931	    assert( num_photon == num_p ); 
    (gdb) p num_photon
    $1 = 0
    (gdb) p num_p
    $2 = 8
    (gdb) 



HMM looks like QEvent::setGenstep never called... where does thap normally 
happen ?  Appears the consistency check should be after the setGenstep call in QSim::mock_propagate. 

::

    2023-07-11 19:24:26.756 INFO  [139071] [QSimTest::mock_propagate@476]  num 8 p (8, 4, 4, )
    2023-07-11 19:24:26.756 INFO  [139071] [QSimTest::mock_propagate@481]  bounce_max 4
    2023-07-11 19:24:26.756 INFO  [139071] [QSimTest::mock_propagate@484]  prd (8, 4, 2, 4, )
    2023-07-11 19:24:26.756 INFO  [139071] [QEvent::setGenstep@159] 
    2023-07-11 19:24:26.756 FATAL [139071] [QEvent::setGenstep@162] Must SEvt::AddGenstep before calling QEvent::setGenstep 
    QSimTest: /data/blyth/junotop/opticks/qudarap/QSim.cc:934: void QSim::mock_propagate(const NP*, unsigned int): Assertion `rc == 0' failed.
    ./QSimTest.sh: line 145: 139071 Aborted                 (core dumped) $bin
    === eprd.sh : run error
    N[blyth@localhost tests]$ 


Hmm probably changes to input photon genstep tee up are 
not yet accomodated by QSim::mock_propagate. 


DONE : review how input photons handled in ordinary running, then bring over similar to mock_propagate
---------------------------------------------------------------------------------------------------------

::

     407 /**
     408 SEvt::setFrame
     409 ------------------
     410 
     411 As it is necessary to have the geometry to provide the frame this 
     412 is now split from eg initInputPhotons.  
     413 
     414 **simtrace running**
     415     MakeCenterExtentGensteps based on the given frame. 
     416 
     417 **simulate inputphoton running**
     418     MakeInputPhotonGenstep and m2w (model-2-world) 
     419     transforms the photons using the frame transform
     420 
     421 Formerly(?) for simtrace and input photon running with or without a transform 
     422 it was necessary to call this for every event due to the former call to addFrameGenstep, 
     423 but now that the genstep setup is moved to SEvt::BeginOfEvent it is only needed 
     424 to call this for each frame, usually once only. 
     425 
     426 **/
     427 
     428 
     429 void SEvt::setFrame(const sframe& fr )
     430 {
     431     frame = fr ;
     432     // former call to addFrameGenstep() is relocated to SEvt::BeginOfEvent
     433     transformInputPhoton();  
     434 }   





DONE : Checking bnd surface names : why no specials ? Because prefixes on opticalsurface NOT skinsurface/bordersurface
------------------------------------------------------------------------------------------------------------------------

Can see that by grepping the gdml::

    GEOM top 

    epsilon:V1J009 blyth$ grep @Hama origin.gdml
        <opticalsurface finish="1" model="0" name="@HamamatsuR12860_PMT_20inch_Mirror_opsurf" type="0" value="0.999">
        <skinsurface name="HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf" surfaceproperty="@HamamatsuR12860_PMT_20inch_Mirror_opsurf">
    epsilon:V1J009 blyth$ grep @NNVT origin.gdml
        <opticalsurface finish="1" model="0" name="@NNVTMCPPMT_PMT_20inch_Mirror_opsurf" type="0" value="0.999">
        <skinsurface name="NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf" surfaceproperty="@NNVTMCPPMT_PMT_20inch_Mirror_opsurf">
    epsilon:V1J009 blyth$ grep \#NNVT origin.gdml
    epsilon:V1J009 blyth$ grep \#Hama origin.gdml


::


    GEOM top 

    epsilon:V1J009 blyth$ grep opticalsurface origin.gdml | grep name
        <opticalsurface finish="3" model="1" name="UpperChimneyTyvekOpticalSurface" type="0" value="0.2">
        <opticalsurface finish="3" model="1" name="opStrutAcrylic" type="0" value="0.2">
        <opticalsurface finish="3" model="1" name="opStrut2Acrylic" type="0" value="0.2">
        <opticalsurface finish="3" model="1" name="opHamamatsuMask" type="0" value="0.2">
        <opticalsurface finish="1" model="0" name="@HamamatsuR12860_PMT_20inch_Mirror_opsurf" type="0" value="0.999">
        <opticalsurface finish="0" model="0" name="plateOpSurface" type="0" value="0.999">
        <opticalsurface finish="3" model="1" name="opNNVTMask" type="0" value="0.2">
        <opticalsurface finish="1" model="0" name="@NNVTMCPPMT_PMT_20inch_Mirror_opsurf" type="0" value="0.999">
        <opticalsurface finish="0" model="0" name="plateOpSurface" type="0" value="0.999">
        <opticalsurface finish="0" model="0" name="Photocathode_opsurf_3inch" type="0" value="1">
        <opticalsurface finish="0" model="0" name="Absorb_opsurf" type="0" value="1">
        <opticalsurface finish="3" model="1" name="ChimneySteelOpticalSurface" type="0" value="0.2">
        <opticalsurface finish="3" model="1" name="CDInnerTyvekOpticalSurface" type="0" value="0.2">
        <opticalsurface finish="0" model="0" name="Photocathode_opsurf" type="0" value="1">
        <opticalsurface finish="1" model="0" name="Mirror_opsurf" type="0" value="0.999">
        <opticalsurface finish="3" model="1" name="CDTyvekOpticalSurface" type="0" value="0.2">
    epsilon:V1J009 blyth$ 

::

    In [9]: np.c_[cf.sim.stree.suname[np.char.startswith(cf.sim.stree.suname, "Hama")]]
    Out[9]: 
    array([['HamamatsuR12860_PMT_20inch_dynode_plate_opsurface'],
           ['HamamatsuR12860_PMT_20inch_inner_ring_opsurface'],
           ['HamamatsuR12860_PMT_20inch_outer_edge_opsurface'],
           ['HamamatsuR12860_PMT_20inch_inner_edge_opsurface'],
           ['HamamatsuR12860_PMT_20inch_dynode_tube_opsurface'],
           ['HamamatsuR12860_PMT_20inch_grid_opsurface'],
           ['HamamatsuR12860_PMT_20inch_shield_opsurface'],
           ['HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf'],
           ['HamamatsuMaskOpticalSurface']], dtype='<U54')

    In [10]: np.c_[cf.sim.stree.suname[np.char.startswith(cf.sim.stree.suname, "NNVT")]]
    Out[10]: 
    array([['NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface'],
           ['NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface'],
           ['NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface'],
           ['NNVTMCPPMT_PMT_20inch_mcp_opsurface'],
           ['NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf'],
           ['NNVTMaskOpticalSurface']], dtype='<U54')




DONE : apply sevt.py machinery to mock_propagate
--------------------------------------------------------

::

    In [1]: t
    Out[1]: SEvt symbol t pid -1 opt  off [0. 0. 0.] t.f.base /tmp/blyth/opticks/GEOM/V1J009/QSimTest/ALL/000 

    In [2]: t.q 
    Out[2]: 
    array([[b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT AB                                                                                     '],
           [b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT BT SR                                                                                  ']], dtype='|S96')



DONE : prd[1] not same as the rest ? Because that prd is saved per step and photon idx 1 expired with AB
--------------------------------------------------------------------------------------------------------------

::

    In [24]: t.q
    Out[24]:
    array([[b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT AB                                                                                     '],
           [b'TO BT BT BT SR                                                                                  '],



WIP : mock_propagate SD as prelim to qpmt.h landings
-------------------------------------------------------



qsim::mock_propagate looks very similar to qsim::simulate by design. 

::

    1429 inline QSIM_METHOD void qsim::mock_propagate( sphoton& p, const quad2* mock_prd, curandStateXORWOW& rng, unsigned idx )
    1430 {
    1431     p.set_flag(TORCH);  // setting initial flag : in reality this should be done by generation
    1432 
    1433     qsim* sim = this ;
    1434 
    1435     sctx ctx = {} ;
    1436     ctx.p = p ;     // Q: Why is this different from CSGOptiX7.cu:simulate ? A: Presumably due to input photon. 
    1437     ctx.evt = evt ;
    1438     ctx.idx = idx ;
    1439 
    1440     int command = START ;
    1441     int bounce = 0 ;
    1442 #ifndef PRODUCTION
    1443     ctx.point(bounce);
    1444 #endif
    1445 
    1446     while( bounce < evt->max_bounce )
    1447     {
    1448         ctx.prd = mock_prd + (evt->max_bounce*idx+bounce) ;
    1449         if( ctx.prd->boundary() == 0xffffu ) break ;   // SHOULD NEVER HAPPEN : propagate can do nothing meaningful without      a boundary 
    1450 #ifndef PRODUCTION
    1451         ctx.trace(bounce);
    1452 #endif
    1453 
    1454 #ifdef DEBUG_PIDX
    1455         if(idx == base->pidx)
    1456         printf("//qsim.mock_propagate idx %d bounce %d evt.max_bounce %d prd.q0.f.xyzw (%10.4f %10.4f %10.4f %10.4f) \n",
    1457              idx, bounce, evt->max_bounce, ctx.prd->q0.f.x, ctx.prd->q0.f.y, ctx.prd->q0.f.z, ctx.prd->q0.f.w );
    1458 #endif
    1459         command = sim->propagate(bounce, rng, ctx );
    1460         bounce++;
    1461 #ifndef PRODUCTION
    1462         ctx.point(bounce);
    1463 #endif
    1464         if(command == BREAK) break ;
    1465     }
    1466 #ifndef PRODUCTION
    1467     ctx.end();
    1468 #endif
    1469     evt->photon[idx] = ctx.p ;
    1470 }



HMM: how to get mock_propagate to mimmick upper hemi actions ? Need to mock 
more of the prd quad2::

    1482 inline QSIM_METHOD int qsim::propagate(const int bounce, curandStateXORWOW& rng, sctx& ctx )
    1483 {
    1484     const unsigned boundary = ctx.prd->boundary() ;
    1485     const unsigned identity = ctx.prd->identity() ;
    1486     const unsigned iindex = ctx.prd->iindex() ;
    1487     const float lposcost = ctx.prd->lposcost() ;
    1488 
    1489     const float3* normal = ctx.prd->normal();
    1490     float cosTheta = dot(ctx.p.mom, *normal ) ;
    1491 


::

    093 /**
     94 quad2
     95 -------
     96 
     97 ::
     98 
     99     +------------+------------+------------+---------------+
    100     | f:normal_x | f:normal_y | f:normal_z | f:distance    |
    101     +------------+------------+------------+---------------+
    102     | f:lposcost | u:iindex   | u:identity | u:boundary    |
    103     +------------+------------+------------+---------------+
    104 
    105 
    106 lposcost
    107     Local position cos(theta) of intersect, 
    108     canonically calculated in CSGOptiX7.cu:__intersection__is
    109     normalize_z(ray_origin + isect.w*ray_direction )
    110     where normalize_z is v.z/sqrtf(dot(v, v)) 
    111 
    112     This is kinda imagining a sphere thru the intersection point 
    113     which is likely onto an ellipsoid or a box or anything 
    114     to provide a standard way of giving a z-polar measure.
    115 
    116 **/




TODO : mock qsim.h/qpmt.h landings with ART 4x4 collection into aux
------------------------------------------------------------------------


DONE : cx/CSGOptiX7.cu need full instance_id in the identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    447 extern "C" __global__ void __closesthit__ch()
    448 {
    449     unsigned iindex = optixGetInstanceIndex() ;    // 0-based index within IAS
    450     unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build 
    451     unsigned prim_idx = optixGetPrimitiveIndex() ; // GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    452 
    453     //unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ; 
    454     unsigned identity = instance_id ;  // CHANGED July 2023, as now carrying sensor_identifier, see sysrap/sqat4.h 
    455 



DONE :  change optical buf Payload_Y to the ems enum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     08 enum {
      9     smatsur_Material                       = 0,
     10     smatsur_NoSurface                      = 1,
     11     smatsur_Surface                        = 2,
     12     smatsur_Surface_zplus_sensor_A         = 3,
     13     smatsur_Surface_zplus_sensor_CustomART = 4
     14 };



sysrap/sstandard.h::

    365                 char OSN0 = *OSN.c_str() ;
    366                 int ems = smatsur::TypeFromChar(OSN0) ;
    367 
    368                 int Payload_Y = ems ;  
    369                 //int Payload_Y = Type ; 

::

    039 inline int smatsur::TypeFromChar(char OpticalSurfaceName0)
     40 {
     41     int type = -1  ;
     42     switch(OpticalSurfaceName0)
     43     {
     44         case '\0': type = smatsur_Material                       ; break ;
     45         case '-':  type = smatsur_NoSurface                      ; break ;
     46         case '@':  type = smatsur_Surface_zplus_sensor_CustomART ; break ;
     47         case '#':  type = smatsur_Surface_zplus_sensor_A         ; break ;
     48         default:   type = smatsur_Surface                        ; break ;
     49     }
     50     return type ;
     51 }




WIP : qsim.h qsim::propagate needs to branch on that enum 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 1st check consistency of optical.x and optical.y

::

    st ; ./stree_py_test.sh 

    In [4]: f.standard.optical.shape
    Out[4]: (52, 4, 4)


Find sensor bnd index::

    In [17]: np.c_[np.arange(len(f.standard.bnd_names)), f.standard.bnd_names]
    Out[17]: 
    array([['0', 'Galactic///Galactic'],
           ['1', 'Galactic///Rock'],
           ['2', 'Rock///Galactic'],
           ['3', 'Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air'],
           ['4', 'Rock///Rock'],
           ['5', 'Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air'],
           ['6', 'Air///Steel'],
           ['7', 'Air///Air'],
           ['8', 'Air///LS'],
           ['9', 'Air///Tyvek'],
           ['10', 'Air///Aluminium'],
           ['11', 'Aluminium///Adhesive'],
           ['12', 'Adhesive///TiO2Coating'],
           ['13', 'TiO2Coating///Scintillator'],
           ['14', 'Rock///Tyvek'],
           ['15', 'Tyvek//VETOTyvekSurface/vetoWater'],
           ['16', 'vetoWater///LatticedShellSteel'],
           ['17', 'vetoWater/CDTyvekSurface//Tyvek'],
           ['18', 'Tyvek//CDInnerTyvekSurface/Water'],
           ['19', 'Water///Acrylic'],
           ['20', 'Acrylic///LS'],
           ['21', 'LS///Acrylic'],
           ['22', 'LS///PE_PA'],
           ['23', 'Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel'],
           ['24', 'Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutSteel'],
           ['25', 'Water///Steel'],
           ['26', 'Water///Water'],
           ['27', 'Water///AcrylicMask'],
           ['28', 'Water/HamamatsuMaskOpticalSurface/HamamatsuMaskOpticalSurface/CDReflectorSteel'],
           ['29', 'Water///Pyrex'],
           ['30', 'Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum'],
           ['31', 'Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['32', 'Vacuum/HamamatsuR12860_PMT_20inch_outer_edge_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['33', 'Vacuum/HamamatsuR12860_PMT_20inch_inner_edge_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['34', 'Vacuum/HamamatsuR12860_PMT_20inch_inner_ring_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['35', 'Vacuum/HamamatsuR12860_PMT_20inch_dynode_tube_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['36', 'Vacuum/HamamatsuR12860_PMT_20inch_grid_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['37', 'Vacuum/HamamatsuR12860_PMT_20inch_shield_opsurface/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['38', 'Water/NNVTMaskOpticalSurface/NNVTMaskOpticalSurface/CDReflectorSteel'],
           ['39', 'Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum'],
           ['40', 'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_edge_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['41', 'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['42', 'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_tube_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['43', 'Vacuum/NNVTMCPPMT_PMT_20inch_mcp_opsurface/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Steel'],
           ['44', 'Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum'],
           ['45', 'Pyrex/PMT_3inch_absorb_logsurf2/PMT_3inch_absorb_logsurf1/Vacuum'],
           ['46', 'Water///LS'],
           ['47', 'Water/Steel_surface/Steel_surface/Steel'],
           ['48', 'vetoWater///Water'],
           ['49', 'Pyrex///Pyrex'],
           ['50', 'Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum'],
           ['51', 'Pyrex/PMT_20inch_veto_mirror_logsurf2/PMT_20inch_veto_mirror_logsurf1/Vacuum']], dtype='<U122')


    In [18]: f.standard.bnd_names[np.array([30,39])]
    Out[18]: 
    array(['Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum',
           'Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum'], dtype='<U122')


    In [20]: f.standard.optical[30]       ## THIS IS BEFORE THE Payload_Y change 
    Out[20]: 
    array([[18,  0,  0,  0],
           [36,  0,  1, 99],                   osur  expect the  
           [36,  0,  1, 99],                   isur 
           [17,  0,  0,  0]], dtype=int32)


    In [1]: f.standard.optical[30]     ## AFTER 
    Out[1]: 
    array([[18,  0,  0,  0],
           [36,  4,  1, 99],
           [36,  4,  1, 99],
           [17,  0,  0,  0]], dtype=int32)






    In [21]: f.standard.optical[39]
    Out[21]: 
    array([[18,  0,  0,  0],
           [38,  0,  1, 99],
           [38,  0,  1, 99],
           [17,  0,  0,  0]], dtype=int32)



DONE : Where Payload_X index is zero (meaning ordinary boundary) Payload_Y is always 1 : for NoSurface which is correct
------------------------------------------------------------------------------------------------------------------------------ 

* converse also try 

::

    In [11]: f.standard.optical[np.where( f.standard.optical[:,:,0] == 0 )]
    Out[11]: 
    array([[0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           ...
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0]], dtype=int32)


    In [15]: f.standard.optical[np.where( f.standard.optical[:,:,0] == 0 )].shape
    Out[15]: (55, 4)



FIXED : Too many optical ems 4 
--------------------------------------------------

* :doc:`optical_ems_4_getting_too_many_from_non_sensor_Vacuum_Steel_borders`



DONE : first cut qsim::propagate branching on the smatsur.h enum
------------------------------------------------------------------------

::

     08 enum {
     09     smatsur_Material                       = 0,
     10     smatsur_NoSurface                      = 1,
     11     smatsur_Surface                        = 2,
     12     smatsur_Surface_zplus_sensor_A         = 3,
     13     smatsur_Surface_zplus_sensor_CustomART = 4
     14 };
     15 

::

    In [15]: f.standard.optical[:,1,1]  ## osur
    Out[15]: array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2], dtype=int32)

    In [16]: f.standard.optical[:,2,1]  ## isur
    Out[16]: array([1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2], dtype=int32)


    In [17]: np.unique( f.standard.optical[:,1,1], return_counts=True )   ## NoSurface, Surface, Surface_zplus_sensor_CustomART
    Out[17]: (array([1, 2, 4], dtype=int32), array([29, 21,  2]))

    In [18]: np.unique( f.standard.optical[:,2,1], return_counts=True )
    Out[18]: (array([1, 2, 4], dtype=int32), array([42,  8,  2]))





::

    1482 inline QSIM_METHOD int qsim::propagate(const int bounce, curandStateXORWOW& rng, sctx& ctx )
    1483 {
    ...
    1484     const unsigned boundary = ctx.prd->boundary() ;
    1485     const unsigned identity = ctx.prd->identity() ;
    1486     const unsigned iindex = ctx.prd->iindex() ;
    1487     const float lposcost = ctx.prd->lposcost() ;
    1488 

    1509 
    1510     if( command == BOUNDARY )
    1511     {
    1512         command = ctx.s.optical.x == 0 ?
    1513                                       propagate_at_boundary( flag, rng, ctx )
    1514                                   :
    1515                                       propagate_at_surface( flag, rng, ctx )
    1516                                   ; 
    1517 
    1518 
    1519     }



DONE : added lposcost and identity mockup to SPrd.h 
-----------------------------------------------------


WIP : mock propagate testing the extended qsim.h 
----------------------------------------------------

::

    qt 
    ./QSimTest.sh


    In [1]: t
    Out[1]: SEvt symbol t pid -1 opt  off [0. 0. 0.] t.f.base /tmp/blyth/opticks/GEOM/V1J009/QSimTest/ALL/000

    In [2]: t.q
    Out[2]:
    array([[b'TO BT BT BT AB                                                                                  '],
           [b'TO BT BT AB                                                                                     '],
           [b'TO BT BT BT SD                                                                                  '],
           [b'TO BT BT BT BT                                                                                  '],
           [b'TO BT BT BT BR                                                                                  '],
           [b'TO BT BT BT BT                                                                                  '],
           [b'TO BT BT BT SA                                                                                  '],
           [b'TO BT BT BT SA                                                                                  ']], dtype='|S96')






::

    PIDX=2 ./QSimTest.sh 
    ...

    //qsim.propagate_at_boundary idx 2 TransCoeff     0.9979 n1c1     1.4779 n2c2     1.3492 E2_t (    0.0000,    1.0455) A_trans (    0.0000,    1.0000,    0.0000) 
    //qsim.propagate_at_boundary idx 2 u_boundary_burn     0.5237 u_reflect     0.9206 TransCoeff     0.9979 reflect 0 
    //qsim.propagate_at_boundary idx 2 reflect 0 tir 0 TransCoeff     0.9979 u_reflect     0.9206 
    //qsim.propagate_at_boundary idx 2 mom_1 (    0.0000     0.0000     1.0000) 
    //qsim.propagate_at_boundary idx 2 pol_1 (    0.0000     1.0000     0.0000) 
    //qsim.mock_propagate idx 2 bounce 3 evt.max_bounce 4 prd.q0.f.xyzw (    0.0000     0.0000    -1.0000   400.0000) 
    //qsim.propagate idx 2 bnc 3 cosTheta    -1.0000 dir (    0.0000     0.0000     1.0000) nrm (    0.0000     0.0000    -1.0000) 
    //qsim.propagate_to_boundary[ idx 2 u_absorption 0.30163988 logf(u_absorption) -1.19852138 absorption_length  1999.2292 absorption_distance 2396.118896 
    //qsim.propagate idx 2 bounce 3 command 3 flag 0 s.optical.x 38 
    //qsim.propagate_at_surface_CustomART idx 2 lpmtid 1001 mct  -1.000 ARTE[   0.476   0.220   0.780   0.436] 
    2023-07-16 15:55:22.382 INFO  [3054721] [SEvt::save@2709]  dir /tmp/blyth/opticks/GEOM/V1J009/QSimTest/ALL/000
    2023-07-16 15:55:22.389 INFO  [3054721] [SEvt::clear@1058] SEvt::clear


    PIDX=3 ./QSimTest.sh 
    ...

    //qsim.propagate_to_boundary[ idx 3 u_absorption 0.04417918 logf(u_absorption) -3.11950135 absorption_length  1999.2292 absorption_distance 6236.598145 
    //qsim.propagate idx 3 bounce 3 command 3 flag 0 s.optical.x 38 
    //qsim.propagate_at_surface_CustomART idx 3 lpmtid 1001 mct  -1.000 ARTE[   0.476   0.220   0.780   0.436] 
    //qsim.propagate_at_boundary idx 3 nrm   (    0.0000     0.0000    -1.0000) 
    //qsim.propagate_at_boundary idx 3 mom_0 (    0.0000     0.0000     1.0000) 
    //qsim.propagate_at_boundary idx 3 pol_0 (    0.0000     1.0000     0.0000) 
    //qsim.propagate_at_boundary idx 3 c1     1.0000 normal_incidence 1 
    //qsim.propagate_at_boundary idx 3 normal_incidence 1 p.pol (    0.0000,    1.0000,    0.0000) p.mom (    0.0000,    0.0000,    1.0000) o_normal (    0.0000,    0.0000,   -1.0000)
    //qsim.propagate_at_boundary idx 3 TransCoeff     0.7800 n1c1     1.4779 n2c2     1.0000 E2_t (    0.0000,    1.1929) A_trans (    0.0000,    1.0000,    0.0000) 
    //qsim.propagate_at_boundary idx 3 u_boundary_burn     0.9660 u_reflect     0.3781 TransCoeff     0.7800 reflect 0 
    //qsim.propagate_at_boundary idx 3 reflect 0 tir 0 TransCoeff     0.7800 u_reflect     0.3781 
    //qsim.propagate_at_boundary idx 3 mom_1 (    0.0000     0.0000     1.0000) 
    //qsim.propagate_at_boundary idx 3 pol_1 (    0.0000     1.0000     0.0000) 
    2023-07-16 15:58:07.919 INFO  [3057116] [SEvt::save@2709]  dir /tmp/blyth/opticks/GEOM/V1J009/QSimTest/ALL/000
    2023-07-16 15:58:07.930 INFO  [3057116] [SEvt::clear@1058] SEvt::clear



