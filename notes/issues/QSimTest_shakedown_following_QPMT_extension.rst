QSimTest_shakedown_following_QPMT_extension
============================================

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



WIP : need lpmtid GPU side for QPMT
---------------------------------------

::

    ct ; ./CSGFoundry_py_test.sh

    cf.inst[:,:,3].view(np.int32)
    [[    0     0    -1    -1]
     [    1     1     0     0]
     [    2     1     1     1]
     [    3     1     2     2]
     [    4     1     3     3]
     ...
     [48472     9    -1    -1]
     [48473     9    -1    -1]
     [48474     9    -1    -1]
     [48475     9    -1    -1]
     [48476     9    -1    -1]]

    In [1]: cf.inst.shape
    Out[1]: (48477, 4, 4)

    In [2]: sensor_identifier = cf.inst[:,2,3].view(np.int32) ; sensor_identifier
    Out[2]: array([-1,  0,  1,  2,  3, ..., -1, -1, -1, -1, -1], dtype=int32)


    In [1]: np.where( sensor_identifier == -1 )
    Out[1]: (array([    0, 25601, 25602, 25603, 25604, ..., 48472, 48473, 48474, 48475, 48476]),)

    In [2]: np.where( sensor_identifier == -1 )[0] 
    Out[2]: array([    0, 25601, 25602, 25603, 25604, ..., 48472, 48473, 48474, 48475, 48476])

    In [3]: np.where( sensor_identifier == -1 )[0].size
    Out[3]: 20477

    In [4]: np.where( sensor_index == -1 )[0].size
    Out[4]: 20477

    In [5]: sensor_identifier.size
    Out[5]: 48477

    In [6]: np.where( np.logical_and( sensor_identifier == sensor_index, sensor_index > 0 ) )
    Out[6]: (array([2, 3, 4]),)


WIP : Not getting expected sensor_id
---------------------------------------

::

    cf.inst[:,:,3].view(np.int32)
    [[    0     0    -1    -1]
     [    1     1     0     0]
     [    2     1     1     1]
     [    3     1     2     2]
     [    4     1     3     3]
     ...
     [48472     9    -1    -1]
     [48473     9    -1    -1]
     [48474     9    -1    -1]
     [48475     9    -1    -1]
     [48476     9    -1    -1]]
    (sid.min(), sid.max())
    (-1, 309883)
    (six.min(), six.max())
    (-1, 27999)
    np.c_[ugas,ngas,cf.mmlabel] 
    [[0 1 '2977:sWorld']
     [1 25600 '5:PMT_3inch_pmt_solid']
     [2 12615 '9:NNVTMCPPMTsMask_virtual']
     [3 4997 '12:HamamatsuR12860sMask_virtual']
     [4 2400 '6:mask_PMT_20inch_vetosMask_virtual']
     [5 590 '1:sStrutBallhead']
     [6 590 '1:uni1']
     [7 590 '1:base_steel']
     [8 590 '1:uni_acrylic1']
     [9 504 '130:sPanel']]
    np.c_[np.unique(sid[gas==0],return_counts=True)]     
    [[-1  1]]
    np.c_[np.unique(sid[gas==1],return_counts=True)]     
    [[     0    127]
     [     1    127]
     [     2    127]
     [     3    127]
     [     4      1]
     ...
     [307479      1]
     [307480      1]
     [307481      1]
     [307482      1]
     [307483      1]]
    np.c_[np.unique(sid[gas==2],return_counts=True)]     
    [[   -1 12615]]
    np.c_[np.unique(sid[gas==3],return_counts=True)]     
    [[  -1 4997]]
    np.c_[np.unique(sid[gas==4],return_counts=True)]     
    [[307484      1]
     [307485      1]
     [307486      1]
     [307487      1]
     [307488      1]
     ...
     [309879      1]
     [309880      1]
     [309881      1]
     [309882      1]
     [309883      1]]
    np.c_[np.unique(sid[gas==5],return_counts=True)]     
    [[ -1 590]]
    np.c_[np.unique(sid[gas==6],return_counts=True)]     
    [[ -1 590]]
    np.c_[np.unique(sid[gas==7],return_counts=True)]     
    [[ -1 590]]
    np.c_[np.unique(sid[gas==8],return_counts=True)]     
    [[ -1 590]]
    np.c_[np.unique(sid[gas==9],return_counts=True)]     
    [[ -1 504]]

    In [1]:                    


::

     40 const U4SensorIdentifier* G4CXOpticks::SensorIdentifier = nullptr ;
     41 void G4CXOpticks::SetSensorIdentifier( const U4SensorIdentifier* sid ){ SensorIdentifier = sid ; }  // static 


::

    240 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    241 {
    242     LOG(LEVEL) << " G4VPhysicalVolume world " << world ;
    243     assert(world);
    244     wd = world ;
    245 
    246     assert(sim && "sim instance should have been created in ctor" );
    247 
    248     stree* st = sim->get_tree();
    249     // TODO: sim argument, not st : or do SSim::Create inside U4Tree::Create 
    250     tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    251 
    252 
    253     // GGeo creation done when starting from a gdml or live G4,  still needs Opticks instance
    254     Opticks::Configure("--gparts_transform_offset --allownokey" );
    255 
    256     GGeo* gg_ = X4Geo::Translate(wd) ;
    257 
    258 
    259     setGeometry(gg_);
    260 }

::

    104     static U4Tree* Create( stree* st, const G4VPhysicalVolume* const top, const U4SensorIdentifier* sid=nullptr );
    105     U4Tree(stree* st, const G4VPhysicalVolume* const top=nullptr, const U4SensorIdentifier* sid=nullptr );
    106     void init();


    174 inline U4Tree::U4Tree(stree* st_, const G4VPhysicalVolume* const top_,  const U4SensorIdentifier* sid_ )
    175     :
    176     st(st_),
    177     top(top_),
    178     sid(sid_ ? sid_ : new U4SensorIdentifierDefault),
    179     level(st->level),
    180     num_surfaces(-1),
    181     rayleigh_table(CreateRayleighTable()),
    182     scint(nullptr)
    183 {
    184     init();
    185 }


Add sensor name dumping
--------------------------

Original sensor_id look OK, so maybe issue with reordering ::

    U4SensorIdentifierDefault::getIdentity copyno 325590 num_sd 2 sensor_id 325590 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325591 num_sd 2 sensor_id 325591 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325592 num_sd 2 sensor_id 325592 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325593 num_sd 2 sensor_id 325593 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325594 num_sd 2 sensor_id 325594 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325595 num_sd 2 sensor_id 325595 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325596 num_sd 2 sensor_id 325596 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325597 num_sd 2 sensor_id 325597 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325598 num_sd 2 sensor_id 325598 pvn PMT_3inch_log_phys
    U4SensorIdentifierDefault::getIdentity copyno 325599 num_sd 2 sensor_id 325599 pvn PMT_3inch_log_phys

    U4SensorIdentifierDefault::getIdentity copyno 2 num_sd 2 sensor_id 2 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 4 num_sd 2 sensor_id 4 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 6 num_sd 2 sensor_id 6 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 21 num_sd 2 sensor_id 21 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 22 num_sd 2 sensor_id 22 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 23 num_sd 2 sensor_id 23 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 24 num_sd 2 sensor_id 24 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 25 num_sd 2 sensor_id 25 pvn pLPMT_NNVT_MCPPMT
    ...
    U4SensorIdentifierDefault::getIdentity copyno 17586 num_sd 2 sensor_id 17586 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 17587 num_sd 2 sensor_id 17587 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 17588 num_sd 2 sensor_id 17588 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 17589 num_sd 2 sensor_id 17589 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 17590 num_sd 2 sensor_id 17590 pvn pLPMT_NNVT_MCPPMT
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 2 sensor_id 0 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 1 num_sd 2 sensor_id 1 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 2 sensor_id 3 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 5 num_sd 2 sensor_id 5 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 7 num_sd 2 sensor_id 7 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 8 num_sd 2 sensor_id 8 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 9 num_sd 2 sensor_id 9 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 10 num_sd 2 sensor_id 10 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 11 num_sd 2 sensor_id 11 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 12 num_sd 2 sensor_id 12 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 13 num_sd 2 sensor_id 13 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 14 num_sd 2 sensor_id 14 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 15 num_sd 2 sensor_id 15 pvn pLPMT_Hamamatsu_R12860
    ...
    U4SensorIdentifierDefault::getIdentity copyno 17606 num_sd 2 sensor_id 17606 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 17607 num_sd 2 sensor_id 17607 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 17608 num_sd 2 sensor_id 17608 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 17609 num_sd 2 sensor_id 17609 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 17610 num_sd 2 sensor_id 17610 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 17611 num_sd 2 sensor_id 17611 pvn pLPMT_Hamamatsu_R12860
    U4SensorIdentifierDefault::getIdentity copyno 30000 num_sd 2 sensor_id 30000 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30001 num_sd 2 sensor_id 30001 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30002 num_sd 2 sensor_id 30002 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30003 num_sd 2 sensor_id 30003 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30004 num_sd 2 sensor_id 30004 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30005 num_sd 2 sensor_id 30005 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30006 num_sd 2 sensor_id 30006 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 30007 num_sd 2 sensor_id 30007 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    ...
    U4SensorIdentifierDefault::getIdentity copyno 32389 num_sd 2 sensor_id 32389 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32390 num_sd 2 sensor_id 32390 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32391 num_sd 2 sensor_id 32391 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32392 num_sd 2 sensor_id 32392 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32393 num_sd 2 sensor_id 32393 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32394 num_sd 2 sensor_id 32394 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32395 num_sd 2 sensor_id 32395 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32396 num_sd 2 sensor_id 32396 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32397 num_sd 2 sensor_id 32397 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32398 num_sd 2 sensor_id 32398 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 32399 num_sd 2 sensor_id 32399 pvn mask_PMT_20inch_vetolMaskVirtual_phys
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 64 sensor_id 0 pvn pPanel_0_f_
    U4SensorIdentifierDefault::getIdentity copyno 1 num_sd 64 sensor_id 1 pvn pPanel_1_f_
    U4SensorIdentifierDefault::getIdentity copyno 2 num_sd 64 sensor_id 2 pvn pPanel_2_f_
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 64 sensor_id 3 pvn pPanel_3_f_
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 64 sensor_id 0 pvn pPanel_0_f_
    ...
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 64 sensor_id 3 pvn pPanel_3_f_
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 64 sensor_id 0 pvn pPanel_0_f_
    U4SensorIdentifierDefault::getIdentity copyno 1 num_sd 64 sensor_id 1 pvn pPanel_1_f_
    U4SensorIdentifierDefault::getIdentity copyno 2 num_sd 64 sensor_id 2 pvn pPanel_2_f_
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 64 sensor_id 3 pvn pPanel_3_f_
    U4SensorIdentifierDefault::getIdentity copyno 0 num_sd 64 sensor_id 0 pvn pPanel_0_f_
    U4SensorIdentifierDefault::getIdentity copyno 1 num_sd 64 sensor_id 1 pvn pPanel_1_f_
    U4SensorIdentifierDefault::getIdentity copyno 2 num_sd 64 sensor_id 2 pvn pPanel_2_f_
    U4SensorIdentifierDefault::getIdentity copyno 3 num_sd 64 sensor_id 3 pvn pPanel_3_f_
    stree::add_inst i   0 gas_idx   1 nodes.size   25600
    stree::add_inst i   1 gas_idx   2 nodes.size   12615









TODO : mock qpmt.h landings with ART 4x4 collection into aux 
---------------------------------------------------------------



