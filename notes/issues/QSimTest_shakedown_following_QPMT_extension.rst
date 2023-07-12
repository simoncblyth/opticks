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




WIP : apply sevt.py machinery to mock_propagate
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

    In [15]: np.all(  t.f.prd[0] == t.f.prd[2] )
    Out[15]: True

    In [20]: np.all(  t.f.prd[0] == t.f.prd[7] )
    Out[20]: True

    In [21]: np.all(  t.f.prd[0] == t.f.prd[1] )
    Out[21]: False

    In [22]: t.f.prd[0]
    Out[22]:
    array([[[  0.,   0.,   1., 100.],
            [  0.,   0.,   0.,   0.]],

           [[  0.,   0.,   1., 200.],
            [  0.,   0.,   0.,   0.]],

           [[  0.,   0.,   1., 300.],
            [  0.,   0.,   0.,   0.]],

           [[  0.,   0.,   1., 400.],
            [  0.,   0.,   0.,   0.]],

           [[  0.,   0.,   0.,   0.],
            [  0.,   0.,   0.,   0.]]], dtype=float32)

    In [23]: t.f.prd[1]
    Out[23]:
    array([[[  0.,   0.,   1., 100.],
            [  0.,   0.,   0.,   0.]],

           [[  0.,   0.,   1., 200.],
            [  0.,   0.,   0.,   0.]],

           [[  0.,   0.,   1., 300.],
            [  0.,   0.,   0.,   0.]],

           [[  0.,   0.,   0.,   0.],
            [  0.,   0.,   0.,   0.]],

           [[  0.,   0.,   0.,   0.],
            [  0.,   0.,   0.,   0.]]], dtype=float32)

    In [24]: t.q
    Out[24]:
    array([[b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT AB                                                                                     '],
           [b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT BT SR                                                                                  '],
           [b'TO BT BT BT SR                                                                                  ']], dtype='|S96')

    In [25]:




TODO : mock_propagate SD as prelim to qpmt.h landings
-------------------------------------------------------



TODO : mock qpmt.h landings with ART 4x4 collection into aux 
---------------------------------------------------------------






