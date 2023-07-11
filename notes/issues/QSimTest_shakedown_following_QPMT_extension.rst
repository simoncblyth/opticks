QSimTest_shakedown_following_QPMT_extension
============================================

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


