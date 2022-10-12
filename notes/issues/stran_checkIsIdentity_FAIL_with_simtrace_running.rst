stran_checkIsIdentity_FAIL_with_simtrace_running
=======================================================

* from :doc:`simtrace_over_1M_unchecked_against_size_of_CurandState`

Capture backtrace from checkIsIdenity issue::

    N[blyth@localhost opticks]$ env | grep SIGINT
    stran_checkIsIdentity_SIGINT=1

    N[blyth@localhost opticks]$ MOI=Hama:0:1000 ~/opticks/g4cx/gxt.sh dbg 
    === cehigh : GEOM J004 MOI Hama:0:1000
    === cehigh_PMT
    CEHIGH_0=-8:8:0:0:-6:-4:1000:4
    gdb -ex r --args G4CXSimtraceTest -ex r
    Wed Oct 12 19:01:37 CST 2022

    stran.h : Tran::checkIsIdentity FAIL :  caller FromPair epsilon 1e-06 mxdif_from_identity 12075.9

    Program received signal SIGINT, Interrupt.
    0x00007fffecd0b4fb in raise () from /lib64/libpthread.so.0
    (gdb) bt
    #0  0x00007fffecd0b4fb in raise () from /lib64/libpthread.so.0
    #1  0x00007fffed64c502 in Tran<double>::checkIsIdentity (this=0x4027090, mat=105 'i', caller=0x7fffed72397c "FromPair", epsilon=9.9999999999999995e-07)
        at /data/blyth/junotop/opticks/sysrap/stran.h:638
    #2  0x00007fffed64b7f0 in Tran<double>::FromPair (t=0x2dd6cc0, v=0x2dd6d00, epsilon=9.9999999999999995e-07) at /data/blyth/junotop/opticks/sysrap/stran.h:712
    #3  0x00007fffed65ef4a in SFrameGenstep::MakeCenterExtentGensteps (fr=...) at /data/blyth/junotop/opticks/sysrap/SFrameGenstep.cc:160
    #4  0x00007fffed676682 in SEvt::setFrame (this=0x2dd6bf0, fr=...) at /data/blyth/junotop/opticks/sysrap/SEvt.cc:269
    #5  0x00007ffff7b8fcfb in G4CXOpticks::simtrace (this=0x7fffffff57a0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:391
    #6  0x0000000000408d52 in main (argc=3, argv=0x7fffffff5908) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimtraceTest.cc:27
    (gdb) 


::

    377 void G4CXOpticks::simtrace()
    378 {
    388     SEvt* sev = SEvt::Get();  assert(sev);
    390     sframe fr = fd->getFrame() ;  // depends on MOI, fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 
    391     sev->setFrame(fr);
    393     cx->setFrame(fr);
    397     qs->simtrace();
    399 }

    0256 void SEvt::setFrame(const sframe& fr )
     257 {
     258     frame = fr ;
     259 
     260     if(SEventConfig::IsRGModeSimtrace())
     261     {
     262         const char* frs = fr.get_frs() ; // nullptr when default -1 : meaning all geometry 
     263         if(frs)
     264         {
     265             LOG(LEVEL) << " non-default frs " << frs << " passed to SEvt::setReldir " ;
     266             setReldir(frs);
     267         }
     268 
     269         NP* gs = SFrameGenstep::MakeCenterExtentGensteps(frame);
     270         LOG(LEVEL) << " simtrace gs " << ( gs ? gs->sstr() : "-" ) ;
     271         addGenstep(gs);
     272 
     273         if(frame.is_hostside_simtrace()) setFrame_HostsideSimtrace();
     274 


    137 NP* SFrameGenstep::MakeCenterExtentGensteps(sframe& fr)
    138 {
    139     const float4& ce = fr.ce ;
    140     float gridscale = SSys::getenvfloat("GRIDSCALE", 0.1 ) ;
    141 
    142     // CSGGenstep::init
    143     std::vector<int> cegs ;
    144     GetGridConfig(cegs, "CEGS", ':', "16:0:9:1000" );
    145     fr.set_grid(cegs, gridscale);
    146 
    147     std::vector<float3> ce_offset ;
    148     CE_OFFSET(ce_offset, ce);
    149 
    150     LOG(LEVEL)
    151         << " ce " << ce
    152         << " ce_offset.size " << ce_offset.size()
    153         ;
    154 
    155 
    156     int ce_scale = SSys::getenvint("CE_SCALE", 1) ; // TODO: ELIMINATE AFTER RTP CHECK 
    157     LOG_IF(fatal, ce_scale == 0) << "warning CE_SCALE is not enabled : NOW THINK THIS SHOULD ALWAYS BE ENABLED " ;
    158 
    159 
    160     Tran<double>* geotran = Tran<double>::FromPair( &fr.m2w, &fr.w2m, 1e-6 );
    161 
    162 
    163     std::vector<NP*> gsl ;
    164     NP* gs_base = MakeCenterExtentGensteps(ce, cegs, gridscale, geotran, ce_offset, ce_scale );
    165     gsl.push_back(gs_base) ;
    166 


Save the unexpected transforms:: 

    +    if(!ok) 
    +    {
    +         std::cerr << "stran.h Tran::FromPair checkIsIdentity FAIL " << std::endl ; 
    +         const char* path = "/tmp/stran_FromPair_checkIsIdentity_FAIL.npy" ; 
    +         std::cerr << "stran.h save to path " << path << std::endl ; 
    +         tr->save_(path); 
    +    }




