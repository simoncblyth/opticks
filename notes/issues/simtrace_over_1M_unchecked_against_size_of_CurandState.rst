simtrace_over_1M_unchecked_against_size_of_CurandState
========================================================


::

    N[blyth@localhost opticks]$ MOI=Hama:0:1000 ~/opticks/g4cx/gxt.sh run 
                       BASH_SOURCE : /home/blyth/opticks/g4cx/../bin/GEOM_.sh 
                               gp_ : J004_GDMLPath 
                                gp :  
                               cg_ : J004_CFBaseFromGEOM 
                                cg : /home/blyth/.opticks/GEOM/J004 
                       TMP_GEOMDIR : /tmp/blyth/opticks/GEOM/J004 
                           GEOMDIR : /home/blyth/.opticks/GEOM/J004 
                       BASH_SOURCE : /home/blyth/opticks/g4cx/../bin/GEOM_.sh 

    === cehigh : GEOM J004 MOI Hama:0:1000
    === cehigh_PMT
    CEHIGH_0=-8:8:0:0:-6:-4:1000:4
    === /home/blyth/opticks/g4cx/gxt.sh : run G4CXSimtraceTest log G4CXSimtraceTest.log
    stran.h : Tran::checkIsIdentity FAIL :  caller FromPair epsilon 1e-06 mxdif_from_identity 12075.9
    stran.h Tran::FromPair checkIsIdentity FAIL 
    //CSGOptiX7.cu : simtrace idx 0 genstep_id 0 evt->num_simtrace 1212000 
    2022-10-12 18:51:32.472 INFO  [419174] [SEvt::save@1568]  dir /home/blyth/.opticks/GEOM/J004/G4CXSimtraceTest/Hama:0:1000
    N[blyth@localhost opticks]$ 
    N[blyth@localhost opticks]$ 


Note that even with only one CEHIGH block, the num_simtrace exceeds the default curandState. 

* This should trigger an assert. 


How to fix ?
----------------

* QRng was accepting a default path to a 1M file, but that is unconnected with SEventConfig::MaxSimtrace or MaxPhoton 
* clearly the QCurandState path accepted by QRng should be based on the SEvt/SEventConfig values of max_simtrace/max_photon 

* start by pulling SCurandState out of QCurandState so can do the CurandState checks/config from SEventConfig SCurandState 
* add SEventConfig::MaxCurandState which is max of MaxPhoton and MaxSimtrace


::

    commit 82f5bee43fb5f461ede5e489d39562c12e98a0d9 (HEAD -> master, origin/master, origin/HEAD)
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Wed Oct 12 17:17:19 2022 +0100

        connect SEventConfig to the SCurandState/QRng choice of curandState file to load

    commit 50f35cc11c35fa13d89cb0b514bfeaffa5b4129a
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Wed Oct 12 16:38:15 2022 +0100

        save the unexpected non-paired transforms

    commit f7fcf1bc012e8f26399cb5a62a50c26426501480
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Wed Oct 12 16:14:41 2022 +0100

        move some of QCurandState down into SCurandState, aiming to tie together SEvt maxima with the number of curandState loaded







::

    (gdb) bt
    #0  0x00007fffeba47387 in raise () from /lib64/libc.so.6
    #1  0x00007fffeba48a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffeba401a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffeba40252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffeda7c89a in QRng::Load (rngmax=@0x2eec4c0: 0, path=0x0) at /data/blyth/junotop/opticks/qudarap/QRng.cc:75
    #5  0x00007fffeda7c3e5 in QRng::QRng (this=0x2eec4b0, path_=0x0, skipahead_event_offset=1) at /data/blyth/junotop/opticks/qudarap/QRng.cc:21
    #6  0x00007fffeda355a4 in QSim::UploadComponents (ssim=0x1fdf1e0) at /data/blyth/junotop/opticks/qudarap/QSim.cc:114
    #7  0x00007fffefcae3d5 in CSGOptiX::InitSim (ssim=0x1fdf1e0) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:183
    #8  0x00007fffefcae6e1 in CSGOptiX::Create (fd=0xd76c80) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:202
    #9  0x00007ffff7b8eb40 in G4CXOpticks::setGeometry (this=0x7fffffff5900, fd_=0xd76c80) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:274
    #10 0x00007ffff7b8dd08 in G4CXOpticks::setGeometry (this=0x7fffffff5900) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:164
    #11 0x0000000000408d46 in main (argc=3, argv=0x7fffffff5a68) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimtraceTest.cc:26
    (gdb) 


::

    258 void G4CXOpticks::setGeometry(CSGFoundry* fd_)
    259 {
    263     fd = fd_ ; 
    267     SEvt* sev = new SEvt ; 
    270     sev->setReldir("ALL"); 
    271     sev->setGeo((SGeo*)fd);
    274     cx = CSGOptiX::Create(fd);   // uploads geometry to GPU 
    276     qs = cx->sim ; 
    280     if( setGeometry_saveGeometry )
    281     {   
    283         saveGeometry(setGeometry_saveGeometry); 
    285     }
    288 }

    0196 CSGOptiX* CSGOptiX::Create(CSGFoundry* fd )
     197 {
     200     QU::alloc = new salloc ;   // HMM: maybe this belongs better in QSim ? 
     202     InitSim(fd->sim);
     203     InitGeo(fd);
     205     CSGOptiX* cx = new CSGOptiX(fd) ;
     207     bool render_mode = SEventConfig::IsRGModeRender() ;
     208     if(render_mode == false)
     209     {
     210         QSim* qs = QSim::Get() ;
     211         qs->setLauncher(cx);
     212         QEvent* event = qs->event ;
     213         event->setMeta( fd->meta.c_str() );
     214     }
     217     return cx ;
     218 }

     175 void CSGOptiX::InitSim( const SSim* ssim  )
     176 {
     178     if(SEventConfig::IsRGModeRender()) return ;
     180     LOG_IF(fatal, ssim == nullptr ) << "simulate/simtrace modes require SSim/QSim setup" ;
     181     assert(ssim);
     183     QSim::UploadComponents(ssim);
     185     QSim* qs = QSim::Create() ;
     187 }


     101 void QSim::UploadComponents( const SSim* ssim  )
     102 {
     107     QBase* base = new QBase ;
     114     QRng* rng = new QRng ;  // loads and uploads curandState 
     119     const NP* optical = ssim->get(SSim::OPTICAL);
     120     const NP* bnd = ssim->get(SSim::BND);

