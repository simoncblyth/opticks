gxr_shakedown
================


::

    gx
    ./gxr.sh dbg

    ...

    2022-07-09 20:59:36.517 INFO  [362662] [CSGOptiX::setFrame@504] ]
    2022-07-09 20:59:36.517 INFO  [362662] [CSGOptiX::init@281] ]

    Program received signal SIGSEGV, Segmentation fault.
    0x00007fffedb35c0a in QSim::setLauncher (this=0x0, cx_=0x14cae00) at /data/blyth/junotop/opticks/qudarap/QSim.cc:225
    225	    cx = cx_ ; 
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffedb35c0a in QSim::setLauncher (this=0x0, cx_=0x14cae00) at /data/blyth/junotop/opticks/qudarap/QSim.cc:225
    #1  0x00007fffefd0c338 in CSGOptiX::Create (fd=0xdf4b00) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:199
    #2  0x00007ffff7bb5de3 in G4CXOpticks::setGeometry (this=0x7fffffff6600, fd_=0xdf4b00) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:112
    #3  0x00007ffff7bb5da1 in G4CXOpticks::setGeometry (this=0x7fffffff6600, gg_=0x7c85b0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:107
    #4  0x00007ffff7bb5d5b in G4CXOpticks::setGeometry (this=0x7fffffff6600, world=0x7ba750) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:101
    #5  0x00007ffff7bb5bf0 in G4CXOpticks::setGeometry (this=0x7fffffff6600) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:82
    #6  0x0000000000404f40 in main (argc=1, argv=0x7fffffff6748) at /data/blyth/junotop/opticks/g4cx/tests/G4CXRenderTest.cc:27
    (gdb) 


    (gdb) f 1
    #1  0x00007fffefd0c338 in CSGOptiX::Create (fd=0xdf4b00) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:199
    199	    qs->setLauncher(cx); 
    (gdb) 


    168 void CSGOptiX::InitSim( const SSim* ssim  )
    169 {
    170     if(SEventConfig::IsRGModeRender()) return ;
    171 
    172     if(ssim == nullptr) LOG(fatal) << "simulate/simtrace modes require SSim/QSim setup" ;
    173     assert(ssim);
    174 
    175     QSim::UploadComponents(ssim);
    176 
    177     QSim* qs = QSim::Create() ;
    178     LOG(LEVEL) << qs->desc() ;
    179 }



    188 CSGOptiX* CSGOptiX::Create(CSGFoundry* fd )
    189 {
    190     LOG(LEVEL) << "fd.descBase " << ( fd ? fd->descBase() : "-" ) ;
    191 
    192     InitSim(fd->sim);
    193     InitGeo(fd);
    194 
    195     CSGOptiX* cx = new CSGOptiX(fd) ;
    196 
    197     QSim* qs = QSim::Get() ;
    198 
    199     qs->setLauncher(cx);
    200 
    201     QEvent* event = qs->event ;
    202     event->setMeta( fd->meta.c_str() );
    203 
    204     // DONE: setup QEvent as SCompProvider of NP arrays allowing SEvt to drive QEvent download
    205     return cx ;
    206 }


