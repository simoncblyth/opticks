legacy-tidy-up
===================


ChromaMaterialMap.json can this be removed ?
-----------------------------------------------

::

    2019-05-27 21:59:24.083 INFO  [41693] [Camera::setType@277]  type 0
    2019-05-27 21:59:24.083 WARN  [41693] [BTree::loadTree@60] BTree.loadTree: can't find file /home/blyth/local/opticks/opticksdata/export/CerenkovMinimal/ChromaMaterialMap.json
    OKTest: /home/blyth/opticks/boostrap/BTree.cc:63: static int BTree::loadTree(boost::property_tree::ptree&, const char*): Assertion `0' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffeafa9207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffeafa9207 in raise () from /lib64/libc.so.6
    #1  0x00007fffeafaa8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffeafa2026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffeafa20d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff24e73d2 in BTree::loadTree (t=..., path=0x64e2f8 "/home/blyth/local/opticks/opticksdata/export/CerenkovMinimal/ChromaMaterialMap.json") at /home/blyth/opticks/boostrap/BTree.cc:63
    #5  0x00007ffff24a0132 in BMap<std::string, unsigned int>::load (this=0x7fffffffd1f0, path=0x648ed0 "/home/blyth/local/opticks/opticksdata/export/CerenkovMinimal/ChromaMaterialMap.json", depth=0) at /home/blyth/opticks/boostrap/BMap.cc:150
    #6  0x00007ffff249f906 in BMap<std::string, unsigned int>::load (mp=0x7fffffffd220, path=0x648ed0 "/home/blyth/local/opticks/opticksdata/export/CerenkovMinimal/ChromaMaterialMap.json", depth=0) at /home/blyth/opticks/boostrap/BMap.cc:58
    #7  0x00007ffff64f0813 in OpticksHub::configureLookupA (this=0x63d720) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:464
    #8  0x00007ffff64ef48f in OpticksHub::init (this=0x63d720) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:218
    #9  0x00007ffff64ef2c0 in OpticksHub::OpticksHub (this=0x63d720, ok=0x626160) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:195
    #10 0x00007ffff7bd587f in OKMgr::OKMgr (this=0x7fffffffd840, argc=4, argv=0x7fffffffd9b8, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:44
    #11 0x0000000000402eab in main (argc=4, argv=0x7fffffffd9b8) at /home/blyth/opticks/ok/tests/OKTest.cc:13
    (gdb) 


Embedded option will prevent lookin for this. Can embedded be equated with running from key ?::

     205 void OpticksHub::init()
     206 {
     207     pLOG(LEVEL,0) << "[" ;   // -1 : one notch more easily seen than LEVEL
     208 
     209     //m_composition->setCtrl(this); 
     210 
     211     add(m_fcfg);
     212 
     213     configure();
     214     // configureGeometryPrep();
     215     configureServer();
     216     configureCompositionSize();
     217 
     218     if(!m_ok->isEmbedded()) configureLookupA();
     219 
     220     m_aim = new OpticksAim(this) ;
     221 
     222     if( m_ggeo == NULL )
     223     {
     224         loadGeometry() ;
     225     }
     226     else



