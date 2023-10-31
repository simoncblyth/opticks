Raindrop_revival_fix_CSGTarget_getFrame_nan_from_not_clearing_identity_info
=============================================================================

Testing pure opticks (no junosw) install on workstation "simon" account::

    /home/simon/opticks/g4cx/tests/G4CXTest_raindrop.sh 

Issue was caused by a bug in CSGTarget::getFrame with the frame m2w w2m transforms not having 
identity info cleared in one branch. 


::

    ]]stree::postcreate
    2023-10-31 19:51:40.407 INFO  [399696] [G4CXOpticks::setGeometry@280] Completed U4Tree::Create 
    Detaching after fork from child process 399914.
    Detaching after fork from child process 399915.
    [New Thread 0x7fffcf611700 (LWP 399916)]
    [New Thread 0x7fffcec8d700 (LWP 399917)]
    2023-10-31 19:51:41.304 ERROR [399696] [QSim::UploadComponents@151]  icdf null, snam::ICDF icdf.npy

    Program received signal SIGFPE, Arithmetic exception.
    0x00007fffeeb12a5b in qat4::is_identity (this=0x12bc4d0, eps=9.99999975e-06) at /data/simon/local/opticks/include/SysRap/sqat4.h:183
    183	            std::abs(q3.f.x)<eps     && std::abs(q3.f.y)<eps     && std::abs(q3.f.z) < eps     && std::abs(q3.f.w-1.f) < eps ;
    Missing separate debuginfos, use: debuginfo-install cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libgcc-4.8.5-44.el7.x86_64 libidn-1.28-4.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libstdc++-4.8.5-44.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-25.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffeeb12a5b in qat4::is_identity (this=0x12bc4d0, eps=9.99999975e-06) at /data/simon/local/opticks/include/SysRap/sqat4.h:183
    #1  0x00007fffeeb25f51 in SGLM::updateModelMatrix (this=0x12bc490) at /data/simon/local/opticks/include/SysRap/SGLM.h:474
    #2  0x00007fffeeb25c10 in SGLM::update (this=0x12bc490) at /data/simon/local/opticks/include/SysRap/SGLM.h:435
    #3  0x00007fffeeb25b55 in SGLM::set_frame (this=0x12bc490, fr_=...) at /data/simon/local/opticks/include/SysRap/SGLM.h:423
    #4  0x00007fffeeb0d4d3 in CSGOptiX::setFrame (this=0x12bc3b0, fr_=...) at /home/simon/opticks/CSGOptiX/CSGOptiX.cc:687
    #5  0x00007ffff7af8597 in G4CXOpticks::setupFrame (this=0x9148f0) at /home/simon/opticks/g4cx/G4CXOpticks.cc:448
    #6  0x00007ffff7af7bcd in G4CXOpticks::setGeometry (this=0x9148f0, fd_=0x9808f0) at /home/simon/opticks/g4cx/G4CXOpticks.cc:329
    #7  0x00007ffff7af7b58 in G4CXOpticks::setGeometry (this=0x9148f0, world=0x906e80) at /home/simon/opticks/g4cx/G4CXOpticks.cc:283
    #8  0x00007ffff7af6485 in G4CXOpticks::SetGeometry (world=0x906e80) at /home/simon/opticks/g4cx/G4CXOpticks.cc:74
    #9  0x00000000004148e8 in G4CXApp::Construct (this=0x8c0660) at /home/simon/opticks/g4cx/tests/G4CXApp.h:182
    #10 0x00007ffff43c7cc2 in G4RunManager::InitializeGeometry (this=0x865710)
        at /data/simon/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/run/src/G4RunManager.cc:588
    #11 0x00007ffff43c7b96 in G4RunManager::Initialize (this=0x865710)
        at /data/simon/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/run/src/G4RunManager.cc:566
    #12 0x00000000004143d3 in G4CXApp::G4CXApp (this=0x8c0660, runMgr=0x865710) at /home/simon/opticks/g4cx/tests/G4CXApp.h:152
    #13 0x000000000041536f in G4CXApp::Create () at /home/simon/opticks/g4cx/tests/G4CXApp.h:306
    #14 0x000000000041542f in G4CXApp::Main () at /home/simon/opticks/g4cx/tests/G4CXApp.h:317
    #15 0x00000000004155ca in main (argc=1, argv=0x7fffffffcca8) at /home/simon/opticks/g4cx/tests/G4CXTest.cc:16
    (gdb) 


    (gdb) f 9
    #9  0x00000000004148e8 in G4CXApp::Construct (this=0x8c0660) at /home/simon/opticks/g4cx/tests/G4CXApp.h:182
    182	        G4CXOpticks::SetGeometry(pv_) ; 
    (gdb) f 8
    #8  0x00007ffff7af6485 in G4CXOpticks::SetGeometry (world=0x906e80) at /home/simon/opticks/g4cx/G4CXOpticks.cc:74
    74	    g4cx->setGeometry(world); 
    (gdb) f 7
    #7  0x00007ffff7af7b58 in G4CXOpticks::setGeometry (this=0x9148f0, world=0x906e80) at /home/simon/opticks/g4cx/G4CXOpticks.cc:283
    283	    setGeometry(fd_); 
    (gdb) f 6
    #6  0x00007ffff7af7bcd in G4CXOpticks::setGeometry (this=0x9148f0, fd_=0x9808f0) at /home/simon/opticks/g4cx/G4CXOpticks.cc:329
    329	    setupFrame();    // EXPT: MOVED HERE TO INITIALIZATION
    (gdb) f 5
    #5  0x00007ffff7af8597 in G4CXOpticks::setupFrame (this=0x9148f0) at /home/simon/opticks/g4cx/G4CXOpticks.cc:448
    448	    if(cx) cx->setFrame(fr);  
    (gdb) list
    443	    sframe fr = fd->getFrameE() ; 
    444	    LOG(LEVEL) << fr ; 
    445	
    446	    SEvt::SetFrame(fr) ; 
    447	
    448	    if(cx) cx->setFrame(fr);  
    449	}
    450	
    451	
    452	
    (gdb) 


    (gdb) f 4
    #4  0x00007fffeeb0d4d3 in CSGOptiX::setFrame (this=0x12bc3b0, fr_=...) at /home/simon/opticks/CSGOptiX/CSGOptiX.cc:687
    687	    sglm->set_frame(fr_); 
    (gdb) list
    682	
    683	**/
    684	
    685	void CSGOptiX::setFrame(const sframe& fr_ )
    686	{
    687	    sglm->set_frame(fr_); 
    688	
    689	    LOG(LEVEL) << "sglm.desc:" << std::endl << sglm->desc() ; 
    690	
    691	    const float4& ce = sglm->fr.ce ; 
    (gdb) 


    (gdb) f 1
    #1  0x00007fffeeb25f51 in SGLM::updateModelMatrix (this=0x12bc490) at /data/simon/local/opticks/include/SysRap/SGLM.h:474
    474	    bool m2w_not_identity = fr.m2w.is_identity(sframe::EPSILON) == false ;
    (gdb) f 0 
    #0  0x00007fffeeb12a5b in qat4::is_identity (this=0x12bc4d0, eps=9.99999975e-06) at /data/simon/local/opticks/include/SysRap/sqat4.h:183
    183	            std::abs(q3.f.x)<eps     && std::abs(q3.f.y)<eps     && std::abs(q3.f.z) < eps     && std::abs(q3.f.w-1.f) < eps ;
    (gdb) p q3.f.x
    $1 = 0
    (gdb) p q3.f.y
    $2 = 0
    (gdb) p q3.f.z
    $3 = 0
    (gdb) p q3.f.w
    $4 = -nan(0x7fffff)
    (gdb) 

Unexpected nan in fr.m2w ?::

     470 inline void SGLM::updateModelMatrix()
     471 {
     472     updateModelMatrix_branch = 0 ;
     473 
     474     bool m2w_not_identity = fr.m2w.is_identity(sframe::EPSILON) == false ;
     475     bool w2m_not_identity = fr.w2m.is_identity(sframe::EPSILON) == false ;
     476 


     
     674 /**
     675 CSGOptiX::setFrame into the SGLM.h instance
     676 ----------------------------------------------
     677 
     678 Note that SEvt already holds an sframe used for input photon transformation, 
     679 the sframe here is used for raytrace rendering.  Could perhaps rehome sglm 
     680 into SEvt and use a single sframe for both input photon transformation 
     681 and rendering ?
     682 
     683 **/
     684 
     685 void CSGOptiX::setFrame(const sframe& fr_ )
     686 {
     687     sglm->set_frame(fr_);
     688 
     689     LOG(LEVEL) << "sglm.desc:" << std::endl << sglm->desc() ;
     690 
     691     const float4& ce = sglm->fr.ce ;
     692     const qat4& m2w = sglm->fr.m2w ;
     693     const qat4& w2m = sglm->fr.w2m ;
     694 


::

    418 /**
    419 G4CXOpticks::setupFrame
    420 -------------------------
    421 
    422 The frame used depends on envvars INST, MOI, OPTICKS_INPUT_PHOTON_FRAME 
    423 it comprises : fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 
    424 
    425 This setupFrame was formerly called from G4CXOpticks::simulate and G4CXOpticks::simtrace
    426 it is now moved to G4CXOpticks::setGeometry to facilitate transformation 
    427 of input photons. 
    428 
    429 Q: why is the frame needed ?
    430 A: cx rendering viewpoint, input photon frame and the simtrace genstep grid 
    431    are all based on the frame center, extent and transforms 
    432 
    433 Q: Given the sframe and SEvt are from sysrap it feels too high level to do this here, 
    434    should be at CSG or sysrap level perhaps ? 
    435    And then CSGOptix could grab the SEvt frame at its initialization. 
    436 
    437 **/
    438 
    439 void G4CXOpticks::setupFrame()
    440 {
    441     // TODO: see CSGFoundry::AfterLoadOrCreate for auto frame hookup
    442 
    443     sframe fr = fd->getFrameE() ;
    444     LOG(LEVEL) << fr ;
    445 
    446     SEvt::SetFrame(fr) ;
    447 
    448     if(cx) cx->setFrame(fr);
    449 }


::

    (gdb) f 0
    #0  0x00007fffeeb12a5b in qat4::is_identity (this=0x12bc4d0, eps=9.99999975e-06) at /data/simon/local/opticks/include/SysRap/sqat4.h:183
    183	            std::abs(q3.f.x)<eps     && std::abs(q3.f.y)<eps     && std::abs(q3.f.z) < eps     && std::abs(q3.f.w-1.f) < eps ;
    (gdb) p q3.f.x
    $1 = 0
    (gdb) p q3.f.y
    $2 = 0
    (gdb) p q3.f.z
    $3 = 0
    (gdb) p q3.f.w
    $4 = -nan(0x7fffff)
    (gdb) p q3.u.w
    $5 = 4294967295
    (gdb) p q3.i.w
    $6 = -1
    (gdb) 




