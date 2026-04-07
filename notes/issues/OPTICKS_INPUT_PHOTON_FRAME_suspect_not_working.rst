FIXED : OPTICKS_INPUT_PHOTON_FRAME_suspect_not_working
============================================================


Issue : 2026 April 7 : oj3 running OPTICKS_INPUT_PHOTON_FRAME not working
---------------------------------------------------------------------------


::

    (ok) A[blyth@localhost sysrap]$ opticks-f InputPhotonFrame
    ./CSG/CSGFoundry.cc:        const char* ipf_ = SEventConfig::InputPhotonFrame();  // OPTICKS_INPUT_PHOTON_FRAME
    ./CSG/CSGFoundry.cc:        fr.set_ekv(SEventConfig::kInputPhotonFrame, ipf );
     ## JUST FOR VIZ WHEN LOADING PERSISTED GEOM WITHOUT MOI

    ./bin/OPTICKS_INPUT_PHOTON.sh:   moi_or_iidx string eg "Hama:0:1000" OR "35000", default of SEventConfig::InputPhotonFrame

    ./sysrap/SCF.h:    const qat4* getInputPhotonFrame(const char* ipf_) const ;
    ./sysrap/SCF.h:    const qat4* getInputPhotonFrame() const ;
    ./sysrap/SCF.h:const qat4* SCF::getInputPhotonFrame(const char* ipf_) const
    ./sysrap/SCF.h:const qat4* SCF::getInputPhotonFrame() const
    ./sysrap/SCF.h:    const char* ipf_ = SEventConfig::InputPhotonFrame();
    ./sysrap/SCF.h:    return getInputPhotonFrame(ipf_);



    ./sysrap/SEventConfig.cc:const char* SEventConfig::_InputPhotonFrameDefault = nullptr ;
    ./sysrap/SEventConfig.cc:const char* SEventConfig::_InputPhotonFrame = ssys::getenvvar(kInputPhotonFrame, _InputPhotonFrameDefault );
    ./sysrap/SEventConfig.cc:SEventConfig::InputPhotonFrame control via OPTICKS_INPUT_PHOTON_FRAME envvar
    ./sysrap/SEventConfig.cc:const char* SEventConfig::InputPhotonFrame(){       return _InputPhotonFrame ; }
    ./sysrap/SEventConfig.cc:void SEventConfig::SetInputPhotonFrame(const char* ip){   _InputPhotonFrame = ip ? strdup(ip) : nullptr ; LIMIT_Check() ; }
    ./sysrap/SEventConfig.cc:       << std::setw(25) << kInputPhotonFrame
    ./sysrap/SEventConfig.cc:       << std::setw(20) << " InputPhotonFrame " << " : " << ( InputPhotonFrame() ? InputPhotonFrame() : "-" )
    ./sysrap/SEventConfig.cc:    const char* ipf = InputPhotonFrame() ;
    ./sysrap/SEventConfig.cc:    if(ipf) meta->set_meta<std::string>("InputPhotonFrame", ipf );
    ./sysrap/SEventConfig.hh:    static constexpr const char* kInputPhotonFrame = "OPTICKS_INPUT_PHOTON_FRAME" ;
    ./sysrap/SEventConfig.hh:    static const char* InputPhotonFrame();
    ./sysrap/SEventConfig.hh:    static void SetInputPhotonFrame(const char* input_photon_frame);
    ./sysrap/SEventConfig.hh:    static const char* _InputPhotonFrameDefault ;
    ./sysrap/SEventConfig.hh:    static const char* _InputPhotonFrame ;

    ./sysrap/SEvt.cc:    const char* ipf = SEventConfig::InputPhotonFrame() ;
    ./sysrap/SEvt.cc:    ss << std::setw(c1) << " SEventConfig::InputPhotonFrame " << div << ( ipf ? ipf : "-" ) << std::endl ;

    ./sysrap/sevt.py:        ipfl = getattr(meta, "InputPhotonFrame", [])

    ./sysrap/tests/SEvtTest.cc:    const char* ipf = SEventConfig::InputPhotonFrame();
    ./sysrap/tests/SEvtTest.cc:    const qat4* q = SEvt::CF->getInputPhotonFrame();
    (ok) A[blyth@localhost opticks]$




HMM : THIS IS FIRST USE OF INPUT PHOTONS SINCE THE FRAME OVERHAUL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     826 #ifdef WITH_OLD_FRAME
     827 void SEvt::setFrame(const sframe& fr )
     828 {
     829     const char* name = fr.get_name() ;
     830     LOG_IF(info, FRAME)
     831         << " [" << SEvt__FRAME << "]"
     832         << " fr.get_name " << ( name ? name : "-" ) << "\n"
     833         << " fr.desc\n"
     834         << fr.desc()
     835         ;
     836     frame = fr ;
     837     transformInputPhoton();
     838 }
     839 #else
     840 void SEvt::setFr(const sfr& _fr )
     841 {
     842     fr = _fr ;
     843     transformInputPhoton();
     844 }
     845 #endif
     846


     oj ; BP=SEvt::setFr ojd


::

    In [14]: a.f.sfr
    Out[14]:
    sframe       :
    path         : ALL0_oj3ip_nnvt_1000/A000/sfr.npy
    meta         : creator:sfr::serialize
    name:ALL
    ce           : array([  0.,   0.,   0., 100.])
    bbmn         : array([0., 0., 0.])
    bbmx         : array([0., 0., 0.])
    m2w          :
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    w2m          :
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    id           :
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    In [15]: b.f.sfr
    Out[15]:
    sframe       :
    path         : ALL0_oj3ip_nnvt_1000/B000/sfr.npy
    meta         : creator:sfr::serialize
    name:ALL
    ce           : array([  0.,   0.,   0., 100.])
    bbmn         : array([0., 0., 0.])
    bbmx         : array([0., 0., 0.])
    m2w          :
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    w2m          :
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    id           :
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    In [16]:



oj3 running BP=SEvt::setFr not hit
------------------------------------

::

     465 void CSGOptiX::init()
     466 {
     ...
     490     initSimulate();
     491
     492 #ifdef WITH_OLD_FRAME
     493     initFrame();
     494 #endif
     495
     496     initRender();



     682 #ifdef WITH_OLD_FRAME
     683
     684 /**
     685 CSGOptiX::initFrame (formerly G4CXOpticks::setupFrame)
     686 ---------------------------------------------------------
     687
     688 The frame used depends on envvars INST, MOI, OPTICKS_INPUT_PHOTON_FRAME
     689 it comprises : fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame
     690
     691 Q: why is the frame needed ?
     692 A: cx rendering viewpoint, input photon frame and the simtrace genstep grid
     693    are all based on the frame center, extent and transforms
     694
     695 Q: Given the sframe and SEvt are from sysrap it feels too high level to do this here,
     696    should be at CSG or sysrap level perhaps ?
     697    And then CSGOptix could grab the SEvt frame at its initialization.
     698
     699 TODO: see CSGFoundry::AfterLoadOrCreate for maybe auto frame hookup
     700
     701 **/
     702
     703 void CSGOptiX::initFrame()
     704 {
     705     assert(0);
     706
     707     sframe _fr = foundry->getFrameE() ;   // TODO: migrate to lighweight sfr from stree level
     708     LOG(LEVEL) << _fr ;
     709
     710     SEvt::SetFrame(_fr) ;
     711
     712     sfr _lfr = _fr.spawn_lite();
     713     setFrame(_lfr);
     714 }
     715
     716 #endif
     717


::

    1534 #ifdef WITH_OLD_FRAME
    1535 void SEvt::SetFrame(const sframe& fr )
    1536 {
    1537     assert(0 && "DONT USE THIS - USE SEvt::SetFr");
    1538     if(Exists(0)) Get(0)->setFrame(fr);
    1539     if(Exists(1)) Get(1)->setFrame(fr);
    1540 }
    1541 #else
    1542 void SEvt::SetFr(const sfr& fr )
    1543 {
    1544     if(Exists(0)) Get(0)->setFr(fr);
    1545     if(Exists(1)) Get(1)->setFr(fr);
    1546 }
    1547 #endif





where to do the frame hookup for simulation ?
-----------------------------------------------

Renamed 


::

    (ok) A[blyth@localhost CSGOptiX]$ opticks-f get_frame_moi
    ./CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.cc:    sfr fr = tree->get_frame_moi();
    ./CSG/tests/CSGFoundry_getFrame_Test.cc:    fr = tree->get_frame_moi();
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.cc:    sfr fr = tree->get_frame_moi();
    ./sysrap/SGLM.h:    moi_fr = tree->get_frame_moi();
    ./sysrap/SGLM.h:    sfr f = tree->get_frame_moi();
    ./sysrap/SSim.cc:    sfr fr = tree->get_frame_moi() ;
    ./sysrap/stree.h:    sfr  get_frame_moi() const ;
    ./sysrap/stree.h:stree::get_frame_moi
    ./sysrap/stree.h:inline sfr stree::get_frame_moi() const
    (ok) A[blyth@localhost opticks]$




::

     540 /**
     541 SSim::afterLoadOrCreate
     542 ------------------------
     543
     544 Trying to progress with FRAME_TRANSITION by replacing the old
     545 CSGFoundry::AfterLoadOrCreate with SSim::afterLoadOrCreate
     546
     547 **/
     548
     549 void SSim::afterLoadOrCreate()
     550 {
     551     SEvt::CreateOrReuse() ;   // creates 1/2 SEvt depending on OPTICKS_INTEGRATION_MODE
     552     assert(tree);
     553
     554
     555     sfr fr = tree->get_frame_moi() ;
     556     SEvt::SetFr(fr);
     557 }
     558














Issue : in cxs_min.sh ana (cxs_min.py) the a.f.sframe has identity transforms unexpectedly for NNVT:0:1000
---------------------------------------------------------------------------------------------------------------

This was caused by a recent change to the starting default in SName.h done
for ELV geometry exclusions. For geometry want to require starting:false but
for INPUT_PHOTON_FRAME targetting need to be more lax with starting:true


INPUT_PHOTON_FRAME
--------------------


::

    P[blyth@localhost sysrap]$ grep INPUT_PHOTON_FRAME *.*
    SEventConfig.cc:SEventConfig::InputPhotonFrame control via OPTICKS_INPUT_PHOTON_FRAME envvar
    SEventConfig.hh:    static constexpr const char* kInputPhotonFrame = "OPTICKS_INPUT_PHOTON_FRAME" ;
    SEvt.cc:| OPTICKS_INPUT_PHOTON_FRAME   |                            |
    P[blyth@localhost sysrap]$ vi SEventConfig.cc
    P[blyth@localhost sysrap]$ opticks-
    P[blyth@localhost sysrap]$ opticks-f InputPhotonFrame

    ./CSG/CSGFoundry.cc:        const char* ipf_ = SEventConfig::InputPhotonFrame();  // OPTICKS_INPUT_PHOTON_FRAME
    ./CSG/CSGFoundry.cc:        fr.set_ekv(SEventConfig::kInputPhotonFrame, ipf );

    ./bin/OPTICKS_INPUT_PHOTON.sh:   moi_or_iidx string eg "Hama:0:1000" OR "35000", default of SEventConfig::InputPhotonFrame

    ./sysrap/SCF.h:    const qat4* getInputPhotonFrame(const char* ipf_) const ;
    ./sysrap/SCF.h:    const qat4* getInputPhotonFrame() const ;
    ./sysrap/SCF.h:const qat4* SCF::getInputPhotonFrame(const char* ipf_) const
    ./sysrap/SCF.h:const qat4* SCF::getInputPhotonFrame() const
    ./sysrap/SCF.h:    const char* ipf_ = SEventConfig::InputPhotonFrame();
    ./sysrap/SCF.h:    return getInputPhotonFrame(ipf_);

    ./sysrap/SEventConfig.cc:const char* SEventConfig::_InputPhotonFrameDefault = nullptr ;
    ./sysrap/SEventConfig.cc:const char* SEventConfig::_InputPhotonFrame = ssys::getenvvar(kInputPhotonFrame, _InputPhotonFrameDefault );
    ./sysrap/SEventConfig.cc:SEventConfig::InputPhotonFrame control via OPTICKS_INPUT_PHOTON_FRAME envvar
    ./sysrap/SEventConfig.cc:const char* SEventConfig::InputPhotonFrame(){   return _InputPhotonFrame ; }
    ./sysrap/SEventConfig.cc:void SEventConfig::SetInputPhotonFrame(const char* ip){   _InputPhotonFrame = ip ? strdup(ip) : nullptr ; Check() ; }
    ./sysrap/SEventConfig.cc:    const char* ipf = InputPhotonFrame() ;
    ./sysrap/SEventConfig.cc:    if(ipf) meta->set_meta<std::string>("InputPhotonFrame", ipf );
    ./sysrap/SEventConfig.hh:    static constexpr const char* kInputPhotonFrame = "OPTICKS_INPUT_PHOTON_FRAME" ;
    ./sysrap/SEventConfig.hh:    static const char* InputPhotonFrame();
    ./sysrap/SEventConfig.hh:    static void SetInputPhotonFrame(const char* input_photon_frame);
    ./sysrap/SEventConfig.hh:    static const char* _InputPhotonFrameDefault ;
    ./sysrap/SEventConfig.hh:    static const char* _InputPhotonFrame ;

    ./sysrap/SEvt.cc:    const char* ipf = SEventConfig::InputPhotonFrame() ;
    ./sysrap/SEvt.cc:    ss << std::setw(c1) << " SEventConfig::InputPhotonFrame " << div << ( ipf ? ipf : "-" ) << std::endl ;
    ./sysrap/sevt.py:        ipfl = getattr(meta, "InputPhotonFrame", [])

    ./sysrap/tests/SEvtTest.cc:    const char* ipf = SEventConfig::InputPhotonFrame();
    ./sysrap/tests/SEvtTest.cc:    const qat4* q = SEvt::CF->getInputPhotonFrame();
    P[blyth@localhost opticks]$



::

    4510 std::string SEvt::descInputPhoton() const
    4511 {
    4512     const char* ip = SEventConfig::InputPhoton() ;
    4513     const char* ipf = SEventConfig::InputPhotonFrame() ;
    4514     int c1 = 35 ;
    4515
    4516     const char* div = " : " ;
    4517     std::stringstream ss ;
    4518     ss << "SEvt::descInputPhoton" << std::endl ;
    4519     ss << std::setw(c1) << " SEventConfig::IntegrationMode "  << div << SEventConfig::IntegrationMode() << std::endl ;
    4520     ss << std::setw(c1) << " SEventConfig::InputPhoton "      << div << ( ip  ? ip  : "-" ) << std::endl ;
    4521     ss << std::setw(c1) << " SEventConfig::InputPhotonFrame " << div << ( ipf ? ipf : "-" ) << std::endl ;
    4522     ss << std::setw(c1) << " hasInputPhoton " << div << ( hasInputPhoton() ? "YES" : "NO " ) << std::endl ;
    4523     ss << std::setw(c1) << " input_photon "   << div << ( input_photon ? input_photon->sstr() : "-" )     << std::endl ;
    4524     ss << std::setw(c1) << " input_photon.lpath " << div << ( input_photon ? input_photon->get_lpath() : "--" ) << std::endl ;
    4525     ss << std::setw(c1) << " hasInputPhotonTransformed " << div << ( hasInputPhotonTransformed() ? "YES" : "NO " ) ;
    4526     std::string s = ss.str();
    4527     return s ;
    4528 }



Where/when does the transformation of input photons happen ?
-----------------------------------------------------------------

::

    P[blyth@localhost CSGOptiX]$ BP=SEvt::transformInputPhoton ./cxs_min.sh dbg


    Breakpoint 1, 0x00007ffff6ee4a40 in SEvt::transformInputPhoton()@plt () from /data/blyth/opticks_Debug/lib/../lib64/libSysRap.so
    (gdb) bt
    #0  0x00007ffff6ee4a40 in SEvt::transformInputPhoton()@plt () from /data/blyth/opticks_Debug/lib/../lib64/libSysRap.so
    #1  0x00007ffff6f948c1 in SEvt::setFrame (this=0xef030f0, fr=...) at /home/blyth/opticks/sysrap/SEvt.cc:659
    #2  0x00007ffff6f972aa in SEvt::SetFrame (fr=...) at /home/blyth/opticks/sysrap/SEvt.cc:1171
    #3  0x00007ffff79cbd13 in CSGFoundry::AfterLoadOrCreate () at /home/blyth/opticks/CSG/CSGFoundry.cc:3670
    #4  0x00007ffff79c973e in CSGFoundry::Load () at /home/blyth/opticks/CSG/CSGFoundry.cc:3064
    #5  0x00007ffff7bfe3c7 in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:166
    #6  0x0000000000404a75 in main (argc=1, argv=0x7fffffff1408) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb)




::

     634 /**
     635 SEvt::setFrame
     636 ------------------
     637
     638 As it is necessary to have the geometry to provide the frame this
     639 is now split from eg initInputPhotons.
     640
     641 **simtrace running**
     642     MakeCenterExtentGensteps based on the given frame.
     643
     644 **simulate inputphoton running**
     645     MakeInputPhotonGenstep and m2w (model-2-world)
     646     transforms the photons using the frame transform
     647
     648 Formerly(?) for simtrace and input photon running with or without a transform
     649 it was necessary to call this for every event due to the former call to addInputGenstep,
     650 but now that the genstep setup is moved to SEvt::beginOfEvent it is only needed
     651 to call this for each frame, usually once only.
     652
     653 **/
     654
     655
     656 void SEvt::setFrame(const sframe& fr )
     657 {
     658     frame = fr ;
     659     transformInputPhoton();
     660 }


     678 void SEvt::transformInputPhoton()
     679 {
     680     bool proceed = SEventConfig::IsRGModeSimulate() && hasInputPhoton() ;
     681     LOG(LEVEL) << " proceed " << ( proceed ? "YES" : "NO " ) ;
     682     if(!proceed) return ;
     683
     684     bool normalize = true ;  // normalize mom and pol after doing the transform
     685
     686     NP* ipt = frame.transform_photon_m2w( input_photon, normalize );
     687
     688     if(transformInputPhoton_WIDE)  // see notes/issues/G4ParticleChange_CheckIt_warnings.rst
     689     {
     690         input_photon_transformed = ipt ;
     691     }
     692     else
     693     {
     694         input_photon_transformed = ipt->ebyte == 8 ? NP::MakeNarrow(ipt) : ipt ;
     695         // narrow here to prevent immediate A:B difference with Geant4 seeing double precision
     696         // and Opticks float precision
     697     }
     698 }
     699


Added _VERBOSE envvar::

    2024-10-15 15:15:57.282  282358227 : [./cxs_min.sh
    2024-10-15 15:16:09.377 INFO  [181941] [SEvt::transformInputPhoton@685]  SEvt__transformInputPhoton_VERBOSE  SEventConfig::IsRGModeSimulate 1 hasInputPhoton 1 proceed YES
    sframe::desc inst 0 frs NNVT:0:1000
     ekvid sframe_OPTICKS_INPUT_PHOTON_FRAME_NNVT_0_1000 ek OPTICKS_INPUT_PHOTON_FRAME ev NNVT:0:1000
     ce  ( 0.000, 0.000, 0.000,60000.000)  is_zero 0
     m2w ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000)
     w2m ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000)
     midx    0 mord    0 gord    0
     inst    0
     ix0     0 ix1     0 iy0     0 iy1     0 iz0     0 iz1     0 num_photon    0
     ins     0 gas     0 sensor_identifier        0 sensor_index      0
     propagate_epsilon    0.05000 is_hostside_simtrace NO


    2024-10-15 15:16:10.697 INFO  [181941] [SEvt::transformInputPhoton@685]  SEvt__transformInputPhoton_VERBOSE  SEventConfig::IsRGModeSimulate 1 hasInputPhoton 1 proceed YES
    sframe::desc inst 0 frs NNVT:0:1000
     ekvid sframe_OPTICKS_INPUT_PHOTON_FRAME_NNVT_0_1000 ek OPTICKS_INPUT_PHOTON_FRAME ev NNVT:0:1000
     ce  ( 0.000, 0.000, 0.000,60000.000)  is_zero 0
     m2w ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000)
     w2m ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000)
     midx    0 mord    0 gord    0
     inst    0
     ix0     0 ix1     0 iy0     0 iy1     0 iz0     0 iz1     0 num_photon    0
     ins     0 gas     0 sensor_identifier        0 sensor_index      0
     propagate_epsilon    0.05000 is_hostside_simtrace NO


So why identity ?::

    P[blyth@localhost CSGOptiX]$ BP=SEvt::setFrame ./cxs_min.sh
    ./cxs_min.sh : FOUND A_CFBaseFromGEOM /cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/J_2024aug27 containing CSGFoundry/prim.npy
    Breakpoint 1, 0x00007ffff6ee19b0 in SEvt::setFrame(sframe const&)@plt () from /data/blyth/opticks_Debug/lib/../lib64/libSysRap.so
    (gdb) bt
    #0  0x00007ffff6ee19b0 in SEvt::setFrame(sframe const&)@plt () from /data/blyth/opticks_Debug/lib/../lib64/libSysRap.so
    #1  0x00007ffff6f964a8 in SEvt::SetFrame (fr=...) at /home/blyth/opticks/sysrap/SEvt.cc:1184
    #2  0x00007ffff79cbd13 in CSGFoundry::AfterLoadOrCreate () at /home/blyth/opticks/CSG/CSGFoundry.cc:3670
    #3  0x00007ffff79c973e in CSGFoundry::Load () at /home/blyth/opticks/CSG/CSGFoundry.cc:3064
    #4  0x00007ffff7bfe3c7 in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:166
    #5  0x0000000000404a75 in main (argc=1, argv=0x7fffffff4678) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb)


::

    3649 /**
    3650 CSGFoundry::AfterLoadOrCreate
    3651 -------------------------------
    3652
    3653 Called from some high level methods eg: CSGFoundry::Load
    3654
    3655 The idea behind this is to auto connect SEvt with the frame
    3656 from the geometry.
    3657
    3658 **/
    3659
    3660 void CSGFoundry::AfterLoadOrCreate() // static
    3661 {
    3662     CSGFoundry* fd = CSGFoundry::Get();
    3663
    3664     SEvt::CreateOrReuse() ;   // creates 1/2 SEvt depending on OPTICKS_INTEGRATION_MODE
    3665
    3666     if(!fd) return ;
    3667
    3668     sframe fr = fd->getFrameE() ;
    3669     LOG(LEVEL) << fr ;
    3670     SEvt::SetFrame(fr); // now only needs to be done once to transform input photons
    3671
    3672 }


Add some more _VERBOSE::

    373 export SEvt__transformInputPhoton_VERBOSE=1
    374 export CSGFoundry__getFrameE_VERBOSE=1
    375 export CSGFoundry__getFrame_VERBOSE=1

midx for NNVT coming back -1::

    /cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2024-10-15 15:41:30.864  864904454 : [./cxs_min.sh
    2024-10-15 15:41:42.652 INFO  [232737] [CSGFoundry::getFrameE@3674]  ipf NNVT:0:1000
    2024-10-15 15:41:42.652 INFO  [232737] [CSGFoundry::getFrame@3533] [CSGFoundry__getFrame_VERBOSE] YES frs NNVT:0:1000 looks_like_moi YES
    2024-10-15 15:41:42.690 INFO  [232737] [CSGFoundry::getFrame@3547] [CSGFoundry__getFrame_VERBOSE] YES frs NNVT:0:1000 looks_like_moi YES midx -1 mord 0 gord 1000 rc 0
    2024-10-15 15:41:42.690 INFO  [232737] [SEvt::transformInputPhoton@685]  SEvt__transformInputPhoton_VERBOSE  SEventConfig::IsRGModeSimulate 1 hasInputPhoton 1 proceed YES
    sframe::desc inst 0 frs NNVT:0:1000




