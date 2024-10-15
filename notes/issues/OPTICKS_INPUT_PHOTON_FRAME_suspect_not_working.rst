FIXED : OPTICKS_INPUT_PHOTON_FRAME_suspect_not_working
============================================================


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




