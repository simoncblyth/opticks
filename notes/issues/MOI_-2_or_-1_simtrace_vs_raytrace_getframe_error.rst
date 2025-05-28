FIXED : MOI_-2_or_-1_simtrace_vs_raytrace_getframe_error
=============================================================


Sometime need -1 and sometimes -2 (convention deviation in global frame access)::

    #moi=sChimneyLS:0:-1     ## for simtrace yields identity transforms
    #moi=sChimneyLS:0:-2      ## for simtrace yields expected transforms 
    moi=sWaterTube:0:-2
    #moi=sWaterTube:0:-1


Inconsistency between simtrace and rendering regards the MOI to use. 
Rendering with "sWaterTube:0:-2" asserts from stree::get_frame.
This must be from multiple get_frame impl.::

    stree::get_frame_instanced FAIL missing_transform  lvid 139 lvid_ordinal 0 repeat_ordinal -2 w2m NO  m2w NO  ii -1
    stree::get_frame FAIL q_spec[sWaterTube:0:-2]
     THIS CAN BE CAUSED BY NOT USING REPEAT_ORDINAL -1 (LAST OF TRIPLET) FOR GLOBAL GEOMETRY 
    CSGOptiXRenderInteractiveTest: /data1/blyth/local/opticks_Debug/include/SysRap/stree.h:1864: sfr stree::get_frame(const char*) const: Assertion `get_rc == 0' failed.

    Program received signal SIGABRT, Aborted.
    0x00007ffff4c8b52c in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-125.el9_5.3.alma.1.x86_64 libX11-1.7.0-9.el9.x86_64 libXau-1.0.9-8.el9.x86_64 libXext-1.3.4-8.el9.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libglvnd-1.3.4-1.el9.x86_64 libglvnd-glx-1.3.4-1.el9.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64 libxcb-1.13.1-9.el9.x86_64 openssl-libs-3.2.2-6.el9_5.1.x86_64
    (gdb) bt
    #0  0x00007ffff4c8b52c in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff4c3e686 in raise () from /lib64/libc.so.6
    #2  0x00007ffff4c28833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff4c2875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff4c373c6 in __assert_fail () from /lib64/libc.so.6
    #5  0x0000000000472615 in stree::get_frame (this=0x5202b0, q_spec=0x7fffffffbf5c "sWaterTube:0:-2") at /data1/blyth/local/opticks_Debug/include/SysRap/stree.h:1864
    #6  0x000000000047d5ca in SGLM::setTreeScene (this=0xf234250, _tree=0x5202b0, _scene=0x522f00) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLM.h:829
    #7  0x0000000000486b89 in CSGOptiXRenderInteractiveTest::init (this=0x7fffffffb310) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:124
    #8  0x0000000000486a77 in CSGOptiXRenderInteractiveTest::CSGOptiXRenderInteractiveTest (this=0x7fffffffb310) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:113
    #9  0x000000000044479c in main (argc=1, argv=0x7fffffffb4d8) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:204
    (gdb) 


::

    116 inline void CSGOptiXRenderInteractiveTest::init()
    117 {
    118     assert( irc == 0 );
    119     assert(fd);
    120     stree* tree = fd->getTree();
    121     assert(tree);
    122     SScene* scene = fd->getScene() ;
    123     assert(scene);
    124     gm->setTreeScene(tree, scene);
    125     gm->setRecordInfo(ar, br);


    0813 inline void SGLM::setTreeScene( stree* _tree, SScene* _scene )
     814 {   
     815     tree = _tree ;
     816     scene = _scene ;
     817 
     818     moi_fr = tree->get_frame_moi();
     819 



::

    3666 sframe CSGFoundry::getFrameE() const
    3667 {
    3668     bool VERBOSE = ssys::getenvbool(getFrameE_VERBOSE) ;
    3669     LOG(LEVEL) << "[" << getFrameE_VERBOSE << "] " << VERBOSE ;
    3670 
    3671     sframe fr = {} ;
    3672 
    3673     if(ssys::getenvbool("INST"))
    3674     {
    3675         int INST = ssys::getenvint("INST", 0);
    3676         LOG_IF(info, VERBOSE) << " INST " << INST ;
    3677         getFrame(fr, INST ) ;
    3678 
    3679         fr.set_ekv("INST");
    3680     }
    3681     else if(ssys::getenvbool("MOI"))
    3682     {
    3683         const char* MOI = ssys::getenvvar("MOI", nullptr) ;
    3684         LOG_IF(info, VERBOSE) << " MOI " << MOI ;
    3685         fr = getFrame() ;
    3686         fr.set_ekv("MOI");
    3687     }
    3688     else
    3689     {
    3690         const char* ipf_ = SEventConfig::InputPhotonFrame();  // OPTICKS_INPUT_PHOTON_FRAME
    3691         const char* ipf = ipf_ ? ipf_ : "0" ;
    3692         LOG_IF(info, VERBOSE) << " ipf " << ipf ;
    3693         fr = getFrame(ipf);
    3694 
    3695         fr.set_ekv(SEventConfig::kInputPhotonFrame, ipf );
    3696     }
    3697 
    3698 
    3699     return fr ;
    3700 }


    3704 /**
    3705 CSGFoundry::AfterLoadOrCreate
    3706 -------------------------------
    3707 
    3708 Called from some high level methods eg: CSGFoundry::Load
    3709 
    3710 The idea behind this is to auto connect SEvt with the frame
    3711 from the geometry.
    3712 
    3713 HMM: not called after Create, see CSGOptiX::initFrame
    3714 
    3715 **/
    3716 
    3717 void CSGFoundry::AfterLoadOrCreate() // static
    3718 {
    3719     CSGFoundry* fd = CSGFoundry::Get();
    3720 
    3721     SEvt::CreateOrReuse() ;   // creates 1/2 SEvt depending on OPTICKS_INTEGRATION_MODE
    3722 
    3723     if(!fd) return ;
    3724 
    3725     sframe fr = fd->getFrameE() ;
    3726     LOG(LEVEL) << fr ;
    3727     SEvt::SetFrame(fr); // now only needs to be done once to transform input photons
    3728 
    3729 }


    3744 /**
    3745 CSGFoundry::getCenterExtent
    3746 -------------------------------
    3747 
    3748 For midx -1 returns ce obtained from the ias bbox,
    3749 otherwise uses CSGTarget to lookup the center extent.
    3750 
    3751 For global geometry which typically means a default gord of 0
    3752 there is special handling for gord -1/-2/-3 in CSGTarget::getCenterExtent
    3753 
    3754 gord -1
    3755     uses getLocalCenterExtent
    3756 
    3757 gord -2
    3758     uses SCenterExtentFrame xyzw : ordinary XYZ frame
    3759 
    3760 gord -3
    3761     uses SCenterExtentFrame rtpw : tangential RTP frame
    3762 
    3763 
    3764 NB gord is the gas ordinal index
    3765 (it was formerly named iidx which was confusing as this is NOT the global instance index)
    3766 
    3767 
    3768 **/
    3769 
    3770 int CSGFoundry::getCenterExtent(float4& ce, int midx, int mord, int gord, qat4* m2w, qat4* w2m  ) const
    3771 {
    3772     int rc = 0 ;
    3773     if( midx == -1 )
    3774     {
    3775         unsigned long long emm = 0ull ;   // hmm instance var ?
    3776         iasCE(ce, emm);
    3777     }
    3778     else
    3779     {
    3780         rc = target->getFrameComponents(ce, midx, mord, gord, m2w, w2m );
    3781     }
    3782 
    3783     if( rc != 0 )
    3784     {
    3785         LOG(error) << " non-zero RC from CSGTarget::getCenterExtent " ;
    3786     }
    3787     return rc ;
    3788 }


    150 int CSGTarget::getFrame(sframe& fr,  int midx, int mord, int gord ) const
    151 {
    152     fr.set_midx_mord_gord( midx, mord, gord );
    153     int rc = getFrameComponents( fr.ce, midx, mord, gord, &fr.m2w , &fr.w2m );
    154     LOG(LEVEL) << " midx " << midx << " mord " << mord << " gord " << gord << " rc " << rc ;
    155     return rc ;
    156 }


Handling the special cases -1/-2/-3::

    222 int CSGTarget::getFrameComponents(float4& ce, int midx, int mord, int gord, qat4* m2w, qat4* w2m ) const
    223 {
    224     LOG(LEVEL) << " (midx mord gord) " << "(" << midx << " " << mord << " " << gord << ") " ;
    225     if( gord == -1 )
    226     {
    227         LOG(info) << "(gord == -1) qptr transform will not be set, typically defaulting to identity " ;
    228         int lrc = getLocalCenterExtent(ce, midx, mord);
    229         if(lrc != 0) return 1 ;
    230     }
    231     else if( gord == -2 || gord == -3 )
    232     {
    233         LOG(LEVEL) << "(gord == -2/-3  EXPERIMENTAL qptr transform will be set to SCenterExtentFrame transforms " ;
    234         int lrc = getLocalCenterExtent(ce, midx, mord);
    235         if(lrc != 0) return 1 ;
    236 
    237         if( gord == -2 )
    238         {
    239             bool rtp_tangential = false ;
    240             bool extent_scale = false ;  // NB recent change switching off extent scaling 
    241             SCenterExtentFrame<double> cef_xyzw( ce.x, ce.y, ce.z, ce.w, rtp_tangential, extent_scale );
    242             m2w->read_narrow(cef_xyzw.model2world_data);
    243             w2m->read_narrow(cef_xyzw.world2model_data);
    244         }
    245         else if( gord == -3 )
    246         {
    247             bool rtp_tangential = true ;
    248             bool extent_scale = false ;   // NB recent change witching off extent scaling 
    249             SCenterExtentFrame<double> cef_rtpw( ce.x, ce.y, ce.z, ce.w, rtp_tangential, extent_scale );
    250             m2w->read_narrow(cef_rtpw.model2world_data);
    251             w2m->read_narrow(cef_rtpw.world2model_data);
    252         }
    253     }
    254     else
    255     {
    256         int grc = getGlobalCenterExtent(ce, midx, mord, gord, m2w, w2m );
    257         //  HMM: the m2w here populated is from the (midx, mord, gord) instance transform, with identity info 
    258         if(grc != 0) return 2 ;
    259     }
    260     return 0 ;
    261 }


CSGFoundry uses sframe, stree uses sfr : is unification needed ?
------------------------------------------------------------------

Relationship between CSGFoundry/SSim/stree::

      80 /**
      81 CSGFoundry::CSGFoundry
      82 ------------------------
      83 
      84 HMM: the dependency between CSGFoundry and SSim is a bit mixed up
      85 because of the two possibilities:
      86 
      87 1. "Import" : create CSGFoundry from SSim/stree using CSGImport
      88 2. "Load"   : load previously created and persisted CSGFoundry + SSim from file system
      89 
      90 sim(SSim) used to be a passive passenger of CSGFoundry but now that CSGFoundry
      91 can be CSGImported from SSim it is no longer so passive.
      92 
      93 **/


stree.h::

      26 SSim+stree vs CSGFoundry
      27 --------------------------
      28 
      29 Some duplication between these is inevitable, however they have
      30 different objectives:
      31 
      32 * *SSim+stree* aims to collect and persist all needed info from Geant4
      33 * *CSGFoundry* aims to prepare the subset that needs to be uploaded to GPU
      34 
      35   * narrowing to float is something that could be done when going from stree->CSGFoundry
      36 



TRY TO RESOLVE BY ALIGHING CONVENTIONS USED IN CSGTarget::getFrameComponents AND stree::_get_frame_global
----------------------------------------------------------------------------------------------------------

:: 

    2129 /**
    2130 stree::_get_frame_global
    2131 --------------------------
    2132 
    2133 This is called for special cased -ve repeat_ordinal, which 
    2134 is only appropriate for global non-instanced volumes. 
    2135 
    2136 1. find the snode using (lvid, lvid_ordinal, ridx_type) 
    2137 2. compute bounding box and hence center_extent for the snode
    2138 3. form frame transforms m2w/w2m using SCenterExtentFrame or not
    2139    depending on repeat_ordinal -1/-2/-3
    2140 
    2141 Global repeat_ordinal special case convention
    2142 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    2143 
    2144 repeat_ordinal:-1
    2145    sets CE only, does not set m2w w2m into the frame
    2146    [WHAT USE IS THIS ?]
    2147 
    2148 repeat_ordinal:-2
    2149    sets CE, m2w, w2m into the frame using SCenterExtentFrame with rtp_tangential:false
    2150 
    2151 repeat_ordinal:-3
    2152    sets CE, m2w, w2m into the frame using SCenterExtentFrame with rtp_tangential:true
    2153 
    2154 
    2155 27 May 2025 behaviour change for repeat_ordinal:-1
    2156 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    2157 
    2158 WIP: test this
    2159 
    2160 Formerly the stree::_get_frame_global repeat_ordinal:-1 gave frames 
    2161 with transforms that CSGTarget::getFrameComponents 
    2162 would need repeat_ordinal:-2 for.
    2163 
    2164 The stree::_get_frame_global implementation is 
    2165 now aligned with CSGTarget::getFrameComponents
    2166 to avoid the need to keep swapping MOI -1/-2 arising from 
    2167 a former difference in the convention used. 
    2168 
    2169 **/
    2170 
    2171 inline int stree::_get_frame_global(sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal, char ridx_type ) const
    2172 {
    2173     assert( repeat_ordinal == -1 || repeat_ordinal == -2 || repeat_ordinal == -3 );
    2174     const snode* _node = pick_lvid_ordinal_node( lvid, lvid_ordinal, ridx_type );
    2175     if(_node == nullptr) return 1 ;
    2176     const snode& node = *_node ;
    2177     
    2178     std::array<double,4> ce = {} ;
    2179     std::array<double,6> bb = {} ;
    2180     int rc = get_node_ce_bb( ce, bb, node );



