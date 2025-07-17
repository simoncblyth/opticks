simtrace_intersection_precision_test
======================================

Overview
---------

simtrace is more appropriate for testing intersect precision
than input photons : so dont have to fight the simulation
randomization.

Hence, need to add some simtrace functionality
that uses SRecord::getPhotonAtTime

Need to do something analogous to input photon gensteps but for simtrace


Changes
--------

* add OpticksGenstep_INPUT_PHOTON_SIMTRACE
* generalize qsim::generate_photon_simtrace
* add simtrace layout to SRecord::getPhotonAtTime and SRecord::getSimtraceAtTime


::

    2447 inline QSIM_METHOD void qsim::generate_photon_simtrace(quad4& p, RNG& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    2448 {
    2449     const int& gencode = gs.q0.i.x ;
    2450     switch(gencode)
    2451     {
    2452         case OpticksGenstep_FRAME:                   generate_photon_simtrace_frame(p, rng, gs, photon_id, genstep_id ); break ;
    2453         case OpticksGenstep_INPUT_PHOTON_SIMTRACE:   { p = (quad4&)evt->photon[photon_id] ; }                          ; break ;
    2454     }
    2455 }
    2456 
    2457 inline QSIM_METHOD void qsim::generate_photon_simtrace_frame(quad4& p, RNG& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    2458 {
    2459     C4U gsid ;
    2460 
    2461     //int gencode          = gs.q0.i.x ;
    2462     int gridaxes           = gs.q0.i.y ;  // { XYZ, YZ, XZ, XY }



TOFIX : more than one simtrace layout
-----------------------------------------

::

    469 static __forceinline__ __device__ void simtrace( const uint3& launch_idx, const uint3& dim, quad2* prd )
    470 {
    471     unsigned idx = launch_idx.x ;
    472     sevent* evt  = params.evt ;
    473     if (idx >= evt->num_simtrace) return;    // num_slot for multi launch simtrace ?
    474 
    475     unsigned genstep_idx = evt->seed[idx] ;
    476     unsigned photon_idx  = params.photon_slot_offset + idx ;
    477     // photon_idx same as idx for first launch, offset beyond first for multi-launch
    478 
    479 #if defined(DEBUG_PIDX)
    480     if(photon_idx == 0) printf("//CSGOptiX7.cu : simtrace idx %d photon_idx %d  genstep_idx %d evt->num_simtrace %d \n", idx, photon_idx, genstep_idx, evt->num_simtrace );
    481 #endif
    482 
    483     const quad6& gs = evt->genstep[genstep_idx] ;
    484 
    485     qsim* sim = params.sim ;
    486     RNG rng ;
    487     sim->rng->init(rng, 0, photon_idx) ;
    488 
    489     quad4 p ;
    490     sim->generate_photon_simtrace(p, rng, gs, photon_idx, genstep_idx );
    491 
    492     
    493     // HUH: this is not the layout of sevent::add_simtrace
    494     const float3& pos = (const float3&)p.q0.f  ;
    495     const float3& mom = (const float3&)p.q1.f ;
    496 





Review
---------

input photon gensteps
~~~~~~~~~~~~~~~~~~~~~~~

::

    094 NP* SEvent::MakeInputPhotonGenstep(const NP* input_photon, const sframe& fr )
     95 {
     96     std::vector<quad6> qgs(1) ;
     97     qgs[0].zero() ;
     98     qgs[0] = MakeInputPhotonGenstep_(input_photon, fr );
     99     NP* ipgs = NPX::ArrayFromVec<float,quad6>( qgs, 6, 4) ;
    100     return ipgs ;
    101 }

    117 quad6 SEvent::MakeInputPhotonGenstep_(const NP* input_photon, const sframe& fr )
    118 {
    119     LOG(LEVEL) << " input_photon " << NP::Brief(input_photon) ;
    120 
    121     quad6 ipgs ;
    122     ipgs.zero();
    123     ipgs.set_gentype( OpticksGenstep_INPUT_PHOTON );
    124     ipgs.set_numphoton(  input_photon->shape[0]  );
    125     fr.m2w.write(ipgs); // copy fr.m2w into ipgs.q2,q3,q4,q5
    126     return ipgs ;
    127 }



    0317 int QEvent::setGenstepUpload(const quad6* qq0, int gs_start, int gs_stop )
     318 {
     ... 
     395     int gencode0 = SGenstep::GetGencode(qq, 0) ; // gencode of first genstep or OpticksGenstep_INVALID for qq nullptr
     396 
     397     if(OpticksGenstep_::IsFrame(gencode0))   // OpticksGenstep_FRAME  (HMM: Obtuse, maybe change to SIMTRACE ?)
     398     {
     399         setNumSimtrace( evt->num_seed );
     400     }
     401     else if(OpticksGenstep_::IsInputPhoton(gencode0)) // OpticksGenstep_INPUT_PHOTON  (NOT: _TORCH)
     402     {
     403         setInputPhotonAndUpload();
     404     }
     405     else
     406     {
     407         setNumPhoton( evt->num_seed );  // *HEAVY* : photon, rec, record may be allocated here depending on SEventConfig
     408     }
     409     upload_count += 1 ;



     497 void QEvent::setInputPhotonAndUpload()
     498 {
     499     LOG_IF(info, LIFECYCLE) ;
     500     LOG(LEVEL);
     501     input_photon = sev->gatherInputPhoton();
     502     checkInputPhoton();
     503 
     504     int numph = input_photon->shape[0] ;
     505     setNumPhoton( numph );
     506     QU::copy_host_to_device<sphoton>( evt->photon, (sphoton*)input_photon->bytes(), numph );
     507 }


qsim::generate_photon::


    2509 inline QSIM_METHOD void qsim::generate_photon(sphoton& p, RNG& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    2510 {
    2511     const int& gencode = gs.q0.i.x ;
    2512     switch(gencode)
    2513     {
    2514         case OpticksGenstep_CARRIER:         scarrier::generate(     p, rng, gs, photon_id, genstep_id)  ; break ;
    2515         case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ;
    2516 
    2517         case OpticksGenstep_G4Cerenkov_modified:
    2518         case OpticksGenstep_CERENKOV:
    2519                                               cerenkov->generate(    p, rng, gs, photon_id, genstep_id ) ; break ;
    2520 
    2521         case OpticksGenstep_DsG4Scintillation_r4695:
    2522         case OpticksGenstep_SCINTILLATION:
    2523                                               scint->generate(        p, rng, gs, photon_id, genstep_id ) ; break ;
    2524 
    2525         case OpticksGenstep_INPUT_PHOTON:    { p = evt->photon[photon_id] ; p.set_flag(TORCH) ; }        ; break ;
    2526         default:                             generate_photon_dummy(  p, rng, gs, photon_id, genstep_id)  ; break ;
    2527     }
    2528     p.set_idx(photon_id);
    2529 }




cxt_min.sh configures simtrace gensteps with CEGS CEHIGH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    188 ## see SFrameGenstep::StandardizeCEGS for CEGS/CEHIGH [4]/[7]/[8] layouts
    189 
    190 export CEGS=16:0:9:2000   # [4] XZ default
    191 #export CEGS=16:0:9:1000  # [4] XZ default
    192 #export CEGS=16:0:9:100   # [4] XZ reduce rays for faster rsync
    193 #export CEGS=16:9:0:1000  # [4] try XY
    194 
    195 export CEHIGH_0=-16:16:0:0:-4:4:2000:4
    196 export CEHIGH_1=-16:16:0:0:4:8:2000:4
    197 
    198 #export CEHIGH_0=16:0:9:0:0:10:2000     ## [7] dz:10 aim to land another XZ grid above in Z 16:0:9:2000
    199 #export CEHIGH_1=-4:4:0:0:-9:9:2000:5   ## [8]
    200 #export CEHIGH_2=-4:4:0:0:10:28:2000:5  ## [8]
    201 



QSim::simtrace
~~~~~~~~~~~~~~~~

::

     664 double QSim::simtrace(int eventID)
     665 {
     666     sev->beginOfEvent(eventID);
     667 
     668     NP* igs = sev->makeGenstepArrayFromVector();
     669     int rc = event->setGenstepUpload_NP(igs) ;
     670 
     671     LOG_IF(error, rc != 0) << " QEvent::setGenstep ERROR : no gensteps collected : will skip cx.simtrace " ;
     672 
     673     sev->t_PreLaunch = sstamp::Now() ;
     674     double dt = rc == 0 && cx != nullptr ? cx->simtrace_launch() : -1. ;
     675     sev->t_PostLaunch = sstamp::Now() ;
     676     sev->t_Launch = dt ;
     677 
     678     // see ~/o/notes/issues/cxt_min_simtrace_revival.rst
     679     sev->gather();
     680 
     681     sev->topfold->concat();
     682     sev->topfold->clear_subfold();
     683 
     684     sev->endOfEvent(eventID);
     685 
     686     return dt ;
     687 }



 
SEvt::addInputGenstep
~~~~~~~~~~~~~~~~~~~~~~~

::

     859 void SEvt::addInputGenstep()
     860 {
     861     LOG_IF(info, LIFECYCLE) << id() ;
     862     LOG(LEVEL);
     863 
     864     if(SEventConfig::IsRGModeSimtrace())
     865     {
     866         const char* frs = frame.get_frs() ; // nullptr when default -1 : meaning all geometry
     867 
     868         LOG_IF(info, SIMTRACE )
     869             << "[" << SEvt__SIMTRACE << "] "
     870             << " frame.get_frs " << ( frs ? frs : "-" ) ;
     871             ;
     872 
     873         //if(frs) SEventConfig::SetEventReldir(frs); // dont do that, default is more standard
     874         // doing this is hangover from separate simtracing of related volumes presumably
     875 
     876         NP* gs = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(frame);
     877         LOG_IF(info, SIMTRACE)
     878             << "[" << SEvt__SIMTRACE << "] "
     879             << " simtrace gs " << ( gs ? gs->sstr() : "-" )
     880             ;
     881 
     882         addGenstep(gs);
     883 
     884         if(frame.is_hostside_simtrace()) setFrame_HostsideSimtrace();
     885     }


CSGOptiX7.cu::

    469 static __forceinline__ __device__ void simtrace( const uint3& launch_idx, const uint3& dim, quad2* prd )
    470 {
    471     unsigned idx = launch_idx.x ;
    472     sevent* evt  = params.evt ;
    473     if (idx >= evt->num_simtrace) return;    // num_slot for multi launch simtrace ?
    474 
    475     unsigned genstep_idx = evt->seed[idx] ;
    476     unsigned photon_idx  = params.photon_slot_offset + idx ;
    477     // photon_idx same as idx for first launch, offset beyond first for multi-launch
    478 
    479 #if defined(DEBUG_PIDX)
    480     if(photon_idx == 0) printf("//CSGOptiX7.cu : simtrace idx %d photon_idx %d  genstep_idx %d evt->num_simtrace %d \n", idx, photon_idx, genstep_idx, evt->num_simtrace );
    481 #endif
    482 
    483     const quad6& gs = evt->genstep[genstep_idx] ;
    484 
    485     qsim* sim = params.sim ;
    486     RNG rng ;
    487     sim->rng->init(rng, 0, photon_idx) ;
    488 
    489     quad4 p ;
    490     sim->generate_photon_simtrace(p, rng, gs, photon_idx, genstep_idx );
    491 
    492     const float3& pos = (const float3&)p.q0.f  ;
    493     const float3& mom = (const float3&)p.q1.f ;
    494 
    495 
    496 #if defined(DEBUG_PIDX)
    497     if(photon_idx == 0) printf("//CSGOptiX7.cu : simtrace idx %d pos.xyz %7.3f,%7.3f,%7.3f mom.xyz %7.3f,%7.3f,%7.3f  \n", idx, pos.x, pos.y, pos.z, mom.x, mom.y, mom.z );
    498 #endif
    499 
    500 
    501 
    502 
    503     trace<false>(
    504         params.handle,
    505         pos,
    506         mom,
    507         params.tmin,
    508         params.tmax,
    509         prd,
    510         params.vizmask,
    511         params.PropagateRefineDistance
    512     );
    513 
    514     evt->add_simtrace( idx, p, prd, params.tmin );  // sevent
    515     // not photon_idx, needs to go from zero for photons from a slice of genstep array
    516 }



qsim::generate_photon_simtrace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2447 inline QSIM_METHOD void qsim::generate_photon_simtrace(quad4& p, RNG& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    2448 {
    2449     C4U gsid ;
    2450 
    2451     //int gencode          = gs.q0.i.x ;
    2452     int gridaxes           = gs.q0.i.y ;  // { XYZ, YZ, XZ, XY }
    2453     gsid.u                 = gs.q0.i.z ;
    2454     //unsigned num_photons = gs.q0.u.w ;
    2455 
    2456     p.q0.f.x = gs.q1.f.x ;   // start with genstep local frame position, typically origin  (0,0,0)
    2457     p.q0.f.y = gs.q1.f.y ;
    2458     p.q0.f.z = gs.q1.f.z ;
    2459     p.q0.f.w = 1.f ;
    2460 
    2461     //printf("//qsim.generate_photon_simtrace gridaxes %d gs.q1 (%10.4f %10.4f %10.4f %10.4f) \n", gridaxes, gs.q1.f.x, gs.q1.f.y, gs.q1.f.z, gs.q1.f.w );
    2462 
    2463     float u0 = curand_uniform(&rng);
    2464     float sinPhi, cosPhi;
    2465 #if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    2466     __sincosf(2.f*M_PIf*u0,&sinPhi,&cosPhi);
    2467 #else
    2468     sincosf(2.f*M_PIf*u0,&sinPhi,&cosPhi);
    2469 #endif
    2470 
    2471     float u1 = curand_uniform(&rng);
    2472     float cosTheta = 2.f*u1 - 1.f ;
    2473     float sinTheta = sqrtf(1.f-cosTheta*cosTheta) ;
    2474 
    2475     //printf("//qsim.generate_photon_simtrace u0 %10.4f sinPhi   %10.4f cosPhi   %10.4f \n", u0, sinPhi, cosPhi );
    2476     //printf("//qsim.generate_photon_simtrace u1 %10.4f sinTheta %10.4f cosTheta %10.4f \n", u1, sinTheta, cosTheta );
    2477     //printf("//qsim.generate_photon_simtrace  u0 %10.4f sinPhi   %10.4f cosPhi   %10.4f u1 %10.4f sinTheta %10.4f cosTheta %10.4f \n",  u0, sinPhi, cosPhi, u1, sinTheta, cosTheta );
    2478 
    2479     switch( gridaxes )
    2480     {
    2481         case YZ:  { p.q1.f.x = 0.f    ;  p.q1.f.y = cosPhi ;  p.q1.f.z = sinPhi ;  p.q1.f.w = 0.f ; } ; break ;
    2482         case XZ:  { p.q1.f.x = cosPhi ;  p.q1.f.y = 0.f    ;  p.q1.f.z = sinPhi ;  p.q1.f.w = 0.f ; } ; break ;
    2483         case XY:  { p.q1.f.x = cosPhi ;  p.q1.f.y = sinPhi ;  p.q1.f.z = 0.f    ;  p.q1.f.w = 0.f ; } ; break ;
    2484         case XYZ: { p.q1.f.x = sinTheta*cosPhi ;
    2485                     p.q1.f.y = sinTheta*sinPhi ;
    2486                     p.q1.f.z = cosTheta        ;
    2487                     p.q1.f.w = 0.f ; } ; break ;   // previously used XZ
    2488     }
    2489 
    2490 
    2491     qat4 qt(gs) ; // copy 4x4 transform from last 4 quads of genstep
    2492     qt.right_multiply_inplace( p.q0.f, 1.f );   // position
    2493     qt.right_multiply_inplace( p.q1.f, 0.f );   // direction



