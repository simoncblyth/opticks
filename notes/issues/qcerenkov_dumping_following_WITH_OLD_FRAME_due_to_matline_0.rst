qcerenkov_dumping_following_WITH_OLD_FRAME_due_to_matline_0
=============================================================

Overview
----------

Caused by omitted set_matline in SEvt::addGenstep::

    2444 sgs SEvt::addGenstep(const quad6& q_)
    2445 {
    2446     LOG_IF(info, LIFECYCLE) << id() ;
    2447     dbg->addGenstep++ ;
    2448     LOG(LEVEL) << " index " << index << " instance " << instance ;
    2449
    2450     unsigned gentype = q_.gentype();
    2451     unsigned matline_ = q_.matline();
    2452
    2453     bool is_cerenkov_gs = OpticksGenstep_::IsCerenkov(gentype);
    2454
    2455     int gidx = int(gs.size())  ;  // 0-based genstep label index
    2456     bool enabled = GIDX == -1 || GIDX == gidx ;
    2457
    2458     quad6& q = const_cast<quad6&>(q_);
    2459     if(!enabled) q.set_numphoton(0);
    2460     // simplify handling of disabled gensteps by simply setting numphoton to zero for them
    2461
    2462     if(matline_ >= G4_INDEX_OFFSET  )
    2463     {
    2464         unsigned mtindex = matline_ - G4_INDEX_OFFSET ;
    2465 #ifdef WITH_OLD_FRAME
    2466         assert(cf);
    2467         int matline = cf ? cf->lookup_mtline(mtindex) : 0 ;
    2468 #else
    2469         // omitting this caused ~/o/notes/issues/qcerenkov_dumping_following_WITH_OLD_FRAME_due_to_matline_0.rst
    2470         assert(sim);
    2471         int matline = sim ? sim->lookup_mtline(mtindex) : 0 ;
    2472 #endif
    2491         q.set_matline(matline);  // <=== THIS IS CHANGING GS BACK IN CALLERS SCOPE
    2492     }


Looking back at when the omission started its probably from::

    2026-02-06 c9cfeaab6 - extend sfr.h to replace sframe.h and hide all use of the old sframe.h behind WITH_OLD_FRAME,
                           all tests passing but strong potential for lurking issues from widespread change eg when return to cxs_min.sh



Issue : TEST=hitlitemerged ojt : getting lots of dumping thats not usually there
----------------------------------------------------------------------------------

* /data2/blyth/local/oj_test/20260318/hitlitemerged/

::

    2026-03-18 15:37:20.524 INFO  [3897404] [SEvt::setNumPhoton@2614]  evt->num_photon 9195 evt->num_tag 0 evt->num_flat 0
    //qcerenkov::wavelength_sampled_bndtex idx   1280 sampledRI   1.000 cosTheta   1.136 sin2Theta   0.000 wavelength 154.707 count 100 matline 0
    //qcerenkov::wavelength_sampled_bndtex idx   2784 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 192.791 count 100 matline 0
    //qcerenkov::wavelength_sampled_bndtex idx   2785 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 125.202 count 100 matline 0
    //qcerenkov::wavelength_sampled_bndtex idx   2786 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 102.389 count 100 matline 0
    //qcerenkov::wavelength_sampled_bndtex idx   2787 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 162.512 count 100 matline 0
    //qcerenkov::wavelength_sampled_bndtex idx   2788 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 471.085 count 100 matline 0


* HMM : this means those wavelengths are not properly sampled, they are just uniform randoms in the range
* DONE: distinguish : bad material, bad genstep, bad matline ?

  * it was caused by matline always zero due to omitted mtline lookup in SEvt::addGenstep


matline 0 is Air ?
~~~~~~~~~~~~~~~~~~~

* HMM: formerly the first material was Galactic which was never used for Cerenkov,
  potentially this issue showing up now is due to some Cerenkov trying to
  happen with matline zero

::

    (ok) A[blyth@localhost material]$ cat NPFold_index.txt
    Air
    Rock
    Galactic
    Steel
    LS
    Scintillator
    TiO2Coating
    Adhesive
    Aluminium
    Tyvek
    Water
    PVDF
    LatticedShellSteel
    Vacuum
    Pyrex
    Acrylic
    PE_PA
    StrutSteel
    AcrylicMask
    CDReflectorSteel
    Teflon
    vetoWater
    Black_HDPE
    (ok) A[blyth@localhost material]$ pwd
    /home/blyth/junosw/InstallArea/blyth-OJ-pmt-hit-type-2-for-muon-hits/.opticks/GEOM/J25_7_2_opticks_Debug/CSGFoundry/SSim/stree/material
    (ok) A[blyth@localhost material]$


    In [1]: f.mtname_names
    Out[1]:
    array(['Air', 'Rock', 'Galactic', 'Steel', 'LS', 'Scintillator', 'TiO2Coating', 'Adhesive', 'Aluminium', 'Tyvek', 'Water', 'PVDF', 'LatticedShellSteel', 'Vacuum', 'Pyrex', 'Acrylic', 'PE_PA',
           'StrutSteel', 'AcrylicMask', 'CDReflectorSteel', 'Teflon', 'vetoWater', 'Black_HDPE'], dtype='<U18')

    In [2]: f.mtname_names.shape
    Out[2]: (23,)


HUH, looks like mtline:0 is still Galactic::

    In [3]: f.mtline
    Out[3]:
    array([[ 15],
           [  7],
           [  0],
           [ 27],
           [ 39],
           [ 59],
           [ 55],
           [ 51],
           [ 47],
           [155],
           [171],
           [167],
           [187],
           [303],
           [299],
           [555],
           [567],
           [587],
           [615],
           [619],
           [691],
           [143],
           [139]], dtype=int32)


This makes me want to know where the potentially dud gs is giving the warning.

First ones before the kernel output buffer filled are near origin, in LS::

    2026-03-18 16:38:07.799 INFO  [4055749] [QSim::simulate@487]    0 : sslice {       0,      90,         0,      9195}  0.009195
    2026-03-18 16:38:07.825 INFO  [4055749] [SEvt::setNumPhoton@2614]  evt->num_photon 9195 evt->num_tag 0 evt->num_flat 0
    //qcerenkov::wavelength_sampled_bndtex idx   1280 sampledRI   1.000 cosTheta   1.136 sin2Theta   0.000 wavelength 154.707 count 100 gs.matline 0 gs.pos (425.867, 42.649,128.000)
    //qcerenkov::wavelength_sampled_bndtex idx   2784 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 192.791 count 100 gs.matline 0 gs.pos (426.203, 43.198,128.248)
    //qcerenkov::wavelength_sampled_bndtex idx   2785 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 125.202 count 100 gs.matline 0 gs.pos (426.203, 43.198,128.248)
    //qcerenkov::wavelength_sampled_bndtex idx   2786 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 102.389 count 100 gs.matline 0 gs.pos (426.203, 43.198,128.248)
    //qcerenkov::wavelength_sampled_bndtex idx   2787 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 162.512 count 100 gs.matline 0 gs.pos (426.203, 43.198,128.248)
    //qcerenkov::wavelength_sampled_bndtex idx   2788 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 471.085 count 100 gs.matline 0 gs.pos (426.203, 43.198,128.248)
    //qcerenkov::wavelength_sampled_bndtex idx   2789 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 101.372 count 100 gs.matline 0 gs.pos (426.203, 43.198,128.248)
    //qcerenkov::wavelength_sampled_bndtex idx   2790 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength  97.554 count 100 gs.matline 0 gs.pos (426.203, 43.198,128.248)
    //qcerenkov::wavelength_sampled_bndtex idx   2791 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 590.119 count 100 gs.matline 0 gs.pos (426.203, 43.198,128.248)
    //qcerenkov::wavelength_sampled_bndtex idx   2792 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 103.062 count 100 gs.matline 0 gs.pos (426.203, 43.198,128.248)
    //qcerenkov::wavelength_sampled_bndtex idx   2793 sampledRI   1.000 cosTheta   1.206 sin2Theta   0.000 wavelength 166.407 count 100 gs.matline 0 gs.pos (426.203, 43.198,128.248)



Genstep collection::

    082 static quad6 MakeGenstep_DsG4Scintillation_r4695(
     83      const G4Track* aTrack,
     84      const G4Step* aStep,
     85      G4int    numPhotons,
     86      G4int    scnt,
     87      G4double ScintillationTime
     88     )
     89 {
     90     G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
     91     G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();
     92
     93     G4ThreeVector x0 = pPreStepPoint->GetPosition();
     94     G4double      t0 = pPreStepPoint->GetGlobalTime();
     95     G4ThreeVector deltaPosition = aStep->GetDeltaPosition() ;
     96     G4double meanVelocity = (pPreStepPoint->GetVelocity()+pPostStepPoint->GetVelocity())/2. ;
     97
     98     const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();
     99     const G4Material* aMaterial = aTrack->GetMaterial();
    100
    101     quad6 _gs ;
    102     _gs.zero() ;
    103
    104     sscint* gs = (sscint*)(&_gs) ;   // warning: dereferencing type-punned pointer will break strict-aliasing rules
    105
    106     gs->gentype = OpticksGenstep_DsG4Scintillation_r4695 ;
    107     gs->trackid = aTrack->GetTrackID() ;
    108     gs->matline = aMaterial->GetIndex() + SEvt::G4_INDEX_OFFSET ;  // offset signals that a mapping must be done in SEvt::setGenstep
    109     gs->numphoton = numPhotons ;
    110
    111     // note that gs->matline is not currently used for scintillation,
    112     // but done here as check of SEvt::addGenstep mtindex to mtline mapping
    113
    114     gs->pos.x = x0.x() ;

    196 static quad6 MakeGenstep_G4Cerenkov_modified(
    197     const G4Track* aTrack,
    198     const G4Step* aStep,
    199     G4int    numPhotons,
    200     G4double    betaInverse,
    201     G4double    pmin,
    202     G4double    pmax,
    203     G4double    maxCos,
    204
    205     G4double    maxSin2,
    206     G4double    meanNumberOfPhotons1,
    207     G4double    meanNumberOfPhotons2
    208     )
    209 {
    210     G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
    211     G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();
    212
    213     G4ThreeVector x0 = pPreStepPoint->GetPosition();
    214     G4double      t0 = pPreStepPoint->GetGlobalTime();
    215     G4ThreeVector deltaPosition = aStep->GetDeltaPosition() ;
    216
    217     const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();
    218
    219     G4double Wmin_nm = h_Planck*c_light/pmax/nm ;
    220     G4double Wmax_nm = h_Planck*c_light/pmin/nm ;
    221
    222     const G4Material* aMaterial = aTrack->GetMaterial();
    223
    224     quad6 _gs ;
    225     _gs.zero() ;
    226
    227     scerenkov* gs = (scerenkov*)(&_gs) ;
    228
    229     gs->gentype = OpticksGenstep_G4Cerenkov_modified ;
    230     gs->trackid = aTrack->GetTrackID() ;
    231     gs->matline = aMaterial->GetIndex() + SEvt::G4_INDEX_OFFSET ;  // offset signals that a mapping must be done in SEvt::setGenstep
    232     gs->numphoton = numPhotons ;
    233
    234     gs->pos.x = x0.x() ;
    235     gs->pos.y = x0.y() ;
    236     gs->pos.z = x0.z() ;
    237     gs->time = t0 ;





matline
~~~~~~~~~

::

    (ok) A[blyth@localhost qudarap]$ opticks-f matline
    ./qcerenkov_dev.h:    unsigned line = gs.matline ; //   line :  4*boundary_idx + OMAT/IMAT (0/3)
    ./qcerenkov_dev.h:    unsigned line = gs.matline ; //   line :  4*boundary_idx + OMAT/IMAT (0/3)
    ./qcerenkov_dev.h:    int matline = 0u ;
    ./qcerenkov_dev.h:    scerenkov::FillGenstep(gs, matline, numphoton_per_genstep, false );
    ./qcerenkov_dev.h:    int matline = 0u ;
    ./qcerenkov_dev.h:    scerenkov::FillGenstep(gs, matline, numphoton_per_genstep, false );
    ./qcerenkov_dev.h:    int matline = 0u ;
    ./qcerenkov_dev.h:    scerenkov::FillGenstep(gs, matline, numphoton_per_genstep, false );
    ./qcerenkov.h:    //printf("//qcerenkov::wavelength_sampled_bndtex bnd %p gs.matline %d \n", bnd, gs.matline );
    ./qcerenkov.h:        float4 props = bnd->boundary_lookup(wavelength, gs.matline, 0u);
    ./qcerenkov.h:    printf("//qcerenkov::wavelength_sampled_bndtex idx %6lld sampledRI %7.3f cosTheta %7.3f sin2Theta %7.3f wavelength %7.3f count %d matline %d \n",
    ./qcerenkov.h:              idx , sampledRI, cosTheta, sin2Theta, wavelength, count, gs.matline );
    ./QDebug.cc:    unsigned cerenkov_matline = qb ? qb->qb->boundary_tex_MaterialLine_LS : 0 ;
    ./QDebug.cc:         << "AS NO QBnd at QDebug::MakeInstance the qdebug cerenkov genstep is using default matline of zero " << std::endl
    ./QDebug.cc:         << " cerenkov_matline " << cerenkov_matline  << std::endl
    ./QDebug.cc:    scerenkov::FillGenstep( cerenkov_gs, cerenkov_matline, 100, dump );
    (ok) A[blyth@localhost qudarap]$




    2440 sgs SEvt::addGenstep(const quad6& q_)
    2441 {
    2442     LOG_IF(info, LIFECYCLE) << id() ;
    2443     dbg->addGenstep++ ;
    2444     LOG(LEVEL) << " index " << index << " instance " << instance ;
    2445
    2446     unsigned gentype = q_.gentype();
    2447     unsigned matline_ = q_.matline();
    2450     bool is_cerenkov_gs = OpticksGenstep_::IsCerenkov(gentype);
    2451
    2465     int gidx = int(gs.size())  ;  // 0-based genstep label index
    2466     bool enabled = GIDX == -1 || GIDX == gidx ;
    2467
    2468     quad6& q = const_cast<quad6&>(q_);
    2469     if(!enabled) q.set_numphoton(0);
    2470     // simplify handling of disabled gensteps by simply setting numphoton to zero for them
    2471
    2472



    2473     if(matline_ >= G4_INDEX_OFFSET  )
    2474     {
    2475         unsigned mtindex = matline_ - G4_INDEX_OFFSET ;
    2476         int matline = cf ? cf->lookup_mtline(mtindex) : 0 ;
    2477         // cf(SGeo) used for lookup
    2478         // BUT: that just uses SSim::lookup_mtline
    2479         // so SEvt should hold sim(SSim) ?




    2480
    2481         bool bad_ck = is_cerenkov_gs && matline == -1 ;
    2482
    2483         LOG_IF(info, bad_ck )
    2484             << " is_cerenkov_gs " << ( is_cerenkov_gs ? "YES" : "NO " )
    2485             << " cf " << ( cf ? "YES" : "NO " )
    2486             << " bad_ck "
    2487             << " matline_ " << matline_
    2488             << " matline " << matline
    2489             << " gentype " << gentype
    2490             << " mtindex " << mtindex
    2491             << " G4_INDEX_OFFSET " << G4_INDEX_OFFSET
    2492             << " desc_mt "
    2493             << std::endl
    2494             << ( cf ? cf->desc_mt() : "no-cf" )
    2495             << std::endl
    2496             ;
    2497
    2498         q.set_matline(matline);  // <=== THIS IS CHANGING GS BACK IN CALLERS SCOPE
    2499
    2500     }
    2501
    2502
    2503 #ifdef SEVT_NUMPHOTON_FROM_GENSTEP_CHECK
    2504     int64_t numphoton_from_genstep = getNumPhotonFromGenstep() ; // sum numphotons from all previously collected gensteps (since last clear)
    2505     assert( numphoton_from_genstep == numphoton_collected );
    2506 #endif
    2507
    2508     int64_t q_numphoton = q.numphoton() ;          // numphoton in this genstep
    2509     if(q_numphoton > numphoton_genstep_max) numphoton_genstep_max = q_numphoton ;



smoky gun : fairly recent change in SEvt frame handling ... SEvt may be missing cf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1138 #ifdef WITH_OLD_FRAME
    1139 /**
    1140 SEvt::setGeo
    1141 -------------
    1142
    1143 SGeo is a protocol for geometry access fulfilled by CSGFoundry (and formerly by GGeo)
    1144
    1145 This connection between the SGeo geometry and SEvt is what allows
    1146 the appropriate instance frame to be accessed. That is vital for
    1147 looking up the sensor_identifier and sensor_index.
    1148
    1149 TODO: replace this with stree.h based approach
    1150
    1151
    1152 Invoked with stack::
    1153
    1154     SEvt::setGeo
    1155     CSGOptiX::InitEvt
    1156     CSGOptiX::Create
    1157     CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:156
    1158     main
    1159
    1160
    1161 **/
    1162
    1163 void SEvt::setGeo(const SGeo* cf_)
    1164 {
    1165     assert(cf_);
    1166     cf = cf_ ;
    1167     tree = cf->getTree();
    1168 }
    1169
    1170 #else
    1171 /**
    1172 SEvt::setSim
    1173 -------------
    1174
    1175 This aims to remove SEvt::setGeo and CSGFoundry SGeo base
    1176 **/
    1177
    1178 void SEvt::setSim(const SSim* sim_)
    1179 {
    1180     assert(sim_);
    1181     sim = sim_ ;
    1182     tree = sim->get_tree();
    1183 }
    1184
    1185 #endif





BINGO : SEvt had sim but was not using it to do the lookups
------------------------------------------------------------
















qcerenkov::wavelength_sampled_bndtex
--------------------------------------

::

    285 inline QCERENKOV_METHOD void qcerenkov::wavelength_sampled_bndtex(float& wavelength, float& cosTheta, float& sin2Theta, RNG& rng, const scerenkov& gs, unsigned long long idx, int gsid ) const
    286 {
    287     //printf("//qcerenkov::wavelength_sampled_bndtex bnd %p gs.matline %d \n", bnd, gs.matline );
    288     float u0 ;
    289     float u1 ;
    290     float w ;
    291     float sampledRI ;
    292     float u_maxSin2 ;
    293
    294     unsigned count = 0 ;
    295
    296     do {
    297         u0 = curand_uniform(&rng) ;
    298
    299         w = gs.Wmin + u0*(gs.Wmax - gs.Wmin) ;
    300
    301         wavelength = gs.Wmin*gs.Wmax/w ; // reciprocalization : arranges flat energy distribution, expressed in wavelength
    302
    303         float4 props = bnd->boundary_lookup(wavelength, gs.matline, 0u);
    304
    305         sampledRI = props.x ;
    306
    307         //printf("//qcerenkov::wavelength_sampled_bndtex count %d wavelength %10.4f sampledRI %10.4f \n", count, wavelength, sampledRI );
    308
    309         cosTheta = gs.BetaInverse / sampledRI ;
    310
    311         sin2Theta = fmaxf( 0.f, (1.f - cosTheta)*(1.f + cosTheta));
    312
    313         u1 = curand_uniform(&rng) ;
    314
    315         u_maxSin2 = u1*gs.maxSin2 ;
    316
    317         count += 1 ;
    318
    319     } while ( u_maxSin2 > sin2Theta && count < 100 );
    320
    321 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    322     if(count > 50)
    323     printf("//qcerenkov::wavelength_sampled_bndtex idx %6lld sampledRI %7.3f cosTheta %7.3f sin2Theta %7.3f wavelength %7.3f count %d matline %d \n",
    324               idx , sampledRI, cosTheta, sin2Theta, wavelength, count, gs.matline );
    325 #endif
    326
    327 }

