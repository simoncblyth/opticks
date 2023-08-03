sphoton_iindex_identity_CPU_GPU_difference
===========================================

Context
---------

* previous ~/j/issues/opticksMode3-contents-comparison.rst


Open Questions
-----------------

* is the wider history deviation a result of this iindex/identity discrepancy between A/B ?

workflow
----------


config, run, pullback, ana, repeat::

    jxv         # laptop, for example change "ntds" ipho stats to 10k 
    jxscp       # laptop, scp jx.bash to remote 

    jxf               # workstation, pick up updated jx.bash functions 
    ntds3_noxj        # workstation, run opticksMode:3 doing both optical simulations in one invokation
    jxf ; ntds3_noxj  # workstation : generally need to do both 


    GEOM tmpget  # laptop, pullback the paired SEvt 
    jxn          # laptop, cd to /Users/blyth/j/ntds
    ./ntds3.sh   # laptop, run analysis ntds3.py loading two SEvt into ipython for comparison, plotting 



sphoton.h
-----------

+----+----------------+----------------+----------------+----------------+--------------------------+
| q  |      x         |      y         |     z          |      w         |  notes                   |
+====+================+================+================+================+==========================+
|    |  pos.x         |  pos.y         |  pos.z         |  time          |                          |
| q0 |                |                |                |                |                          |
|    |                |                |                |                |                          |
+----+----------------+----------------+----------------+----------------+--------------------------+
|    |  mom.x         |  mom.y         | mom.z          |  iindex        |                          |
| q1 |                |                |                | (unsigned)     |                          |
|    |                |                |                |                |                          |
+----+----------------+----------------+----------------+----------------+--------------------------+
|    |  pol.x         |  pol.y         |  pol.z         |  wavelength    |                          |
| q2 |                |                |                |                |                          |
|    |                |                |                |                |                          |
+----+----------------+----------------+----------------+----------------+--------------------------+
|    | boundary_flag  |  identity      |  orient_idx    |  flagmask      |  (unsigned)              |
| q3 |                |                |  orient:1bit   |                |                          |
|    |                |                |                |                |                          |
+----+----------------+----------------+----------------+----------------+--------------------------+



iindex
--------


HMM, iindex is discrepant::

    In [12]: a.f.record[AIDX,:4,1,3].view(np.int32)
    Out[12]: array([1065353216,      39216,      39216,      39216], dtype=int32)

    In [13]: b.f.record[AIDX,:4,1,3].view(np.int32)
    Out[13]: array([   0,    0,    0, 3354], dtype=int32)


Somewhere 1.f is planted into iindex:u::

    In [2]: np.array([1.], dtype=np.float32 ).view(np.int32)
    Out[2]: array([1065353216], dtype=int32)



Check hit iindex, B:CPU looks as would expect with 42k/100k hits with same iindex 3354::

    In [21]: np.c_[np.unique( b.f.hit[:,1,3].view(np.int32), return_counts=True ))
    Out[21]: 
    array([[     4,      1],
           [    29,      1],
           [    66,      1],
           [    84,      1],
           [    89,      1],
           [   100,      1],
           ...
           [  3006,     40],
           [  3128,      1],
           [  3179,      1],
           [  3180,      1],
           [  3181,      2],
           [  3353,      1],
           [  3354,  42549],
           [  3452,      1],
           [  3528,      1],
           [  3702,     28],
           [  3703,      1],
           [  3705,      1],
           [  3756,      1],
           [  4143,      1],
           ...
           [ 17567,      1],
           [ 17578,      1],
           [ 17590,      1],
           [ 17603,      1],
           [300060,      1],
           [300209,      1],
           [304583,      1],
           [310634,      1],
           [310949,      1],
           [312853,      2],
           [314267,      1],
           [317239,      1],
           [317336,    132],
           [317337,     19],
           [317818,     26],
           [317819,    103],
           [319130,      1],
           [320026,      1],
           [322647,      1],
           [322685,      1],
           [325472,      2]])



Very different sphoton iindex for hits between A and B
---------------------------------------------------------

Look at hits because they are all onto a PMT boundary::

    ahit = np.c_[np.unique( a.f.hit[:,1,3].view(np.int32), return_counts=True )]  
    bhit = np.c_[np.unique( b.f.hit[:,1,3].view(np.int32), return_counts=True )]  

    In [34]: ahit[np.where(ahit[:,1] > 5)]
    Out[34]: 
    array([[17337,    91],
           [17338,    28],
           [17819,    30],
           [17820,    91],
           [25762,    10],
           [26101,    13],
           [26728,    15],
           [26968,     9],
           ...
           [36511,     6],
           [37509,     7],
           [38516,    14],
           [38579,     7],
           [39124,    77],
           [39216, 35066],
           [39707,     8],
           [40135,    11],
           [43000,     6],
           [43208,     6]])

    In [35]: bhit[np.where(bhit[:,1] > 5)]
    Out[35]: 
    array([[  3006,     40],
           [  3354,  42549],
           [  3702,     28],
           [317336,    132],
           [317337,     19],
           [317818,     26],
           [317819,    103]])



::

    ## iindex 

    In [20]: ahit[:,0].min(),ahit[:,0].max()    ## this is the 0-base IAS index
    Out[20]: (673, 43210)

    In [21]: bhit[:,0].min(),bhit[:,0].max()    ## this is pmtid 
    Out[21]: (4, 325472)

    ## identity 

    In [22]: ahid[:,0].min(),ahid[:,0].max()    ## this is sensor_identifier, probably pmtid+1 
    Out[22]: (17, 325574)

    In [23]: bhid[:,0].min(),bhid[:,0].max()    ## unfilled all 0 
    Out[23]: (0, 0)





j/ntds/ntds3.py::

     32     ahit_ = a.f.hit[:,1,3].view(np.int32)   ## iindex
     33     bhit_ = b.f.hit[:,1,3].view(np.int32)   
     34     
     35     ahid_ = a.f.hit[:,3,1].view(np.int32)   ## identity
     36     bhid_ = b.f.hit[:,3,1].view(np.int32)   
     37     
     38     ahit = np.c_[np.unique( ahit_, return_counts=True )]
     39     bhit = np.c_[np.unique( bhit_, return_counts=True )]
     40     
     41     ahid = np.c_[np.unique( ahid_, return_counts=True )]
     42     bhid = np.c_[np.unique( bhid_, return_counts=True )]


::

    In [12]: bhit[bhit[:,1]>10]   
    Out[12]: 
    array([[  3006,     40],
           [  3354,  42549],
           [  3702,     28],
           [317336,    132],
           [317337,     19],
           [317818,     26],
           [317819,    103]])

    In [13]: ahid[ahid[:,1]>10]  ## ahid + 1  looks like bhit 
    Out[13]: 
    array([[   707,     13],
           [  1001,     14],
           [  1566,     15],
           [  3007,     77],
           [  3355,  35066],
           [  3703,     45],
           [  5821,     13],
           [  6645,     11],
           [  7929,     11],
           [317337,     91],
           [317338,     28],
           [317819,     30],
           [317820,     91]])

    In [14]: bhid                      ## bhid all zero 
    Out[14]: array([[    0, 43143]])


TODO: review where (1,3) sphoton::iindex and (3,1) sphoton::identity comes from in A/B GPU/CPU side
-----------------------------------------------------------------------------------------------------

::

    348 extern "C" __global__ void __raygen__rg()
    349 {
    350     const uint3 idx = optixGetLaunchIndex();
    351     const uint3 dim = optixGetLaunchDimensions();
    352 
    353     quad2 prd ;
    354     prd.zero();
    355  
    356     switch( params.raygenmode )
    357     {
    358         case SRG_RENDER:    render(   idx, dim, &prd ) ; break ;
    359         case SRG_SIMTRACE:  simtrace( idx, dim, &prd ) ; break ;
    360         case SRG_SIMULATE:  simulate( idx, dim, &prd ) ; break ;
    361     }
    362 }


    242 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    243 {
    244     sevent* evt = params.evt ;
    245     if (launch_idx.x >= evt->num_photon) return;
    246 
    247     unsigned idx = launch_idx.x ;  // aka photon_idx
    248     unsigned genstep_idx = evt->seed[idx] ;
    249     const quad6& gs = evt->genstep[genstep_idx] ;
    250 
    251     qsim* sim = params.sim ;
    252     curandState rng = sim->rngstate[idx] ;    // TODO: skipahead using an event_id 
    253 
    254     sctx ctx = {} ;
    255     ctx.evt = evt ;
    256     ctx.prd = prd ;
    257     ctx.idx = idx ;
    258 
    259     sim->generate_photon(ctx.p, rng, gs, idx, genstep_idx );
    260 
    261     int command = START ;
    262     int bounce = 0 ; 
    266     while( bounce < evt->max_bounce )
    267     {   
    268         trace( params.handle, ctx.p.pos, ctx.p.mom, params.tmin, params.tmax, prd);  // geo query filling prd      
    269         if( prd->boundary() == 0xffffu ) break ; // SHOULD ONLY HAPPEN FOR PHOTONS STARTING OUTSIDE WORLD
    274         command = sim->propagate(bounce, rng, ctx);
    275         bounce++;    
    279         if(command == BREAK) break ;
    280     }   
    284     evt->photon[idx] = ctx.p ;
    285 }

* above trace causes a CH(closest-hit) call that populates prd(quad2)

cx/CSGOptiX7.cu::

    447 extern "C" __global__ void __closesthit__ch()
    448 {
    449     unsigned iindex = optixGetInstanceIndex() ;    // 0-based index within IAS
    450     unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build 
    451     unsigned prim_idx = optixGetPrimitiveIndex() ; // GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    452 
    453     //unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ; 
    454     unsigned identity = instance_id ;  // CHANGED July 2023, as now carrying sensor_identifier, see sysrap/sqat4.h 
    455 
    456 #ifdef WITH_PRD
    457     quad2* prd = getPRD<quad2>();
    458 
    459     prd->set_identity( identity ) ;

    460     prd->set_iindex(   iindex ) ;

    461     //printf("//__closesthit__ch prd.boundary %d \n", prd->boundary() );  // boundary set in IS for WITH_PRD
    462     float3* normal = prd->normal();
    463     *normal = optixTransformNormalFromObjectToWorldSpace( *normal ) ;
    464 
    465 #else
    466     const float3 local_normal =    // geometry object frame normal at intersection point 
    467         make_float3(
    468                 uint_as_float( optixGetAttribute_0() ),
    469                 uint_as_float( optixGetAttribute_1() ),
    470                 uint_as_float( optixGetAttribute_2() )
    471                 );
    472 
    473     const float distance = uint_as_float(  optixGetAttribute_3() ) ;
    474     unsigned boundary = optixGetAttribute_4() ;
    475     const float lposcost = uint_as_float( optixGetAttribute_5() ) ;
    476     float3 normal = optixTransformNormalFromObjectToWorldSpace( local_normal ) ;
    477 
    478     setPayload( normal.x, normal.y, normal.z, distance, identity, boundary, lposcost, iindex );  // communicate from ch->rg
    479 #endif
    480 }


* first thing done by qsim::propagate is to pass prd into ctx.p with sphoton::set_prd

::

    1726 inline QSIM_METHOD int qsim::propagate(const int bounce, curandStateXORWOW& rng, sctx& ctx )
    1727 {
    1728     const unsigned boundary = ctx.prd->boundary() ;
    1729     const unsigned identity = ctx.prd->identity() ;
    1730     const unsigned iindex = ctx.prd->iindex() ;
    1731     const float lposcost = ctx.prd->lposcost() ;  // local frame intersect position cosine theta 
    1732 
    1733     const float3* normal = ctx.prd->normal();
    1734     float cosTheta = dot(ctx.p.mom, *normal ) ;
    1735 
    1736 #ifdef DEBUG_PIDX
    1737     if( ctx.idx == base->pidx )
    1738     printf("//qsim.propagate idx %d bnc %d cosTheta %10.4f dir (%10.4f %10.4f %10.4f) nrm (%10.4f %10.4f %10.4f) \n",
    1739                  ctx.idx, bounce, cosTheta, ctx.p.mom.x, ctx.p.mom.y, ctx.p.mom.z, normal->x, normal->y, normal->z );
    1740 #endif
    1741 
    1742     ctx.p.set_prd(boundary, identity, cosTheta, iindex );  // HMM: lposcost not passed along 
    1743 
    1744     bnd->fill_state(ctx.s, boundary, ctx.p.wavelength, cosTheta, ctx.idx );
    1745 
    1746     unsigned flag = 0 ;
    1747 
    1748     int command = propagate_to_boundary( flag, rng, ctx );



Where is the corresponding CPU code ?::

    epsilon:u4 blyth$ opticks-f set_prd
    ./ana/p.py:     67     SPHOTON_METHOD void set_prd( unsigned  boundary, unsigned  identity, float  orient );
    ./ana/p.py:    105 SPHOTON_METHOD void sphoton::set_prd( unsigned  boundary_, unsigned  identity_, float  orient_ )
    ./sysrap/squad.h:    SQUAD_METHOD void set_prd( unsigned  boundary, unsigned  identity, float  orient ); 
    ./sysrap/squad.h:SQUAD_METHOD void quad4::set_orient( float orient )  // not typically used as set_prd more convenient, but useful for debug 
    ./sysrap/squad.h:SQUAD_METHOD void quad4::set_prd( unsigned  boundary, unsigned  identity, float  orient )
    ./sysrap/tests/squadTest.cc:void test_quad4_set_idx_set_prd_get_idx_get_prd()
    ./sysrap/tests/squadTest.cc:        p.set_prd( boundary[0], identity[0], orient[0] ); 
    ./sysrap/tests/squadTest.cc:    test_quad4_set_idx_set_prd_get_idx_get_prd(); 
    ./sysrap/sphoton.h:    SPHOTON_METHOD void set_prd( unsigned  boundary, unsigned  identity, float  orient, unsigned iindex );
    ./sysrap/sphoton.h:SPHOTON_METHOD void sphoton::set_prd( unsigned  boundary_, unsigned  identity_, float  orient_, unsigned iindex_ )
    ./qudarap/qsim.h:    ctx.p.set_prd(boundary, identity, cosTheta, iindex );  // HMM: lposcost not passed along 
    epsilon:opticks blyth$ 


::

     767 template <typename T>
     768 void U4Recorder::UserSteppingAction_Optical(const G4Step* step)
     769 {
     770     const G4Track* track = step->GetTrack();
     771     G4VPhysicalVolume* pv = track->GetVolume() ;
     772     const G4VTouchable* touch = track->GetTouchable();
     ...
     785     SEvt* sev = SEvt::Get_ECPU();
     786     sev->checkPhotonLineage(ulabel);
     787 
     788     sphoton& current_photon = sev->current_ctx.p ;
     789     quad4&   current_aux    = sev->current_ctx.aux ;
     790     current_aux.zero_v(3, 3);   // may be set below
     ...

     816 /*
     817 #ifdef U4RECORDER_EXPENSIVE_IINDEX
     818     // doing replica number search for every step is very expensive and often pointless
     819     // its the kind of thing to do only for low stats or simple geometry running 
     820     current_photon.iindex = U4Touchable::ReplicaNumber(touch, REPLICA_NAME_SELECT);  
     821 #else
     822     current_photon.iindex = is_surface_flag ? U4Touchable::ReplicaNumber(touch, REPLICA_NAME_SELECT) : -2 ;  
     823 #endif
     824 */
     825     current_photon.iindex = is_detect_flag ? 
     826               U4Touchable::ImmediateReplicaNumber(touch)
     827               :  
     828               U4Touchable::AncestorReplicaNumber(touch) 
     829               ;
     830 
     831    







     36 void U4StepPoint::Update(sphoton& photon, const G4StepPoint* point)  // static
     37 {
     38     const G4ThreeVector& pos = point->GetPosition();
     39     const G4ThreeVector& mom = point->GetMomentumDirection();
     40     const G4ThreeVector& pol = point->GetPolarization();
     41 
     42     G4double time = point->GetGlobalTime();
     43     G4double energy = point->GetKineticEnergy();
     44     G4double wavelength = h_Planck*c_light/energy ;
     45     
     46     photon.pos.x = pos.x();
     47     photon.pos.y = pos.y();
     48     photon.pos.z = pos.z();
     49     photon.time  = time/ns ;
     50 
     51     photon.mom.x = mom.x();
     52     photon.mom.y = mom.y();
     53     photon.mom.z = mom.z();
     54     //photon.iindex = 0u ; 
     55 
     56     photon.pol.x = pol.x();
     57     photon.pol.y = pol.y();
     58     photon.pol.z = pol.z();
     59     photon.wavelength = wavelength/nm ;
     60 }






(3,0) also different
----------------------

* A (3,0) has boundary index in upper 16 bits, B lacks that 

* TODO: CPU side boundary ? 

::

    In [15]: a.f.hit[:,3,0].view(np.int32)
    Out[15]: array([1966144, 2555968, 2883648, 2555968, 1966144, ..., 1966144, 2555968, 2883648, 2555968, 2555968], dtype=int32)

    In [16]: b.f.hit[:,3,0].view(np.int32)
    Out[16]: array([64, 64, 64, 64, 64, ..., 64, 64, 64, 64, 64], dtype=int32)

    In [17]: a.f.hit[:,3,0].view(np.int32) & 0xffff
    Out[17]: array([64, 64, 64, 64, 64, ..., 64, 64, 64, 64, 64], dtype=int32)

    In [18]: np.all( b.f.hit[:,3,0].view(np.int32)  == 64 )
    Out[18]: True

    In [19]: np.all( ( a.f.hit[:,3,0].view(np.int32) & 0xffff ) == 64 )
    Out[19]: True

    In [20]: a.f.hit[:,3,0].view(np.int32) >> 16
    Out[20]: array([30, 39, 44, 39, 30, ..., 30, 39, 44, 39, 39], dtype=int32)

    In [21]: np.c_[np.unique( a.f.hit[:,3,0].view(np.int32) >> 16, return_counts=True )]
    Out[21]:
    array([[   30, 35483],
           [   39,   889],
           [   44,   325]])


    In [15]: cf.sim.stree.standard.bnd_names[np.array([30,39,44])]
    Out[15]: 
    array(['Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum',
           'Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum',
           'Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum'], dtype='<U122')






