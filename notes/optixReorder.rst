notes/optixReorder
=====================


API from OptiX 8
-----------------

* https://raytracing-docs.nvidia.com/optix8/api/group__optix__device__api.html

::

    static __forceinline__ __device__ void 	optixReorder ()
    static __forceinline__ __device__ void 	optixReorder (unsigned int coherenceHint, unsigned int numCoherenceHintBitsFromLSB)


numCoherenceHintBitsFromLSB
   eg 4 : look at least significant 4 bits of coherenceHint 


ideas for coherenceHint
---------------------------

* p.flag index : added p.flagidx() [0->15] for this
* global position radius
* form some grouped boundary, or facilitate lookup imat/omat and use that ?

Alternates to try for the coherenceHint:

1. ctx.p.time based time_bins ?
2. 3-bit boundary group combined with 3-bit temporal bin ?





how to get the flag index : added sphoton::flagidx [0->15]
-----------------------------------------------------------



coherenceHint
----------------

* https://github.com/BNLNPPS/simphony/blob/feat/ser/CSGOptiX/CSGOptiX7.cu

::

    #if OPTIX_VERSION >= 80000
            if (params.SerInterval > 0 && (bounce % params.SerInterval) == 0)
            {
                unsigned ser_key = 0u;
                switch (params.SerHint)
                {
                case 1:
                    ser_key = bounce / params.SerInterval;
                    break;
                case 2:
                    ser_key = ((ctx.p.boundary() & 7u) << 1) | (ctx.p.wavelength < 200.f ? 1u : 0u);
                    break;
                }
                optixReorder(ser_key, 4);
            }
    #endif



The ctx.p.boundary is the boundary index that establishes unique combinations
of inner and outer material and surface indices - the ordering is not
controlled and there are typically a few hundred distinct boundary indices in a
geometry. So trying to use three bits from it would seem to be rather random
and unlikely to yield any useful coherence ? Do you have counter arguments ?
Any suggestions for what else could be used to better provide a coherence hint
?

The primary cause of divergence is that the photon lifetimes can be very
different. Photons right next to each other in the same volume and about to hit
the same surface can have very different lifetimes. Doesnt that suggest that
spatial grouping would be less useful than temporal grouping ?

You have hit on the core tension between traditional computer graphics ray
tracing and Monte Carlo particle transport. Your observation is entirely
correct: in optical simulations, spatial proximity does not guarantee
similarity in history, state, or remaining lifespan.

This exact reality explains why temporal and lifecycle grouping (Hint #1) is
far more important for Monte Carlo simulation than pure spatial grouping.

The Two Competing Forces: Trace vs. Transport

To understand why your insight is spot-on, it helps to look at what happens
during a single iteration of the while loop:


1. The trace() Phase (Ray-Geometry Intersection):

   * This is handled directly by the OptiX hardware ray tracer.

   * What it cares about: Spatial locality. If two photons are physically close to
     each other—even if one is on bounce 2 and the other is on bounce 50—their rays
     traverse the same bounding volume hierarchy (BVH) nodes. Grouping them
     spatially ensures that the GPU cores fetch the same acceleration structure
     nodes from memory, maximizing cache hits.

2. The propagate() & Branching Phase (Physics / State Updates):

   * This is handled by custom CUDA code (propagate logic, scattering,
     absorption checks, boundary handling).

   * What it cares about: Lifecycle and branch coherence. If a warp contains a mix
     of newly born photons and photons about to terminate, the warp suffers from
     massive thread divergence and idle lanes. Furthermore, if photon lifetimes
     diverge wildly, dead threads begin to accumulate in warps, starving the GPU of
     active work.





Cosmic muon simulations with many hundreds of millions or even billions of
photons is a important use case of Opticks - is trying 6 bits worthwhile ?

At the scale of hundreds of millions or billions of concurrent photons (such as
cosmic muon bundles passing through a massive detector), the calculus changes
completely.

In that regime, trying 6 bits (64 bins) is absolutely worthwhile.


The primary risk of a higher bin count (like 6 bits) is that bins become
under-populated, leaving fewer than 32 threads per bin and causing warp
starvation. However, when your initial launch size is 1 billion threads

::

    // 6-bit breakdown: 3 bits temporal, 3 bits material/state category
    unsigned int time_bin = (bounce / params.SerInterval) & 0x7;          // 3 bits (8 temporal steps)
    unsigned int state_bin = get_high_level_physics_state(ctx.p) & 0x7;    // 3 bits (8 physics states)
    unsigned int ser_key  = state_bin | (time_bin << 3);                 // Combined 6-bit key



The below is an extract from the CUDA kernel of the Simphony Opticks fork.
Please explain the optixReorder usage::

    static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    {
        sevent* evt = params.evt ;
        if (launch_idx.x >= evt->num_seed) return;   // was evt->num_photon

        unsigned idx = launch_idx.x ;
        unsigned genstep_idx = evt->seed[idx] ;
        const quad6& gs = evt->genstep[genstep_idx] ;
        // genstep needs the raw index, from zero for each genstep slice sub-launch

        unsigned long long photon_idx = params.photon_slot_offset + idx ;
        // 2025/10/20 change from unsigned to avoid clocking photon_idx and duplicating
        //
        // rng_state access and array recording needs the absolute photon_idx
        // for multi-launch and single-launch simulation to match.
        // The offset hides the technicality of the multi-launch from output.

        qsim* sim = params.sim ;

        RNG rng ;
        sim->rng->init( rng, sim->evt->index, photon_idx );

        sctx ctx = {} ;
        ctx.evt = evt ;   // sevent.h
        ctx.prd = prd ;   // squad.h quad2

        ctx.idx = idx ;
        ctx.pidx = photon_idx ;

    #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
        ctx.pidx_debug = sim->base->pidx == photon_idx ;
    #endif

        sim->generate_photon(ctx.p, rng, gs, photon_idx, genstep_idx );
        // Only Cerenkov and scintillation gensteps store a source-material line in
        // q0.u.z. TORCH and FRAME use that slot for a genstep id, so leave the
        // carried line invalid for all other genstep types.
        const unsigned gentype = gs.q0.u.x;
        const bool carries_matline = CSGOptiX7_GenstepCarriesMaterialLine(gentype);
        ctx.current_matline = carries_matline ? gs.q0.u.z : 0xFFFFFFFFu;

        FlowAction command = FlowAction::Start;
        int bounce = 0 ;
        if(evt->record) ctx.point(bounce);
        while( bounce < evt->max_bounce && ctx.p.time < params.max_time )
        {
            float tmin = ( ctx.p.orient_boundary_flag & params.PropagateEpsilon0Mask ) ? params.tmin0 : params.tmin ;

    #if OPTIX_VERSION >= 80000
            if (params.SerInterval > 0 && (bounce % params.SerInterval) == 0)
            {
                unsigned ser_key = 0u;
                switch (params.SerHint)
                {
                case 1:
                    ser_key = bounce / params.SerInterval;
                    break;
                case 2:
                    ser_key = ((ctx.p.boundary() & 7u) << 1) | (ctx.p.wavelength < 200.f ? 1u : 0u);
                    break;
                }
                optixReorder(ser_key, 4);
            }
    #endif

            // intersect query filling (quad2)prd
            switch(params.PropagateRefine)
            {
                case 0u: trace<false>( params.handle, ctx.p.pos, ctx.p.mom, tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
                case 1u: trace<true>(  params.handle, ctx.p.pos, ctx.p.mom, tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
            }

            if( prd->boundary() == 0xffffu ) break ; // SHOULD ONLY HAPPEN FOR PHOTONS STARTING OUTSIDE WORLD
            // propagate can do nothing meaningful without a boundary

            // HMM: normalize here or within CSG ? Actually only needed for
            // geometry with active scaling, such as ellipsoid.
            // TODO: move this so its only done when needed
            //     ~/o/notes/issues/CSGOptiX_simulate_avoid_normalizing_every_normal.rst
            //

            float3* normal = prd->normal();
            *normal = normalize(*normal);

    #ifndef PRODUCTION
            ctx.trace(bounce);
    #endif
            command = sim->propagate(bounce, rng, ctx);
            bounce++;
            if(evt->record) ctx.point(bounce) ;
            if(command == FlowAction::Break) break;
        }
    #ifndef PRODUCTION
        ctx.end();  // write seq, tag, flat
    #endif


        if( evt->photon )
        {
            evt->photon[idx] = ctx.p ;  // *idx* (not *photon_idx*) as needs to go from zero for photons from a slice of genstep array
        }









Initial Check (lo) : medium_evt : 100M photons 10 genstep
----------------------------------------------------------

::

    TEST=medium_evt cxs_min.sh

    ...
    sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792395314.727872 1792395314.737066              9194                 0                 2          0.009194                              init 
                             load_geom 1792395314.737066 1792395316.543765           1806699                 2                 3          1.806699                         load_geom 
                           upload_geom 1792395316.543805 1792395316.896839            353034                 4                 5          0.353034                       upload_geom 
                         slice_genstep 1792395316.897158 1792395316.897194                36                 9                10          0.000036                     slice_genstep 
                  upload genstep slice 1792395316.897201 1792395316.932829             35628                11                12          0.035628              upload genstep slice 
                        simulate slice 1792395316.932829 1792395348.042201          31109372                12                13         31.109372                    simulate slice 
                        download slice 1792395348.042201 1792395348.401072            358871                13                14          0.358871                    download slice 
                        concat slices  1792395348.401145 1792395348.401167                22                15                16          0.000022                    concat slices  
                          save arrays  1792395348.401190 1792395348.974603            573413                17                22          0.573413                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792395314727872 2026-10-19T15:35:14.727
                                TOTAL: 16131557941580467 16131557975826736          34246269                83                97


    ]sreport.desc_runprof
    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792395565.381901 1792395565.391286              9385                 0                 2          0.009385                              init 
                             load_geom 1792395565.391286 1792395568.739588           3348302                 2                 3          3.348302                         load_geom 
                           upload_geom 1792395568.739635 1792395569.068797            329162                 4                 5          0.329162                       upload_geom 
                         slice_genstep 1792395569.069144 1792395569.069181                37                 9                10          0.000037                     slice_genstep 
                  upload genstep slice 1792395569.069188 1792395569.105105             35917                11                12          0.035917              upload genstep slice 
                        simulate slice 1792395569.105105 1792395600.402010          31296905                12                13         31.296905                    simulate slice 
                        download slice 1792395600.402010 1792395600.765469            363459                13                14          0.363459                    download slice 
                        concat slices  1792395600.765556 1792395600.765580                24                15                16          0.000024                    concat slices  
                          save arrays  1792395600.765604 1792395607.572834           6807230                17                22          6.807230                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792395565381901 2026-10-19T15:39:25.381
                                TOTAL: 16131560208689429 16131560250879850          42190421                83                97

     0 :                ta : ta
     1 :                tb : tb
     2 :                ab : ab





Enabling SER - cuts a third of the time::


    ]sreport.desc_runprof
    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792395736.001683 1792395736.010607              8924                 0                 2          0.008924                              init 
                             load_geom 1792395736.010607 1792395738.804113           2793506                 2                 3          2.793506                         load_geom 
                           upload_geom 1792395738.804155 1792395739.137216            333061                 4                 5          0.333061                       upload_geom 
                         slice_genstep 1792395739.137531 1792395739.137567                36                 9                10          0.000036                     slice_genstep 
                  upload genstep slice 1792395739.137573 1792395739.173158             35585                11                12          0.035585              upload genstep slice 
                        simulate slice 1792395739.173158 1792395761.172138          21998980                12                13         21.998980                    simulate slice 
                        download slice 1792395761.172138 1792395761.528129            355991                13                14          0.355991                    download slice 
                        concat slices  1792395761.528198 1792395761.528220                22                15                16          0.000022                    concat slices  
                          save arrays  1792395761.528244 1792395768.226710           6698466                17                22          6.698466                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792395736001683 2026-10-19T15:42:16.001
                                TOTAL: 16131561712493287 16131561744717858          32224571                83                97


Check reproducibility::

    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792395945.725984 1792395945.735997             10013                 0                 2          0.010013                              init 
                             load_geom 1792395945.735997 1792395947.545850           1809853                 2                 3          1.809853                         load_geom 
                           upload_geom 1792395947.545892 1792395947.874361            328469                 4                 5          0.328469                       upload_geom 
                         slice_genstep 1792395947.874689 1792395947.874727                38                 9                10          0.000038                     slice_genstep 
                  upload genstep slice 1792395947.874734 1792395947.910399             35665                11                12          0.035665              upload genstep slice 
                        simulate slice 1792395947.910399 1792395969.925805          22015406                12                13         22.015406                    simulate slice 
                        download slice 1792395969.925805 1792395970.282517            356712                13                14          0.356712                    download slice 
                        concat slices  1792395970.282590 1792395970.282614                24                15                16          0.000024                    concat slices  
                          save arrays  1792395970.282637 1792395976.995215           6712578                17                22          6.712578                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792395945725984 2026-10-19T15:45:45.725
                                TOTAL: 16131563593158727 16131563624427485          31268758                83                97

     0 :                ta : ta



After adding hintMethod:2 but before trying it::

    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792403553.687116 1792403553.697434             10318                 0                 2          0.010318                              init 
                             load_geom 1792403553.697434 1792403555.519887           1822453                 2                 3          1.822453                         load_geom 
                           upload_geom 1792403555.519925 1792403562.371342           6851417                 4                 5          6.851417                       upload_geom 
                         slice_genstep 1792403562.371707 1792403562.371748                41                 9                10          0.000041                     slice_genstep 
                  upload genstep slice 1792403562.371755 1792403562.407504             35749                11                12          0.035749              upload genstep slice 
                        simulate slice 1792403562.407504 1792403584.388512          21981008                12                13         21.981008                    simulate slice 
                        download slice 1792403584.388512 1792403584.982603            594091                13                14          0.594091                    download slice 
                        concat slices  1792403584.982658 1792403584.982680                22                15                16          0.000022                    concat slices  
                          save arrays  1792403584.982701 1792403591.487044           6504343                17                22          6.504343                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792403553687116 2026-10-19T17:52:33.687
                                TOTAL: 16131632104409312 16131632142208754          37799442                83                97

     0 :                ta : ta


Now try::

    [Params_Helper::set_SER
     SER                        : hintMethod:2,coherenceHintBits:4,bounceInterval:4
     hintMethod                 : 2
     bounceInterval             : 4
     coherenceHintBits          : 4
    ]Params_Helper::set_SER




    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792403683.642459 1792403683.652998             10539                 0                 2          0.010539                              init 
                             load_geom 1792403683.652998 1792403685.439370           1786372                 2                 3          1.786372                         load_geom 
                           upload_geom 1792403685.439409 1792403685.771612            332203                 4                 5          0.332203                       upload_geom 
                         slice_genstep 1792403685.771943 1792403685.771980                37                 9                10          0.000037                     slice_genstep 
                  upload genstep slice 1792403685.771986 1792403685.807708             35722                11                12          0.035722              upload genstep slice 
                        simulate slice 1792403685.807708 1792403707.601964          21794256                12                13         21.794256                    simulate slice 
                        download slice 1792403707.601964 1792403708.195047            593083                13                14          0.593083                    download slice 
                        concat slices  1792403708.195109 1792403708.195132                23                15                16          0.000023                    concat slices  
                          save arrays  1792403708.195155 1792403714.663062           6467907                17                22          6.467907                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792403683642459 2026-10-19T17:54:43.642
                                TOTAL: 16131633234078731 16131633265098873          31020142                83                97









::

    2026-10-19 15:42:19.173 INFO  [226252] [CSGOptiX::prepareParamSimulate_SER@1061] [CSGOptiX::prepareParamSimulate_SER
     SEventConfig::kSER         : OPTICKS_SER
     SEventConfig::_SERDefault  : coherenceHintBits:0,bounceInterval:0
     SEventConfig::SER()        : coherenceHintBits:4,bounceInterval:4
     bounceInterval             : 4
     coherenceHintBits          : 4
    ]CSGOptiX::prepareParamSimulate_SER





