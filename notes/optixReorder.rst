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
