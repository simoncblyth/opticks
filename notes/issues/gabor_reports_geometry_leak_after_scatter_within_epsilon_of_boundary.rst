gabor_reports_geometry_leak_after_scatter_within_epsilon_of_boundary
=====================================================================


Gabor::

    The radii of the spheres are  15 mm and 25 mm so there are no surfaces close to
    coincidence. I have also tested this leakage with a box and still got photons
    leaking. The exact model is the following:
     
    https://github.com/BNLNPPS/esi-g4ox/blob/LeakTest/geom/sphere_leak.gdml
     
              +-----------------------------+
              |         WorldBox            |
              |       (radius 50 mm)        |
              |                             |
              |  +----------------------+   |
              |  |  DetectorSphere      |   |   ← Thin shell (30 mm - 25 mm)
              |  |                      |   |
              |  |  +---------------+   |   |
              |  |  | GlassSphere   |   |   |   ← Inner core (15 mm)
              |  |  |               |   |   |
              |  |  +---------------+   |   |
              |  +----------------------+   |
              +-----------------------------+
     
    I performed several simulations with different OPTICKS_PROPAGATE_EPSILON values
    from 1e-07 to the default setting.
     
    What I have seen is that for very low OPTICKS_PROPAGATE_EPSILON more photons
    start to leak. This is due to single precision. However what is surprising is
    that above 50 nm the number of leaking photons increase.
     
    An other team that uses OptiX (with triangle intersection only) also saw
    teleporting photons. They consulted with NVIDIA engineers and their conclusion
    was that if there is a scattering event close to the surface the photons can
    escape.
     
    The photons can escape if OPTICKS_PROPAGATE_EPSILON is smaller than the
    distance from the new StepStart position, which makes sense. Since OptiX will
    discard an intersection with the surface if the distance is smaller than
    ray_tmin.
     
    I am currently preparing a publication, will add a subsection to describe this.
     
    Sure, we will add a test so you can also take a look at this.




Hmm howabout a 2nd smaller propagate epsilon [could be zero] post scatter or reemission (hmm generation too?)
------------------------------------------------------------------------------------------------------------------

* would need to choose between params.tmin and "params.tmin_after_scatter" based on the photon flag from prior 



::

    313 //#define OLD_WITHOUT_SKIPAHEAD 1
    314 #ifdef OLD_WITHOUT_SKIPAHEAD
    315     RNG rng = sim->rngstate[photon_idx] ;
    316 #else
    317     RNG rng ;
    318     sim->rng->init( rng, sim->evt->index, photon_idx );
    319 #endif
    320 
    321     sctx ctx = {} ;
    322     ctx.evt = evt ;
    323     ctx.prd = prd ;
    324     //ctx.idx = idx ;
    325     ctx.idx = photon_idx ; // 2025/06 change to absolute idx for PIDX dumping
    326 
    327     sim->generate_photon(ctx.p, rng, gs, photon_idx, genstep_idx );
    328 
    329     int command = START ;
    330     int bounce = 0 ;
    331 #ifndef PRODUCTION
    332     ctx.point(bounce);
    333 #endif
    334     while( bounce < evt->max_bounce )
    335     {


                float u_tmin = ctx.p.

    336         trace( params.handle, ctx.p.pos, ctx.p.mom, params.tmin, params.tmax, prd, params.vizmask );  // geo query filling prd
    337         if( prd->boundary() == 0xffffu ) break ; // SHOULD ONLY HAPPEN FOR PHOTONS STARTING OUTSIDE WORLD
    338         // propagate can do nothing meaningful without a boundary
    339 
    340         // HMM: normalize here or within CSG ? Actually only needed for
    341         // geometry with active scaling, such as ellipsoid.
    342         // TODO: move this so its only done when needed
    343         //     ~/o/notes/issues/CSGOptiX_simulate_avoid_normalizing_every_normal.rst
    344         //
    345 
    346         float3* normal = prd->normal();
    347         *normal = normalize(*normal);
    348 
    349 #ifndef PRODUCTION
    350         ctx.trace(bounce);
    351 #endif
    352         command = sim->propagate(bounce, rng, ctx);
    353         bounce++;
    354 #ifndef PRODUCTION
    355         ctx.point(bounce) ;
    356 #endif
    357         if(command == BREAK) break ;
    358     }


::

    155     SPHOTON_METHOD unsigned flag() const {     return boundary_flag & 0xffffu ; } // flag___     = lambda p:p.view(np.uint32)[...,3,0] & 0xffff



    float u_tmin = ctx.p.boundary_flag & (BULK_REEMIT | BULK_SCATTER | CERENKOV | SCINTILLATION | TORCH ) ?  params.tmin0 : params.tmin ;


::

     22 enum
     23 {
     24     CERENKOV          = 0x1 <<  0,    
     25     SCINTILLATION     = 0x1 <<  1,    
     26     MISS              = 0x1 <<  2,
     27     BULK_ABSORB       = 0x1 <<  3,
     28     BULK_REEMIT       = 0x1 <<  4,
     29     BULK_SCATTER      = 0x1 <<  5,
     30     SURFACE_DETECT    = 0x1 <<  6,
     31     SURFACE_ABSORB    = 0x1 <<  7,
     32     SURFACE_DREFLECT  = 0x1 <<  8,
     33     SURFACE_SREFLECT  = 0x1 <<  9,
     34     BOUNDARY_REFLECT  = 0x1 << 10,
     35     BOUNDARY_TRANSMIT = 0x1 << 11,
     36     TORCH             = 0x1 << 12,
     37     NAN_ABORT         = 0x1 << 13,
     38     EFFICIENCY_CULL    = 0x1 << 14,
     39     EFFICIENCY_COLLECT = 0x1 << 15,
     40     __NATURAL         = 0x1 << 16,
     41     __MACHINERY       = 0x1 << 17,
     42     __EMITSOURCE      = 0x1 << 18,
     43     PRIMARYSOURCE     = 0x1 << 19,
     44     GENSTEPSOURCE     = 0x1 << 20,
     45     DEFER_FSTRACKINFO = 0x1 << 21
     46 }; 
     47 








