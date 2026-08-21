opticks_simulation_truncation_MAX_BOUNCE_MAX_TIME
===================================================

Overview
---------

Optical simulation need truncations to avoid costing resources for very long lived photons.

* within Opticks code "bounce" is a colloquialism for the step index between optical generation/interaction
* for example photons forming the primary bow of a rainbow by bouncing in a droplet of water would have a history like::

      TO BT BR BT SA   # TO:TORCH generation BT:BOUNDARY_TRANSMIT BR:BOUNDARY_REFLECT SA:SURFACE_ABSORB
      0  1  2  3  4    # bounce index (more precisely step index)


Controlling Truncation via envvars
----------------------------------

Two environment variables (with default values) may be used to truncate the Opticks simulation. Both are active. 

+-------------------------------------+----------------------+----------------------------+------------------------+----------------------------------------------------+
|   CUDA CSGOptiX/CSGOptiX7.cu        |  Envvar              |  sysrap static method      |   Default Value        |   Note                                             |
+=====================================+======================+============================+========================+====================================================+
|   (sevent*)evt->max_bounce          | OPTICKS_MAX_BOUNCE   | SEventConfig::MaxBounce()  |     31                 |  0-based integer generation/interaction step index |    
+-------------------------------------+----------------------+----------------------------+------------------------+----------------------------------------------------+
|   (sevent*)evt->max_time            | OPTICKS_MAX_TIME     | SEventConfig::MaxTime()    |      1.e27f            |  units are ns : so default is ~unlimited           |
+-------------------------------------+----------------------+----------------------------+------------------------+----------------------------------------------------+


How to observe the impact of truncation
------------------------------------------

To study truncation impact on your simulation results and speed:

1. repeat with changed envvars, eg::

    source /path/to/envset.sh
    export OPTICKS_MAX_BOUNCE=64

2. if you have access to photon step counts and end times - compare histograms of these between simulations


Choosing what is an appropriate truncation is a compromise between resource usage and accuracy of time tails.


Introspecting the maxima within your running environment with SEventConfigTest binary
--------------------------------------------------------------------------------------

After sourcing your $ENVSET to get into the environment:: 

    A[blyth@localhost ~]$ echo $ENVSET
    /cvmfs/opticks.ihep.ac.cn/oj/releases/LatestRelease/el9_amd64_gcc11/LastRef/envset.sh
    A[blyth@localhost ~]$ 
    A[blyth@localhost ~]$ realpath $ENVSET
    /cvmfs/opticks.ihep.ac.cn/oj/releases/J26.4.1_Opticks-v0.6.6/el9_amd64_gcc11/2026_08_14/envset.sh
    A[blyth@localhost ~]$ 
    A[blyth@localhost ~]$ which SEventConfigTest
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.6/lib/SEventConfigTest
    A[blyth@localhost ~]$ 
    A[blyth@localhost ~]$ SEventConfigTest
    2026-08-21 11:22:00.158 INFO  [1198293] [SEventConfigTest::Desc@28] 
    [SEventConfig::Desc
     OPTICKS_INTEGRATION_MODE    IntegrationMode  : 1
           OPTICKS_EVENT_MODE          EventMode  : Minimal
           OPTICKS_EVENT_NAME          EventName  : -
                                      DeviceName  : NVIDIA RTX 5000 Ada Generation
         OPTICKS_RUNNING_MODE        RunningMode  : 0
                                RunningModeLabel  : SRM_DEFAULT
            OPTICKS_NUM_EVENT           NumEvent  : 1
           OPTICKS_NUM_PHOTON       NumPhoton(0)  : 0       NumPhoton(1)  : 0      NumPhoton(-1)  : 0
          OPTICKS_NUM_GENSTEP      NumGenstep(0)  : 0      NumGenstep(1)  : 0     NumGenstep(-1)  : 0
         OPTICKS_G4STATE_SPEC        G4StateSpec  : 1000:38
                                G4StateSpecNotes  : 38=2*17+4 is appropriate for MixMaxRng
        OPTICKS_G4STATE_RERUN       G4StateRerun  : -1
           OPTICKS_MAX_CURAND          MaxCurand  : 1000000000000        MaxCurand/M  : 1000000
             OPTICKS_MAX_SLOT            MaxSlot  : 257000000          MaxSlot/M  : 257
          OPTICKS_MAX_GENSTEP         MaxGenstep  : 10000000       MaxGenstep/M  : 10
           OPTICKS_MAX_PHOTON          MaxPhoton  : 1000000000000        MaxPhoton/M  : 1000000
         OPTICKS_MAX_SIMTRACE        MaxSimtrace  : 1000000000000      MaxSimtrace/M  : 1000000
           OPTICKS_MAX_BOUNCE          MaxBounce  : 31
                                  MaxBounceNotes  : NB bounce limit is now separate from the non-PRODUCTION record limit which is inherent from sseq.h sseq::SLOTS 
             OPTICKS_MAX_TIME            MaxTime  : 1e+27
                                    MaxTimeNotes  : NB time limit(ns) can truncate simulation together with bounce limit, default timer limit is so high to be unlimited 
           OPTICKS_MAX_RECORD          MaxRecord  : 0
              OPTICKS_MAX_REC             MaxRec  : 0



Where the truncation is applied
--------------------------------


~/opticks/CSGOptiX/CSGOptiX7.cu::

    477     while( bounce < evt->max_bounce && ctx.p.time < params.max_time )



~/opticks/CSGOptiX/CSGOptiX7.cu::

    432 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    433 {
    ...
    470     sim->generate_photon(ctx.p, rng, gs, photon_idx, genstep_idx );
    471 
    472     int command = START ;
    473     int bounce = 0 ;
    474 #ifndef PRODUCTION
    475     ctx.point(bounce);
    476 #endif
    477     while( bounce < evt->max_bounce && ctx.p.time < params.max_time )
    478     {
    479         float tmin = ( ctx.p.orient_boundary_flag & params.PropagateEpsilon0Mask ) ? params.tmin0 : params.tmin ;
    480 
    481         // intersect query filling (quad2)prd
    482         switch(params.PropagateRefine)
    483         {
    484             case 0u: trace<false>( params.handle, ctx.p.pos, ctx.p.mom, tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
    485             case 1u: trace<true>(  params.handle, ctx.p.pos, ctx.p.mom, tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
    486         }
    487 
    488         if( prd->boundary() == 0xffffu ) break ; // SHOULD ONLY HAPPEN FOR PHOTONS STARTING OUTSIDE WORLD
    ...
    490 
    497         float3* normal = prd->normal();
    498         *normal = normalize(*normal);
    499 
    500 #ifndef PRODUCTION
    501         ctx.trace(bounce);
    502 #endif
    503         command = sim->propagate(bounce, rng, ctx);
    504         bounce++;
    505 #ifndef PRODUCTION
    506         ctx.point(bounce) ;
    507 #endif
    508         if(command == BREAK) break ;
    509     }
    510 #ifndef PRODUCTION
    511     ctx.end();  // write seq, tag, flat
    512 #endif
    513 
    514 
    515     if( evt->photon )
    516     {
    517         evt->photon[idx] = ctx.p ;  // *idx* (not *photon_idx*) as needs to go from zero for photons from a slice of genstep array
    518     }







