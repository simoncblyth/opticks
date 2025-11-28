how_to_debug_no_hits
=====================


Run small simulation with debug array saving
----------------------------------------------

Opticks hits are a subselection of the photons
which have flagmask that matches the SEventConfig::HitMask.

SEventConfig::HitMask (config by envvar OPTICKS_HIT_MASK, a comma delimited string
that determines which subset of photons are downloaded into the "hit" array)
Default is "SD".  Any flag combination can be used, the typical ones to use are::

    SD : SURFACE_DETECT
    EC : EFFICIENCY_COLLECT
    EX : EFFICIENCY_CULL


To debug lack of hits the first thing to do is run a one event simulation
with a few million photons maximum with debug array saving enabled::

    export OPTICKS_EVENT_MODE=DebugLite  # enable saving of debug arrays
    export SEvt=INFO                     # enable SEvt logging to see output directory

Arrays saved will include:

photon.npy
    final photons
    shape (num_photon, 4, 4)

record.npy
    step-by-step photon histories for first 32 step points of the photon
    shape (num_photon, 32, 4, 4)

    THIS IS MOST USEFUL FOR DEBUG : AS GIVES PHOTON AT EVERY STEP POINT

hit.npy
    subset of the photon.npy (when num_hit zero may be skipped)
    shape (num_hit, 4, 4)

seq.npy
    photon history large integers


The method SEventConfig::Initialize_Comp_Simulate_ sets up the arrays that are
gathered from GPU and saved to file with the "DebugLite" and other EventMode


Load the output folder into ipython::

    from opticks.ana.fold import Fold
    f = Fold.Load(symbol="f")
    print(repr(f))

    print(f.record)



The opticks env includes a bash function and scripts to do that::

    (ok) A[blyth@localhost A000]$ f
    Python 3.13.2 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:02) [GCC 11.2.0]
    Type 'copyright', 'credits' or 'license' for more information
    IPython 9.1.0 -- An enhanced Interactive Python. Type '?' for help.
    Tip: `?` alone on a line will brings up IPython's help
    pvplt MODE:2
    f

    CMDLINE:/data1/blyth/local/opticks_Debug/bin/f.py
    f.base:.

      : f.genstep                                          :           (90, 6, 4) : 2 days, 5:05:11.757160
      : f.seq                                              :         (9195, 2, 2) : 2 days, 5:05:11.757160
      : f.photonlite                                       :            (9195, 4) : 2 days, 5:05:11.757160
      : f.hitlite                                          :            (1614, 4) : 13 days, 23:15:16.685795
      : f.seqnib                                           :              (9195,) : 2 days, 5:05:11.756160
      : f.seqnib_table                                     :              (33, 1) : 2 days, 5:05:11.756160
      : f.NPFold_index                                     :                 (6,) : 2 days, 5:05:11.756160
      : f.NPFold_meta                                      :                   25 : 2 days, 5:05:11.756160
      : f.NPFold_names                                     :                 (0,) : 2 days, 5:05:11.756160
      : f.sframe                                           :            (4, 4, 4) : 2 days, 5:05:11.756160
      : f.sframe_meta                                      :                    7 : 2 days, 5:05:11.756160
      : f.hitlitemerged                                    :            (1611, 4) : 2 days, 5:05:11.757160

     min_stamp : 2025-11-14 22:39:23.591071
     max_stamp : 2025-11-26 16:49:28.520706
     dif_stamp : 11 days, 18:10:04.929635
     age_stamp : 2 days, 5:05:11.756160

    In [1]:


From v0.5.7 the shortcut bash function "f" can be used to run the installed scripts f.sh, f.py::

    (ok) A[blyth@localhost A000]$ t f
    f ()
    {
        f.sh $*;
        : opticks.bash load .npy from PWD or argument fold
    }


For Opticks prior to v0.5.6 those scripts are not installed, but you can use
the scripts from source::

    ~/opticks/bin/f.sh



To understand the meanings of array entries see sysrap/sphoton.h::


+----+----------------+----------------+----------------+----------------+------------------------------+
| q  |      x         |      y         |     z          |      w         |  notes                       |
+====+================+================+================+================+==============================+
|    |  pos.x         |  pos.y         |  pos.z         |  time          |                              |
| q0 |                |                |                |                |                              |
|    |                |                |                |                |                              |
+----+----------------+----------------+----------------+----------------+------------------------------+
|    |  mom.x         |  mom.y         | mom.z          |  orient_iindex | orient:1bit iindex:31 bit    |
| q1 |                |                |                | (unsigned)     |                              |
|    |                |                |                | (1,3)          |                              |
+----+----------------+----------------+----------------+----------------+------------------------------+
|    |  pol.x         |  pol.y         |  pol.z         |  wavelength    |                              |
| q2 |                |                |                |                |                              |
|    |                |                |                |                |                              |
+----+----------------+----------------+----------------+----------------+------------------------------+
|    | boundary_flag  |  identity      |  index         |  flagmask      |  (unsigned)                  |
| q3 |                |                |                |                |                              |
|    | (3,0)          |  (3,1)         |  (3,2)         |  (3,3)         | (3,2) formerly orient_idx    |
|    |                | hi8 ext idx    |                |                |                              |
+----+----------------+----------------+----------------+----------------+------------------------------+


Note, I am currently rearranging orient_iindex and boundary_flag to squeeze
in 16 bits of hitcount needed for GPU (CUDA thrust implemented) hit (identity, time_bucket)
merging.


Things to check:

1. are the positions and times in the record array as expected ?




Sensor geometry setup : g4cx/G4CXOpticks.hh u4/U4SensorIdentifier.h u4/U4SensorIdentifierDefault.h
-----------------------------------------------------------------------------------------------------

*G4CXOpticks::SetSensorIdentifier* is a static method that
optionally allows the default technique to find sensor volumes
to be overridded.

::

    struct G4CX_API G4CXOpticks
    {
        static const plog::Severity LEVEL ;
        static U4SensorIdentifier* SensorIdentifier ;
        static void SetSensorIdentifier( U4SensorIdentifier* sid );



u4/U4SensorIdentifierDefault.h
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    #pragma once
    /**
    u4/U4SensorIdentifierDefault.h
    ================================

    This fulfils U4SensorIdentifier protocol, it is used
    to identify sensors in the geometry.  To override this
    implementation use G4CXOpticks::SetSensorIdentifier.


    **/

    #include <vector>
    #include <iostream>
    #include <map>

    #include "G4PVPlacement.hh"

    #include "sstr.h"
    #include "ssys.h"

    #include "U4SensorIdentifier.h"
    #include "U4Boundary.h"


    struct U4SensorIdentifierDefault : public U4SensorIdentifier
    {
        static std::vector<std::string>* GLOBAL_SENSOR_BOUNDARY_LIST ;

        void setLevel(int _level);
        int getGlobalIdentity(const G4VPhysicalVolume* pv, const G4VPhysicalVolume* ppv ) ;
        int getInstanceIdentity(const G4VPhysicalVolume* instance_outer_pv ) const ;
        static void FindSD_r( std::vector<const G4VPhysicalVolume*>& sdpv , const G4VPhysicalVolume* pv, int depth );
        static bool IsInterestingCopyNo( int copyno );

        int level = 0 ;
        std::vector<int> count_global_sensor_boundary ;

    };





u4/U4SensorIdentifier.h
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    #pragma once
    /**
    U4SensorIdentifier.h
    ======================

    Pure virtual protocol base used to interface detector
    specific characteristics of sensors with Opticks.

    getGlobalIdentity
        method is called on ALL of the remainder non-instanced nodes,
        it is expected to return an integer value uniquely identifiying
        any sensors. For non-sensor volumes an integer of -1 should be returned

        AS JUNO HAS NO global sensors this is untested.

    getInstanceIdentity
        method is called on ONLY the outer volume of every factorized
        instance during geometry translation
        If the subtree of volumes within the outer volume provided
        contains a sensor then this method is expected to return an
        integer value that uniquely identifies the sensor.
        If the subtree does not contain a sensor, then -1 should be returned.

        CURRENTLY HAVING MORE THAN ONE ACTUAL SENSOR PER INSTANCE IS NOT SUPPORTED

        An ACTUAL sensor is one that would yield hits with Geant4 : ie it
        must have an EFFICIENCY property with non-zero values and have
        G4LogicalVolume::SetSensitiveDetector associated.

    U4SensorIdentifierDefault.h provided the default implementation.
    To override this default use G4CXOpticks::SetSensorIdentifier

    **/
    class G4VPhysicalVolume ;

    struct U4SensorIdentifier
    {
        virtual void setLevel(int _level) = 0 ;
        virtual int getGlobalIdentity(const G4VPhysicalVolume* node_pv, const G4VPhysicalVolume* node_ppv ) = 0 ;
        virtual int getInstanceIdentity(const G4VPhysicalVolume* instance_outer_pv ) const = 0 ;
    };









PIDX dumping : in Debug builds enables dumping from the kernel
------------------------------------------------------------------

::

    PIDX=0 ./script_invoking_your_simulation.sh   # dump from simulation of 0th photon


To follow the simulation and understand what is dumped start from CSGOptiX/CSGOptiX7.cu::

    373 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    374 {
    375     sevent* evt = params.evt ;
    376     if (launch_idx.x >= evt->num_seed) return;   // was evt->num_photon
    377
    378     unsigned idx = launch_idx.x ;
    379     unsigned genstep_idx = evt->seed[idx] ;
    380     const quad6& gs = evt->genstep[genstep_idx] ;
    381     // genstep needs the raw index, from zero for each genstep slice sub-launch
    382
    383     unsigned long long photon_idx = params.photon_slot_offset + idx ;
    384     // 2025/10/20 change from unsigned to avoid clocking photon_idx and duplicating
    385     //
    386     // rng_state access and array recording needs the absolute photon_idx
    387     // for multi-launch and single-launch simulation to match.
    388     // The offset hides the technicality of the multi-launch from output.
    389
    390     qsim* sim = params.sim ;


    414     int bounce = 0 ;
    415 #ifndef PRODUCTION
    416     ctx.point(bounce);
    417 #endif
    418     while( bounce < evt->max_bounce && ctx.p.time < params.max_time )
    419     {
    ...
    430         float tmin = ( ctx.p.orient_boundary_flag & params.PropagateEpsilon0Mask ) ? params.tmin0 : params.tmin ;
    431
    432         // intersect query filling (quad2)prd
    433         switch(params.PropagateRefine)
    434         {
    435             case 0u: trace<false>( params.handle, ctx.p.pos, ctx.p.mom, tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
    436             case 1u: trace<true>(  params.handle, ctx.p.pos, ctx.p.mom, tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
    437         }
    438
    439         if( prd->boundary() == 0xffffu ) break ; // SHOULD ONLY HAPPEN FOR PHOTONS STARTING OUTSIDE WORLD
    ...
    448         float3* normal = prd->normal();
    449         *normal = normalize(*normal);
    450
    451 #ifndef PRODUCTION
    452         ctx.trace(bounce);
    453 #endif
    454         command = sim->propagate(bounce, rng, ctx);
    455         bounce++;
    456 #ifndef PRODUCTION
    457         ctx.point(bounce) ;
    458 #endif
    459         if(command == BREAK) break ;
    460     }
    461 #ifndef PRODUCTION
    462     ctx.end();  // write seq, tag, flat
    463 #endif
    464
    465
    466     if( evt->photon )
    467     {
    468         evt->photon[idx] = ctx.p ;  // *idx* (not *photon_idx*) as needs to go from zero for photons from a slice of genstep array
    469     }



Then proceed to qudarap/qsim.h qsim::propagate::

    2218 inline QSIM_METHOD int qsim::propagate(const int bounce, RNG& rng, sctx& ctx )  // ::simulate
    2219 {
    2220     const unsigned boundary = ctx.prd->boundary() ;
    2221     const unsigned identity = ctx.prd->identity() ; // sensor_identifier+1, 0:not-a-sensor
    2222     const unsigned iindex = ctx.prd->iindex() ;
    2223     const float lposcost = ctx.prd->lposcost() ;  // local frame intersect position cosine theta
    2224
    2225     const float3* normal = ctx.prd->normal();
    2226     float cosTheta = dot(ctx.p.mom, *normal ) ;
    2227


