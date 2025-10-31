photonlite_hitlite_optimization_eg_for_muon_running
=====================================================

JUNOSW "--pmt-hit-type 2" which is often used to reduce resources for prolific events
such as muon events 


smaller photon/hit for muon events
-------------------------------------

The "junoHit_PMT_muon" is a subset of "junoHit_PMT" so in principal not difficult to support,
and it offers strong potential for optimization::

     44   private:
     45     G4int pmtID;         // the ID of the PMT the photon hits
     46     G4double time;       // the time when photon hitting PMT
     47     G4int iHitCount;     // the Number of p.e. from a Hit
     48     G4float localtheta;  // the photon local theth
     49     G4float localphi;    // the photon local phi


photonlite/hitlite
    suggests squeezing photon into 4x4 bytes instead of the normal 4x4x4 bytes
   
pmtID 
    does not need full 4 bytes, 2 bytes is enough 0xffff = 65535

iHitCount
    count is a result of hit merging (currently done on CPU after downloading)
    but it would be good for photonlite to support future optimization of doing hit 
    merging on GPU : 2 bytes enough again ? 0xffff

time 
    using [ns] units for time, means no loss from using float, as do already anyhow

localtheta/localphi [or lposcost/lposphi]
    could squeeze the angles using domain compression into 16bits each without loosing much


::

    struct sphotonlite
    {
        uint32_t  hitcount_pmtid ;    
        float32_t time ;
        uint32_t  lposcost_lposphi ;  
        uint32_t  flagmask ;
    };


DONE : added lposfphi to squad.h
-------------------------------------






What is needed for future GPU side hit merge ?
-------------------------------------------------

* NOT MUCH, LOOKS DOABLE
* ~/j/CUDA_Thrust_PMTHitMerge/CUDA_Thrust_PMTHitMerge_two_phase_less_memory_cleaner.cu


jcv PMTHitMerger
~~~~~~~~~~~~~~~~~

::

    PMTHitMerger::PMTHitMerger() {
      hitCollection = 0;
      hitCollection_muon = 0;
      m_merge_flag = false;
      m_time_window = 1.0; // 1ns


Simplified CPU impl snippet::  

    bool PMTHitMerger::doMerge(int pmtid, double hittime) 
    {   
        std::map<int, std::vector<junoHit_PMT*> >::iterator pmt = m_PMThit.find(pmtid);
        if (pmt != m_PMThit.end()) 
        {   
            int time1 = static_cast<int>(hittime/m_time_window);
            std::vector<junoHit_PMT*>::iterator it = pmt->second.begin();
            for ( ; it != pmt->second.end(); ++it) 
            {   
                // compare the time
                if (time1 == static_cast<int>((*it)->GetTime()/m_time_window)) 
                {
                    if (hittime < (*it)->GetTime()) (*it)->SetTime(hittime);
                    // === update the count
                   (*it)->SetCount(1 + (*it)->GetCount());
                   return true;
                }
            }
        }
        return false ;
    }



Q: How important is it to know the exact earliest time within the bucket, 
   or it it enough to just know the bucket ?





HMM : sphoton.h/sphotonlite.h : How to switch between the full and lite photon impl ?
---------------------------------------------------------------------------------------

* switching from (sphoton)p to (sphotonlite)l would need too much code change
* adding *l* and using it for persisting instead of *p* looks more plausible
* qsim.h mostly unchanged just ctx.p not persisted, instead persist ctx.l in lite mode


Where does original (sphoton)p come from and where is that connected to the global array ?
-----------------------------------------------------------------------------------

ctx.p comes into existance below::

    372 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    373 {
    374     sevent* evt = params.evt ;
    375     if (launch_idx.x >= evt->num_seed) return;   // was evt->num_photon
    376 
    377     unsigned idx = launch_idx.x ;
    378     unsigned genstep_idx = evt->seed[idx] ;
    379     const quad6& gs = evt->genstep[genstep_idx] ;
    380     // genstep needs the raw index, from zero for each genstep slice sub-launch
    381 
    382     unsigned long long photon_idx = params.photon_slot_offset + idx ;
    383     // 2025/10/20 change from unsigned to avoid clocking photon_idx and duplicating
    384     //
    385     // rng_state access and array recording needs the absolute photon_idx
    386     // for multi-launch and single-launch simulation to match.
    387     // The offset hides the technicality of the multi-launch from output.
    388 
    389     qsim* sim = params.sim ;
    390 
    391 //#define OLD_WITHOUT_SKIPAHEAD 1
    392 #ifdef OLD_WITHOUT_SKIPAHEAD
    393     RNG rng = sim->rngstate[photon_idx] ;
    394 #else
    395     RNG rng ;
    396     sim->rng->init( rng, sim->evt->index, photon_idx );
    397 #endif
    398 
    399     sctx ctx = {} ;
    400     ctx.evt = evt ;   // sevent.h
    401     ctx.prd = prd ;   // squad.h quad2
    402 
    403     ctx.idx = idx ;
    404     ctx.pidx = photon_idx ;
    405 
    406 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    407     ctx.pidx_debug = sim->base->pidx == photon_idx ;
    408 #endif
    409 
    410     sim->generate_photon(ctx.p, rng, gs, photon_idx, genstep_idx );
    411 
    ...
    450 #ifndef PRODUCTION
    451     ctx.end();  // write seq, tag, flat
    452 #endif
    453     evt->photon[idx] = ctx.p ;
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ CONNECT ((sphoton)ctx.p) WITH THE GLOBAL ARRAY VIA ((sevent)params.evt)->photon
    ...
    455 }


sevent.h::

    159 
    160     quad6*   genstep ;    //QEvent::device_alloc_genstep
    161     int*     seed ;
    162     sphoton* hit ;        //QEvent::gatherHit_ allocates event by event depending on num_hit
    163     sphoton* photon ;     //QEvent::device_alloc_photon


lite mode would need some loading/compression into::

         sphotonlite l ;
         l.init(ctx.p);
         evt->photonlite[idx] = l 



lposfphi lposcost
-------------------

::

    (ok) A[blyth@localhost CSGOptiX]$ opticks-f set_lposcost
    ./CSGOptiX/CSGOptiX7.cu:    prd->set_lposcost(__uint_as_float(p6)) ;
    ./CSGOptiX/CSGOptiX7.cu:    prd->set_lposcost(lposcost);
    ./CSGOptiX/CSGOptiX7.cu:        prd->set_lposcost(lposcost);
    ./CSGOptiX/CSGOptiX7.cu:            prd->set_lposcost(lposcost);

    ./sysrap/SOPTIX.cu:    prd->set_lposcost(lposcost);
    ./sysrap/SOPTIX.cu:    prd->set_lposcost(lposcost);

    ./sysrap/SPrd.h:        pr.set_lposcost( lposcost[i] );

    ./sysrap/squad.h:    SQUAD_METHOD void set_lposcost(float lpc);
    ./sysrap/squad.h:SQUAD_METHOD void           quad2::set_lposcost(float lpc)   { q1.f.x = lpc ; }
    (ok) A[blyth@localhost opticks]$ 







GPU side collection of lposcost+lposfphi or similar via squad.h (quad)prd
-----------------------------------------------------------------------------

::


    (ok) A[blyth@localhost opticks]$ opticks-f set_lposcost
    ./CSGOptiX/CSGOptiX7.cu:    prd->set_lposcost(__uint_as_float(p6)) ;   // trace.not-WITH_PRD
    ./CSGOptiX/CSGOptiX7.cu:    prd->set_lposcost(lposcost);   // __miss__ms.WITH_PRD
    ./CSGOptiX/CSGOptiX7.cu:        prd->set_lposcost(lposcost);   // __closesthit__ch.WITH_PRD.TRIANGLE
    ./CSGOptiX/CSGOptiX7.cu:            prd->set_lposcost(lposcost);    // __intersection__is.WITH_PRD.CUSTOM

    ./sysrap/SOPTIX.cu:    prd->set_lposcost(lposcost);  // __miss__ms.TRIANGLE
    ./sysrap/SOPTIX.cu:    prd->set_lposcost(lposcost);   // __closesthit__ch.TRIANGLE 

    ./sysrap/SPrd.h:        pr.set_lposcost( lposcost[i] );    // SPrd::init_prd testing dummy PRD
    ./sysrap/squad.h:    SQUAD_METHOD void set_lposcost(float lpc);
    ./sysrap/squad.h:SQUAD_METHOD void           quad2::set_lposcost(float lpc)   { q1.f.x = lpc ; }











    (ok) A[blyth@localhost opticks]$ opticks-f set_iindex
    ./CSGOptiX/CSGOptiX7.cu:    prd->set_iindex(p7) ;
    ./CSGOptiX/CSGOptiX7.cu:        prd->set_iindex(   iindex ) ;
    ./CSGOptiX/CSGOptiX7.cu:        prd->set_iindex(   iindex ) ;

    ./sysrap/SOPTIX.cu:    prd->set_iindex(   iindex ) ;

    ./sysrap/squad.h:    SQUAD_METHOD void set_iindex(  unsigned ii);
    ./sysrap/squad.h:SQUAD_METHOD void           quad2::set_iindex(  unsigned ii) { q1.u.y = ii ;  }

    ./sysrap/sphoton.h:    SPHOTON_METHOD void set_iindex(unsigned ii ){ orient_iindex = ( orient_iindex & 0x80000000u ) | ( 0x7fffffffu & ii ) ; }   // retain bit 31 asis
    ./u4/U4Recorder.cc:    current_photon.set_iindex( iindex );
    (ok) A[blyth@localhost opticks]$ 








HMM : selecting hitlite from photonlite array requires flagmask ?
-------------------------------------------------------------------

* Q: how many bits does flagmask need ? 
* A: From OpticksPhoton.h all important flags that are OR-ed into the flagmask are within the first 16 bits
* A: BUT, hardcoding hitmask might allow a 1 bit "is_hit" instead ?

::

    973 struct sphoton_selector
    974 {
    975     unsigned hitmask ;
    976     sphoton_selector(unsigned hitmask_) : hitmask(hitmask_) {};
    977     SPHOTON_METHOD bool operator() (const sphoton& p) const { return ( p.flagmask  & hitmask ) == hitmask  ; }   // require all bits of the mask to be set
    978     SPHOTON_METHOD bool operator() (const sphoton* p) const { return ( p->flagmask & hitmask ) == hitmask  ; }   // require all bits of the mask to be set
    979 };




HMM : how to get localtheta/localphi
---------------------------------------------------------

::


    086 /**
     87 squad.h/quad2
     88 --------------
     89 
     90 ::
     91 
     92     +------------+------------+------------+---------------+
     93     | f:normal_x | f:normal_y | f:normal_z | f:distance    |
     94     +------------+------------+------------+---------------+
     95     | f:lposcost | u:iindex   | u:identity | u:boundary    |
     96     +------------+------------+------------+---------------+
     97 
     98 WIP: HMM? pack boundary_iindex to spare 4 bytes for f:lposphi ?
     99 would avoid awkward transform lookups by using OptiX local
    100 coordinates directly
    101 
    102 
    103 f:lposcost
    104     Local position cos(theta) of intersect,
    105     canonically calculated in CSGOptiX7.cu:__intersection__is
    106     normalize_z(ray_origin + isect.w*ray_direction )
    107     where normalize_z is v.z/sqrtf(dot(v, v))
    108 
    109     This is kinda imagining a sphere thru the intersection point
    110     which is likely onto an ellipsoid or a box or anything
    111     to provide a standard way of giving a z-polar measure.
    112 





