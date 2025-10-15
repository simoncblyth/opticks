#pragma once
/**
sevent.h : host/device communication instance : holder of array pointers
===========================================================================

Instantiation of sevent is done by SEvt::SEvt
and the instance is subsequently uploaded to the device after
device buffer allocations hence the *sevent* instance
provides event config and device buffer pointers
both on device and host.

Note that *num_seed* and *num_photon* will be equal in
normal operation which uses QEvent::setGensteps.
However for clarity separate fields are used to
distinguish photon test running that directly uses
QEvent::setNumPhoton

Users of sevent.h
-------------------

qudarap/QU.cc
    template instanciations of QU::UploadArray QU::device_alloc QU::copy_host_to_device

CSGOptiX/CSGOptiX7.cu
    simulate : reads params.evt, sets sctx.evt
    reads evt.seed evt.genstep evt.max_bounce
    writes evt.photon

    simtrace : reads params.evt, evt.num_simtrace, calls evt::add_simtrace

sysrap/sctx.h
    *evt* member of device context *sctx*
    sctx::point writes evt.record evt.rec
    sctx::trace writes evt.prd
    sctx::end writes evt.seq evt.tag evt.flat

sysrap/SEvt.hh sysrap/SEvt.cc
    SEvt::SEvt instanciates hostside sevent evt member
    SEvt::init/sevent::init sets maxima

    SEvt::addGenstep/SEvt::setNumPhoton
    sets evt.num_photon/seq/tag/flat/record/rec/prd depending on the evt.max

qudarap/QEvent.hh qudarap/QEvent.cc
    **fundamentally integrated with sevent.h**

    QEvent::QEvent

    * gets sevent:evt pointer from SEvt
    * allocates space on device for d_evt

    QEvent::setNumPhoton/QEvent::uploadEvt

    * updates device d_evt by copying from evt


qudarap/qsim.h
    qsim::mock_propagate mocks the CSGOptiX7.cu:simulate
    qsim::generate_photon uses evt to read input photons from buffer

qudarap/QSim.cu
    used from several tests for evt.num, evt.seed, evt.genstep

qudarap/QEvent.cu
    sevent.h used for tests of seeding and genstep photon counting

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SEVENT_METHOD __device__ __forceinline__
#else
#    define SEVENT_METHOD inline
#endif

struct float4 ;
struct float2 ;
struct quad4 ;
struct quad6 ;

//#if !defined(PRODUCTION)
struct srec ;
struct sseq ;
struct stag ;
struct sflat ;
//#endif


struct sphoton ;

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <sstream>
#include <iomanip>
#include "SEventConfig.hh"
#include "NP.hh"
#endif

struct sevent
{
    static constexpr int64_t M = 1000000 ;
    static constexpr int64_t genstep_itemsize = 6*4 ;
    static constexpr int64_t genstep_numphoton_offset = 3 ;

    static constexpr float w_lo = 60.f ;
    static constexpr float w_hi = 820.f ;  // these need to match sdomain.h
    static constexpr float w_center = (w_lo+w_hi)/2.f ; // convert wavelength range into center-extent form
    static constexpr float w_extent = (w_hi-w_lo)/2.f ;

    float4 center_extent ;
    float2 time_domain ;
    float2 wavelength_domain ;

    // TODO: make all these below unsigned
    // sevent::init sets these max using values from SEventConfig
    int64_t   max_curand ;
    int64_t   max_slot   ;
    int64_t   max_genstep ;  // eg:      100,000
    int64_t   max_photon  ;  // eg:  100,000,000
    int64_t   max_simtrace ; // eg: 100,000,000

    int   max_bounce  ; // eg: 0:32  (not including 32)
    int   max_record  ; // eg: 10  full step record
    int   max_rec     ; // eg: 10  compressed step record
    int   max_seq     ; // now restricted to 0 or 1 only
    int   max_prd     ; // eg: 16
    int   max_tag     ; // 0 or 1 only
    int   max_flat    ; // 0 or 1 only
    int   max_aux     ; // eg 0, 16, 32 : when greater than zero typically follows max_record
    int   max_sup     ;

    int   index ;

    //[ counts and pointers, zeroed by sevent::zero
    //  only first 4 are always in use, the last 7 are only relevant whilst debugging
    //
    int64_t      num_curand ;
    int64_t      num_genstep ;
    int64_t      num_seed ;
    int64_t      num_hit ;    // set by QEvent::gatherHit using SU::count_if_sphoton
    int64_t      num_photon ;

    int64_t      num_record ;
    int64_t      num_rec ;
    int64_t      num_seq ;
    int64_t      num_prd ;
    int64_t      num_tag ;
    int64_t      num_flat ;
    int64_t      num_simtrace ;
    int64_t      num_aux ;
    int64_t      num_sup ;
    // TODO: make all these above unsigned

    // With QEvent device running the below are pointers to device buffers.
    // Most are allocated ONCE ONLY by QEvent::device_alloc_genstep/photon/simtrace
    // sized by configured maxima.
    // Only the hit buffer is allocated more dynamically depending on num_hit.
    // TODO: check for leaking of hit buffers

    quad6*   genstep ;    //QEvent::device_alloc_genstep
    int*     seed ;
    sphoton* hit ;        //QEvent::gatherHit_ allocates event by event depending on num_hit
    sphoton* photon ;     //QEvent::device_alloc_photon


    sphoton* record ;
    srec*    rec ;
    sseq*    seq ;

    quad2*   prd ;
    stag*    tag ;
    sflat*   flat ;
    quad4*   simtrace ;   //QEvent::device_alloc_simtrace
    quad4*   aux ;
    quad6*   sup ;

    //] counts and pointers


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SEVENT_METHOD void init();
    SEVENT_METHOD void init_domain(float extent, float time_max);

    SEVENT_METHOD std::string descMax() const ;
    SEVENT_METHOD std::string descNum() const ;
    SEVENT_METHOD std::string descBuf() const ;
    SEVENT_METHOD std::string desc() const ;

    SEVENT_METHOD void get_domain(quad4& dom) const ;
    SEVENT_METHOD void get_config(quad4& cfg) const ;
    SEVENT_METHOD void get_meta(std::string& meta) const ;

    SEVENT_METHOD void zero();
#endif

#ifndef PRODUCTION
    SEVENT_METHOD void add_rec( srec& r, unsigned idx, unsigned bounce, const sphoton& p);
#endif
    SEVENT_METHOD void add_simtrace( unsigned idx, const quad4& p, const quad2* prd, float tmin );

};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

SEVENT_METHOD void sevent::init()
{
    // int64_t
    max_curand   = SEventConfig::MaxCurand() ;
    max_slot     = SEventConfig::MaxSlot() ;
    max_genstep  = SEventConfig::MaxGenstep() ;
    max_photon   = SEventConfig::MaxPhoton()  ;
    max_simtrace = SEventConfig::MaxSimtrace()  ;

    // int
    max_bounce   = SEventConfig::MaxBounce()  ;
    max_record   = SEventConfig::MaxRecord()  ;  // full step record
    max_rec      = SEventConfig::MaxRec()  ;     // compressed step record
    max_seq      = SEventConfig::MaxSeq()  ;     // seqhis
    max_prd      = SEventConfig::MaxPrd()  ;
    max_tag      = SEventConfig::MaxTag()  ;
    max_flat     = SEventConfig::MaxFlat()  ;
    max_aux      = SEventConfig::MaxAux()  ;
    max_sup      = SEventConfig::MaxSup()  ;


    zero();  // pointers and counts

    float max_extent_domain = SEventConfig::MaxExtentDomain() ;
    float max_time_domain = SEventConfig::MaxTimeDomain() ;

    init_domain( max_extent_domain, max_time_domain );
}

/**
sevent::init_domain
--------------------

Domains are mainly(only?) used for the compressed photon "rec",
but now are making more use of the full photon "record"
as thats considerably simpler that "rec".

**/
SEVENT_METHOD void sevent::init_domain(float max_extent_domain, float max_time_domain)
{
    center_extent.x = 0.f ;
    center_extent.y = 0.f ;
    center_extent.z = 0.f ;
    center_extent.w = max_extent_domain ;

    time_domain.x = 0.f ;
    time_domain.y = max_time_domain ;

    wavelength_domain.x = w_center ;  // TODO: try to make this constexpr
    wavelength_domain.y = w_extent ;
}

SEVENT_METHOD std::string sevent::descMax() const
{
    int w = 5 ;
    std::stringstream ss ;
    ss
        << "sevent::descMax    " << std::endl
        << " evt.max_curand    " << std::setw(w) << max_curand   << std::endl
        << " evt.max_slot      " << std::setw(w) << max_slot     << std::endl
        << " evt.max_genstep   " << std::setw(w) << max_genstep  << std::endl
        << " evt.max_photon    " << std::setw(w) << max_photon   << std::endl
        << " evt.max_simtrace  " << std::setw(w) << max_simtrace << std::endl
        << " evt.max_bounce    " << std::setw(w) << max_bounce   << std::endl
        << " evt.max_record    " << std::setw(w) << max_record   << std::endl
        << " evt.max_rec       " << std::setw(w) << max_rec      << std::endl
        << " evt.max_aux       " << std::setw(w) << max_aux      << std::endl
        << " evt.max_sup       " << std::setw(w) << max_sup      << std::endl
        << " evt.max_seq       " << std::setw(w) << max_seq      << std::endl
        << " evt.max_prd       " << std::setw(w) << max_prd      << std::endl
        << " evt.max_tag       " << std::setw(w) << max_tag      << std::endl
        << " evt.max_flat      " << std::setw(w) << max_flat     << std::endl
        ;

    std::string s = ss.str();
    return s ;
}

SEVENT_METHOD std::string sevent::descNum() const
{
    int w = 5 ;
    std::stringstream ss ;
    ss
        << " sevent::descNum  "  << std::endl
        << " evt.num_curand   "  << std::setw(w) << num_curand   << std::endl
        << " evt.num_genstep  "  << std::setw(w) << num_genstep  << std::endl
        << " evt.num_seed     "  << std::setw(w) << num_seed     << std::endl
        << " evt.num_photon   "  << std::setw(w) << num_photon   << std::endl
        << " evt.num_record   "  << std::setw(w) << num_record   << std::endl
        << " evt.num_rec      "  << std::setw(w) << num_rec      << std::endl
        << " evt.num_aux      "  << std::setw(w) << num_aux      << std::endl
        << " evt.num_sup      "  << std::setw(w) << num_sup      << std::endl
        << " evt.num_seq      "  << std::setw(w) << num_seq      << std::endl
        << " evt.num_hit      "  << std::setw(w) << num_hit      << std::endl
        << " evt.num_simtrace "  << std::setw(w) << num_simtrace << std::endl
        << " evt.num_prd      "  << std::setw(w) << num_prd      << std::endl
        << " evt.num_tag      "  << std::setw(w) << num_tag      << std::endl
        << " evt.num_flat     "  << std::setw(w) << num_flat     << std::endl
        ;
    std::string s = ss.str();
    return s ;
}

SEVENT_METHOD std::string sevent::descBuf() const
{
    int w = 5 ;
    std::stringstream ss ;
    ss
        << std::setw(20) << " sevent::descBuf "
        << std::endl
        << std::setw(20) << " evt.genstep     " << std::setw(w) << ( genstep ? "Y" : "N" ) << " " << std::setw(20) << genstep
        << std::setw(20) << " num_genstep "     << std::setw(7) << num_genstep
        << std::setw(20) << " max_genstep "     << std::setw(7) << max_genstep
        << std::endl
        << std::setw(20) << " evt.seed   "      << std::setw(w) << ( seed    ? "Y" : "N" ) << " " << std::setw(20) << seed
        << std::setw(20) << " num_seed "        << std::setw(7) << num_seed
        << std::setw(20) << " max_photon "      << std::setw(7) << max_photon
        << std::endl
        << std::setw(20) << " evt.photon "      << std::setw(w) << ( photon  ? "Y" : "N" ) << " " << std::setw(20) << photon
        << std::setw(20) << " num_photon "      << std::setw(7) << num_photon
        << std::setw(20) << " max_photon "      << std::setw(7) << max_photon
        << std::endl
        << std::setw(20) << " evt.record "      << std::setw(w) << ( record  ? "Y" : "N" ) << " " << std::setw(20) << record
        << std::setw(20) << " num_record "      << std::setw(7) << num_record
        << std::setw(20) << " max_record "      << std::setw(7) << max_record
        << std::endl
        << std::setw(20) << " evt.rec "         << std::setw(w) << ( rec  ? "Y" : "N" ) << " " << std::setw(20) << rec
        << std::setw(20) << " num_rec "         << std::setw(7) << num_rec
        << std::setw(20) << " max_rec "         << std::setw(7) << max_rec
        << std::endl
        << std::setw(20) << " evt.aux "         << std::setw(w) << ( aux  ? "Y" : "N" ) << " " << std::setw(20) << aux
        << std::setw(20) << " num_aux "         << std::setw(7) << num_aux
        << std::setw(20) << " max_aux "         << std::setw(7) << max_aux
        << std::endl
        << std::setw(20) << " evt.sup "         << std::setw(w) << ( sup  ? "Y" : "N" ) << " " << std::setw(20) << sup
        << std::setw(20) << " num_sup "         << std::setw(7) << num_sup
        << std::setw(20) << " max_sup "         << std::setw(7) << max_sup
        << std::endl
        << std::setw(20) << " evt.seq "         << std::setw(w) << ( seq     ? "Y" : "N" ) << " " << std::setw(20) << seq
        << std::setw(20) << " num_seq "         << std::setw(7) << num_seq
        << std::setw(20) << " max_seq "         << std::setw(7) << max_seq
        << std::endl
        << std::setw(20) << " evt.hit "         << std::setw(w) << ( hit     ? "Y" : "N" ) << " " << std::setw(20) << hit
        << std::setw(20) << " num_hit "         << std::setw(7) << num_hit
        << std::setw(20) << " max_photon "      << std::setw(7) << max_photon
        << std::endl
        << std::setw(20) << " evt.simtrace "    << std::setw(w) << ( simtrace  ? "Y" : "N" ) << " " << std::setw(20) << simtrace
        << std::setw(20) << " num_simtrace "    << std::setw(7) << num_simtrace
        << std::setw(20) << " max_simtrace "    << std::setw(7) << max_simtrace
        << std::endl
        << std::setw(20) << " evt.prd "         << std::setw(w) << ( prd  ? "Y" : "N" ) << " " << std::setw(20) << prd
        << std::setw(20) << " num_prd "         << std::setw(7) << num_prd
        << std::setw(20) << " max_prd "         << std::setw(7) << max_prd
        << std::endl
        << std::setw(20) << " evt.tag "         << std::setw(w) << ( tag  ? "Y" : "N" ) << " " << std::setw(20) << tag
        << std::setw(20) << " num_tag "         << std::setw(7) << num_tag
        << std::setw(20) << " max_tag "         << std::setw(7) << max_tag
        << std::endl
        << std::setw(20) << " evt.flat "         << std::setw(w) << ( flat  ? "Y" : "N" ) << " " << std::setw(20) << flat
        << std::setw(20) << " num_flat "         << std::setw(7) << num_flat
        << std::setw(20) << " max_flat "         << std::setw(7) << max_flat
        << std::endl
        ;
    std::string s = ss.str();
    return s ;
}


SEVENT_METHOD std::string sevent::desc() const
{
    std::stringstream ss ;
    ss << descMax() << std::endl ;
    ss << descBuf() << std::endl ;
    ss << descNum() << std::endl ;
    std::string s = ss.str();
    return s ;
}

/**
sevent::get_domain
-------------------

HMM: could also use metadata (key, value) pairs on the domain NP array

ana/evt.py::

    2052         post_center, post_extent = self.post_center_extent()  # center and extent are quads, created by combining position and time domain ce
    2053         p = self.rx[:,recs,0].astype(np.float32)*post_extent/32767.0 + post_center

**/


SEVENT_METHOD void sevent::get_domain( quad4& dom ) const
{
   float4 post_center = make_float4( center_extent.x, center_extent.y, center_extent.z, time_domain.x );
   float4 post_extent = make_float4( center_extent.w, center_extent.w, center_extent.w, time_domain.y );
   float4 polw_center = make_float4( 0.f, 0.f, 0.f, wavelength_domain.x  );
   float4 polw_extent = make_float4( 1.f, 1.f, 1.f, wavelength_domain.y );

   dom.q0.f = post_center ;
   dom.q1.f = post_extent ;
   dom.q2.f = polw_center ;
   dom.q3.f = polw_extent ;

   // xyz duplication allows position and time to be decompressed together
   // also polarization and wavelength can be decompressed together using same trick
}


SEVENT_METHOD void sevent::get_config( quad4& cfg ) const
{
   cfg.q0.u.x = max_genstep ;
   cfg.q0.u.y = max_photon ;
   cfg.q0.u.z = max_bounce ;
   cfg.q0.u.w = 0 ;

   cfg.q1.u.x = max_record ;
   cfg.q1.u.y = max_rec ;
   cfg.q1.u.z = max_seq ;
   cfg.q1.u.w = 0 ;

   cfg.q2.u.x = num_genstep ;
   cfg.q2.u.y = num_seed ;
   cfg.q2.u.z = num_photon ;
   cfg.q2.u.w = num_hit ;

   cfg.q3.u.x = num_record ;
   cfg.q3.u.y = num_rec ;
   cfg.q3.u.z = num_seq ;
   cfg.q3.u.w = 0 ;
}


SEVENT_METHOD void sevent::get_meta(std::string& meta) const
{
    NP::SetMeta<int64_t>(meta,"evt.max_curand", max_curand);
    NP::SetMeta<int64_t>(meta,"evt.max_slot", max_slot);
    NP::SetMeta<int64_t>(meta,"evt.max_photon", max_photon);
    NP::SetMeta<int64_t>(meta,"evt.num_photon", num_photon);

    NP::SetMeta<int64_t>(meta,"evt.max_curand/M", max_curand/M);
    NP::SetMeta<int64_t>(meta,"evt.max_slot/M", max_slot/M);
    NP::SetMeta<int64_t>(meta,"evt.max_photon/M", max_photon/M);
    NP::SetMeta<int64_t>(meta,"evt.num_photon/M", num_photon/M);

    NP::SetMeta<int64_t>(meta,"evt.max_slot*evt.max_record", max_slot*max_record);
    NP::SetMeta<int64_t>(meta,"evt.max_slot*evt.max_record/M", max_slot*max_record/M);

    NP::SetMeta<int64_t>(meta,"evt.max_record", max_record);
    NP::SetMeta<int64_t>(meta,"evt.max_record/M", max_record/M);
    NP::SetMeta<int64_t>(meta,"evt.max_rec", max_rec);
    NP::SetMeta<int64_t>(meta,"evt.max_seq", max_seq);
    NP::SetMeta<int64_t>(meta,"evt.max_prd", max_prd);
    NP::SetMeta<int64_t>(meta,"evt.max_tag", max_tag);
    NP::SetMeta<int64_t>(meta,"evt.max_flat", max_flat);

    NP::SetMeta<int64_t>(meta,"evt.num_record", num_record);
    NP::SetMeta<int64_t>(meta,"evt.num_rec", num_rec);
    NP::SetMeta<int64_t>(meta,"evt.num_seq", num_seq);
    NP::SetMeta<int64_t>(meta,"evt.num_prd", num_prd);
    NP::SetMeta<int64_t>(meta,"evt.num_tag", num_tag);
    NP::SetMeta<int64_t>(meta,"evt.num_flat", num_flat);
}



/**
sevent::zero
--------------

CAUTION with QEvent device side running the pointers
are to device buffers that are allocated once based on
configured maxima and then reused for each launch

**/
SEVENT_METHOD void sevent::zero()
{
    index = 0 ;

    //num_slot = 0 ;
    num_curand = 0 ;
    num_genstep = 0 ;
    num_seed  = 0 ;
    num_hit = 0 ;
    num_photon = 0 ;

    num_record = 0 ;
    num_rec = 0 ;
    num_aux = 0 ;
    num_sup = 0 ;
    num_seq = 0 ;
    num_prd = 0 ;
    num_tag = 0 ;
    num_flat = 0 ;
    num_simtrace = 0 ;


    genstep = nullptr ;
    seed = nullptr ;
    hit = nullptr ;
    photon = nullptr ;

    record = nullptr ;
    rec = nullptr ;
    aux = nullptr ;
    sup = nullptr ;
    seq = nullptr ;
    prd = nullptr ;
    tag = nullptr ;
    flat = nullptr ;
    simtrace = nullptr ;
}
#endif     // ends host only block


#ifndef PRODUCTION
/**
sevent::add_rec
----------------

Populates compressed "srec& r" from "const sphoton& p" using the domains
and copies into evt->rec array using the (idx,bounce) slot.

Note no flags. Have moved to using the full uncompressed "record"
for simplicity instead of this domain compressed one.

**/
SEVENT_METHOD void  sevent::add_rec( srec& r, unsigned idx, unsigned bounce, const sphoton& p )
{
    r.set_position(     p.pos,  center_extent );
    r.set_time(         p.time, time_domain );
    r.set_polarization( p.pol );
    r.set_wavelength(   p.wavelength, wavelength_domain );

    rec[max_rec*idx+bounce] = r ;
}
#endif

/**
sevent::add_simtrace : fills simtrace[idx] with values from p, prd
--------------------------------------------------------------------

NB simtrace "photon" *a* is very different from real sphoton.

a.q0.f
    prd.q0.f normal, distance, aka "isect"
a.q1
    intersect position from pos+t*dir, 0.
a.q2
    initial pos, prd.globalPrimIdx_boundary (formerly boundary, and before that tmin)
a.q3
    initial dir, prd.identity


+----+----------------+----------------+----------------+-------------------------+--------------------------+
| q  |      x         |      y         |     z          |      w                  |  notes                   |
+====+================+================+================+=========================+==========================+
|    |  nrm.x         |  nrm.y         |  nrm.z         |  t/distance             | surface normal at isect, |
| q0 |                |                |                |                         | ray trace distance       |
|    |                |                |                |                         |                          |
+----+----------------+----------------+----------------+-------------------------+--------------------------+
|    |  pos.x         |  pos.y         | pos.z          |  tmin                   | intersect_position, tmin |
| q1 |                |                |                |                         |                          |
|    |                |                |                |                         |                          |
+----+----------------+----------------+----------------+-------------------------+--------------------------+
|    |  pos0.x        |  pos0.y        |  pos0.z        | (unsigned)              | origin_position,boundary |
| q2 |                |                |                | globalPrimIdx_boundary  |                          |
|    |                |                |                | (2,3)                   |                          |
+----+----------------+----------------+----------------+-------------------------+--------------------------+
|    | dir0.x         |  dir0.y        |  dir0.z        | (unsigned)              | initial direction, id    |
| q3 | (3,0)          |                |                | identity                |                          |
|    |                |                |                | (3,3)                   |                          |
+----+----------------+----------------+----------------+-------------------------+--------------------------+

NB : simtrace has almost totally different content and layout compared with photon and record

SEE qsim::generate_photon_simtrace

HUH: INPUT SIMTRACE p HAS DIFFERENT LAYOUT FROM THE OUTPUT


**/

SEVENT_METHOD void sevent::add_simtrace( unsigned idx, const quad4& p, const quad2* prd, float tmin )
{
    float t = prd->distance() ;
    quad4 a ;

    a.q0.f  = prd->q0.f ;                // float4 : intersect normal and distance

    a.q1.f.x = p.q0.f.x + t*p.q1.f.x ;   // intersect position
    a.q1.f.y = p.q0.f.y + t*p.q1.f.y ;
    a.q1.f.z = p.q0.f.z + t*p.q1.f.z ;
    a.q1.f.w = tmin ;   // was 0.f

    a.q2.f.x = p.q0.f.x ;                //  initial pos
    a.q2.f.y = p.q0.f.y ;
    a.q2.f.z = p.q0.f.z ;
    a.q2.u.w = prd->globalPrimIdx_boundary() ;

    a.q3.f.x = p.q1.f.x ;                // initial dir
    a.q3.f.y = p.q1.f.y ;
    a.q3.f.z = p.q1.f.z ;
    a.q3.u.w = prd->identity() ;  // identity from __closesthit__ch (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff )

    simtrace[idx] = a ;
    // NB this is the from zero slot idx not the offset "photon_idx"
    // for multi-launch simtrace with separate buffers for each launch
}

