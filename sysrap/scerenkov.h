#pragma once
/**
scerenkov.h : replace (but stay similar to) : npy/NStep.hpp optixrap/cu/cerenkovstep.h
========================================================================================

* FOLLOWING PATTERN OF storch.h

**/


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define SCERENKOV_METHOD __device__
#else
   #define SCERENKOV_METHOD
#endif

#include "OpticksGenstep.h"
#include "OpticksPhoton.h"

#include "smath.h"
#include "scuda.h"
#include "squad.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include "NP.hh"
#endif

struct scerenkov
{
    // ctrl
    unsigned gentype ;   // formerly Id
    unsigned trackid ;   // formerly ParentId
    unsigned matline ;   // formerly MaterialIndex, used by qbnd::boundary_lookup
    unsigned numphoton ; // formerly NumPhotons

    float3   pos ;  // formerly x0
    float    time ; // formerly t0

    float3 DeltaPosition ;  // aka p0 in G4Cerenkov,  G4Step::GetDeltaPosition is not normalized
    float  step_length ;

    int code;
    float charge ;
    float weight ;
    float preVelocity ;

    /// the above first 4 quads are common to both CerenkovStep and ScintillationStep

    float BetaInverse ;
    float Wmin ;
    float Wmax ;
    float maxCos ;

    float maxSin2 ;
    float MeanNumberOfPhotons1 ;
    float MeanNumberOfPhotons2 ;
    float postVelocity ;

    SCERENKOV_METHOD float Pmin() const { return smath::hc_eVnm/Wmax ; }
    SCERENKOV_METHOD float Pmax() const { return smath::hc_eVnm/Wmin ; }

   static void FillGenstep( scerenkov& gs, int matline, int numphoton_per_genstep, bool dump ) ;

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   float* cdata() const {  return (float*)&gentype ; }
   std::string desc() const ;
   static void MinMaxPost( float* mn, float* mx, const NP* genstep);

   template<typename T>
   static bool IsGenstepArray( const NP* a );

#endif

};



/**
scerenkov::FillGenstep : fabricate some values for a demo genstep
---------------------------------------------------------------------

Uses hard coded values depending on RINDEX of material (LS) that will be used : ie fixing the cone angle.
A better way of doing this would use the MaterialLine as input and obtain the values from the RINDEX,

* (as this code is only needed on CPU only it would be perfectly feasible to read RINDEX arrays)
* however this is debugging code so non-general techniques are acceptable
* NB matline is crucial as that determines which materials RINDEX is used

**/

inline void scerenkov::FillGenstep( scerenkov& gs, int matline, int numphoton_per_genstep, bool dump )
{
    float nMax = 1.793f ;
    float BetaInverse = 1.500f ;
    float maxCos = BetaInverse / nMax;
    float maxSin2 = (1.f - maxCos) * (1.f + maxCos) ;

    float Pmin = 1.55f ;    // eV
    float Pmax = 15.5f ;
    float Wmin = smath::hc_eVnm/Pmax ; // close to: 1240./15.5 = 80.
    float Wmax = smath::hc_eVnm/Pmin ;  // close to: 1240./1.55 = 800.
    // NB in reality this should not use standard domain, but rather the domain of the RINDEX property

    gs.gentype = OpticksGenstep_CERENKOV ;
    gs.trackid = 0u ;
    gs.matline = matline ;
    gs.numphoton = numphoton_per_genstep  ;

    gs.pos.x = 100.f ;
    gs.pos.y = 100.f ;
    gs.pos.z = 100.f ;
    gs.time = 20.f ;

    gs.DeltaPosition.x = 1000.f ;    // aka p0
    gs.DeltaPosition.y = 1000.f ;
    gs.DeltaPosition.z = 1000.f ;
    gs.step_length = 1000.f ;

    gs.code = 1 ;
    gs.charge = 1.f ;
    gs.weight = 1.f ;
    gs.preVelocity = 10.f ;

    gs.BetaInverse = BetaInverse ;
    gs.Wmin = Wmin ;
    gs.Wmax = Wmax ;
    gs.maxCos = maxCos ;  // NOT USED ?

    gs.maxSin2 = maxSin2 ;              // constrains cone angle rejection sampling
    gs.MeanNumberOfPhotons1 = 100.f ;   // used for profile sampling to decide where along the step
    gs.MeanNumberOfPhotons2 = 200.f ;
    gs.postVelocity = 20.f ;

}



#if defined(__CUDACC__) || defined(__CUDABE__)
#else

inline std::string scerenkov::desc() const
{
    std::stringstream ss ;
    ss << "scerenkov::desc"
       << " gentype " << gentype
       ;
    std::string s = ss.str();
    return s ;
}


inline void scerenkov::MinMaxPost( float* mn, float* mx, const NP* _a )
{
    NP* a = const_cast<NP*>(_a);

    bool is_f = IsGenstepArray<float>(a);

    if(!is_f) std::cerr
        << "scerenkov::MinMaxPost FATAL UNEXPECTED ARRAY "
        << " is_f " << ( is_f ? "YES" : "NO " )
        << "\n"
        ;

    bool expect = is_f ;
    assert(expect);
    if(!expect) std::raise(SIGINT);

    int ni = a->num_items() ;
    int nj = 4 ;

    float MAX = std::numeric_limits<float>::max() ;
    for(int j=0 ; j < nj ; j++)
    {
        mn[j] = MAX ;
        mx[j] = -MAX ;
    }

    for(int i=0 ; i < ni ; i++)
    {
        scerenkov* gs = reinterpret_cast<scerenkov*>(a->bytes() + i*a->item_bytes());
        assert( gs );

        //const float3& dpos = gs->DeltaPosition ;
        float4 _p0 = make_float4( gs->pos.x, gs->pos.y, gs->pos.z, gs->time );

        float* p0 = &_p0.x ;

        for(int j=0 ; j < 4 ; j++)
        {
            float vj = p0[j];
            if( vj < mn[j] ) mn[j] = vj ;
            if( vj > mx[j] ) mx[j] = vj ;
        }
    }



}

template<typename T>
inline bool scerenkov::IsGenstepArray( const NP* a )
{
    return a && a->has_shape(-1,6,4) && a->ebyte == sizeof(T) ;
}


#endif

