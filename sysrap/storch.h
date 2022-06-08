#pragma once
/**
storch.h : replace (but stay similar to) : npy/NStep.hpp optixrap/cu/torchstep.h  
===================================================================================

NB sizeof storch struct is **CONSTRAINED TO MATCH quad6** like all gensteps 

Bringing over some of the old torch genstep generation into the modern workflow 
with mocking on CPU and pure-CUDA test cababilities. 

Notes
--------

Techniques to implement the spirit of the old torch gensteps in much less code

* sharing types and code between GPU and CPU 
* quad6 and NP and casting between them
* union between quad6 and simple torch struct eliminates tedious get/set of NStep.hpp
* macros to use same headers on CPU and GPU, eg enum strings live with enum values in same header 
  but are hidden from nvcc


Old Implementation
--------------------

optixrap/cu/torchstep.h 
   OptiX 6 generation 

npy/TorchStepNPY.hpp
npy/TorchStepNPY.cpp
   CPU side preparation of the torch gensteps with enum name strings  
  
   * parsing config ekv strings into gensteps with param language 
   * TorchStepNPY::updateAfterSetFrameTransform 
     frame transform is used to convert from local frame 
     source, target and polarization into the frame provided.
  
npy/GenstepNPY.hpp 
npy/GenstepNPY.cpp 
    * holds m_onestep NStep struct 
    * handles frame targetting 

npy/NStep.hpp
    6 transport quads that are copied into the genstep buffer by addStep
    m_ctrl/m_post/m_dirw/m_polw/m_zeaz/m_beam

    m_array NPY<float> of shape (1,6,4)

npy/NStep.cpp
    NPY::setQuadI NPY::setQuad into the array

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define STORCH_METHOD __device__
#else
   #define STORCH_METHOD 
#endif 


#include "OpticksGenstep.h"
#include "OpticksPhoton.h"

#include "scurand.h"
#include "smath.h"
#include "storchtype.h"

/**
**/

struct storch
{
    // ctrl
    unsigned gentype ;  // eg OpticksGenstep_TORCH
    unsigned trackid ; 
    unsigned matline ; 
    unsigned numphoton ; 
    
    float3   pos ;
    float    time ; 

    float3   mom ;
    float    weight ; 
 
    float3   pol ;
    float    wavelength ; 
 
    float2  zenith ; 
    float2  azimuth ; 

    // beam
    float    radius ; 
    float    distance ; 
    unsigned mode ;     // basemode 
    unsigned type ;     // basetype

    // NB : organized into 6 quads : are constained not to change that 

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) 
   STORCH_METHOD static void generate( sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ); 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   float* cdata() const {  return (float*)&gentype ; }
   static void FillGenstep( storch& gs, unsigned genstep_id, unsigned numphoton_per_genstep ) ; 
   std::string desc() const ; 
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline void storch::FillGenstep( storch& gs, unsigned genstep_id, unsigned numphoton_per_genstep )
{
    float3 mom = make_float3( 0.f, 0.f, 1.f );  

    gs.gentype = OpticksGenstep_TORCH ; 
    gs.wavelength = 501.f ; 
    gs.mom = normalize(mom); 
    gs.radius = 50.f ; 
    gs.pos = make_float3( 0.f, 0.f, -90.f );  
    gs.time = 0.f ; 
    gs.zenith = make_float2( 0.f, 1.f );  
    gs.azimuth = make_float2( 0.f, 1.f );  
    gs.type = storchtype::Type("disc");  
    gs.mode = 255 ;    //torchmode::Type("...");  
    gs.numphoton = numphoton_per_genstep  ;   
}


inline std::string storch::desc() const 
{
    std::stringstream ss ; 
    ss << "storch::desc"
       << " gentype " << gentype 
       << " mode " << mode 
       << " type " << type 
       ;
    std::string s = ss.str(); 
    return s ; 
} 
#endif


#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) 

/**
storch::generate
-----------------

Populate "sphoton& p" as parameterized by "const quad6& gs_" which casts to "const storch& gs",
the photon_id and genstep_id inputs are informational only. 

**/

STORCH_METHOD void storch::generate( sphoton& p, curandStateXORWOW& rng, const quad6& gs_, unsigned photon_id, unsigned genstep_id )  // static
{
    const storch& gs = (const storch&)gs_ ;   // casting between union-ed types  

#ifdef STORCH_DEBUG
    printf("//storch::generate photon_id %3d genstep_id %3d  gs gentype/trackid/matline/numphoton(%3d %3d %3d %3d) type %d \n", 
       photon_id, 
       genstep_id, 
       gs.gentype, 
       gs.trackid,
       gs.matline, 
       gs.numphoton,
       gs.type
      );  
#endif

    if( gs.type == T_DISC )
    {
        //printf("//storch::generate T_DISC gs.type %d gs.mode %d  \n", gs.type, gs.mode ); 

        p.wavelength = gs.wavelength ; 
        p.time = gs.time ; 
        p.mom = gs.mom ; 

        float u_zenith  = gs.zenith.x  + curand_uniform(&rng)*(gs.zenith.y-gs.zenith.x)   ;
        float u_azimuth = gs.azimuth.x + curand_uniform(&rng)*(gs.azimuth.y-gs.azimuth.x) ;

        float r = gs.radius*u_zenith ;

        float sinPhi, cosPhi;
        __sincosf(2.f*M_PIf*u_azimuth,&sinPhi,&cosPhi);   // HMM: think thats an apple extension 

        p.pos.x = r*cosPhi ;
        p.pos.y = r*sinPhi ; 
        p.pos.z = 0.f ;   
        // 3D rotate the positions to make their disc perpendicular to p.mom for a nice beam   
        smath::rotateUz(p.pos, p.mom) ; 
        p.pos = p.pos + gs.pos ; // translate position after orienting the disc 

        p.pol.x = sinPhi ;
        p.pol.y = -cosPhi ; 
        p.pol.z = 0.f ;    
        // pol.z zero in initial frame, so rotating the frame to arrange z to be in p.mom direction makes pol transverse to mom
        smath::rotateUz(p.pol, p.mom) ; 
    }

    p.zero_flags(); 
    p.set_flag(TORCH); 
}

#endif



/**
* qtorch : union between quad6 and specific genstep types for easy usage and yet no serialize/deserialize needed
**/

union qtorch
{
   quad6  q ; 
   storch t ; 
};   



