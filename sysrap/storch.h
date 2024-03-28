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
   #define STORCH_METHOD inline
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
 
    float2  zenith ;  // for T_RECTANGLE : repurposed for the Z values of rect sides
    float2  azimuth ; // for T_RECTANGLE : repurposed for the X values of rect sides 

    // beam
    float    radius ; 
    float    distance ; 
    unsigned mode ;     // basemode 
    unsigned type ;     // basetype

    // NB : organized into 6 quads : are constained not to change that 

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
   STORCH_METHOD static void generate( sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ); 
#else
   STORCH_METHOD static void generate( sphoton& p, srng&              rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ); 
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   float* cdata() const {  return (float*)&gentype ; }
   static constexpr const char* storch_FillGenstep_pos        = "storch_FillGenstep_pos" ; 
   static constexpr const char* storch_FillGenstep_time       = "storch_FillGenstep_time" ; 
   static constexpr const char* storch_FillGenstep_mom        = "storch_FillGenstep_mom" ; 
   static constexpr const char* storch_FillGenstep_wavelength = "storch_FillGenstep_wavelength" ; 
   static constexpr const char* storch_FillGenstep_distance   = "storch_FillGenstep_distance" ; 
   static constexpr const char* storch_FillGenstep_weight     = "storch_FillGenstep_weight" ; 
   static constexpr const char* storch_FillGenstep_radius     = "storch_FillGenstep_radius" ; 
   static constexpr const char* storch_FillGenstep_zenith     = "storch_FillGenstep_zenith" ; 
   static constexpr const char* storch_FillGenstep_azimuth    = "storch_FillGenstep_azimuth" ; 
   static constexpr const char* storch_FillGenstep_type       = "storch_FillGenstep_type" ; 
   static void FillGenstep( storch& gs, int genstep_id, int numphoton_per_genstep, bool dump=false ) ; 
   std::string desc() const ; 
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

/**
storch::FillGenstep
----------------------

Canonically invoked from SEvent::MakeGensteps

**/
inline void storch::FillGenstep( storch& gs, int genstep_id, int numphoton_per_genstep, bool dump )
{
    gs.gentype = OpticksGenstep_TORCH ; 
    gs.numphoton = numphoton_per_genstep  ;   

    qvals( gs.pos , storch_FillGenstep_pos , "0,0,-90" );    
    if(dump) printf("//storch::FillGenstep storch_FillGenstep_pos gs.pos (%10.4f %10.4f %10.4f) \n", gs.pos.x, gs.pos.y, gs.pos.z ); 

    qvals( gs.time, storch_FillGenstep_time, "0.0" ); 
    if(dump) printf("//storch::FillGenstep storch_FillGenstep_time gs.time (%10.4f) \n", gs.time ); 

    qvals( gs.mom , storch_FillGenstep_mom , "0,0,1" );    
    gs.mom = normalize(gs.mom);  // maybe should skip this float normalized, relying instead on U4VPrimaryGenerator::GetPhotonParam to do the normalize ?
    if(dump) printf("//storch::FillGenstep storch_FillGenstep_mom gs.mom (%10.4f %10.4f %10.4f) \n", gs.mom.x, gs.mom.y, gs.mom.z ); 

    qvals( gs.wavelength, storch_FillGenstep_wavelength, "420" ); 
    if(dump) printf("//storch::FillGenstep storch_FillGenstep_wavelength gs.wavelength (%10.4f) \n", gs.wavelength  ); 

    qvals( gs.distance, storch_FillGenstep_distance, "0" ); 
    if(dump) printf("//storch::FillGenstep storch_FillGenstep_distance gs.distance (%10.4f) \n", gs.distance  ); 

    qvals( gs.weight, storch_FillGenstep_weight, "0" ); 
    if(dump) printf("//storch::FillGenstep storch_FillGenstep_weight gs.weight (%10.4f) \n", gs.weight  ); 

    qvals( gs.zenith,  storch_FillGenstep_zenith,  "0,1" ); 
    if(dump) printf("//storch::FillGenstep storch_FillGenstep_zenith gs.zenith (%10.4f,%10.4f) \n", gs.zenith.x, gs.zenith.y  ); 

    qvals( gs.azimuth,  storch_FillGenstep_azimuth,  "0,1" ); 
    if(dump) printf("//storch::FillGenstep storch_FillGenstep_azimuth gs.azimuth (%10.4f,%10.4f) \n", gs.azimuth.x, gs.azimuth.y  ); 

    qvals( gs.radius, storch_FillGenstep_radius, "50" ); 
    if(dump) printf("//storch::FillGenstep storch_FillGenstep_radius gs.radius (%10.4f) \n", gs.radius ); 

    const char* type = qenv(storch_FillGenstep_type, "disc" );  
    unsigned ttype = storchtype::Type(type) ; 
    bool ttype_valid = storchtype::IsValid(ttype) ; 
    if(!ttype_valid) printf("//storch::FillGenstep FATAL TTYPE INVALID %s:[%s][%d] \n", storch_FillGenstep_type, type, ttype ) ; 
    assert(ttype_valid);  

    gs.type = ttype ;  
    if(dump) printf("//storch::FillGenstep storch_FillGenstep_type %s  gs.type %d \n", type, gs.type ); 
    gs.mode = 255 ;    //torchmode::Type("...");  
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



/**
storch::generate
-----------------

On GPU this is invoked by::

   CSGOptiX7.cu:simulate 
   qsim::generate_photon

On CPU this is invoked using MOCK_CURAND with for example::

   G4CXApp::GeneratePrimaries
   U4VPrimaryGenerator::GeneratePrimaries
   SGenerate::GeneratePhotons
   storch::generate    


Populate "sphoton& p" as parameterized by "const quad6& gs_" which casts to "const storch& gs",
the photon_id and genstep_id inputs are informational only. 

Old workflow equivalent ~/opticks/optixrap/cu/torchstep.h


**/

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
STORCH_METHOD void storch::generate( sphoton& p, curandStateXORWOW& rng, const quad6& gs_, unsigned photon_id, unsigned genstep_id )  // static
#else
STORCH_METHOD void storch::generate( sphoton& p, srng&              rng, const quad6& gs_, unsigned photon_id, unsigned genstep_id )  // static
#endif
{
    const storch& gs = (const storch&)gs_ ;   // casting between union-ed types : quad6 and storch  

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

        /**
        disc/T_DISC
            zenith.x->zenith.y radial range, eg [0. 100.] filled disc, [90., 100.] annulus 
            azimuth.x->azimuth.y phi segment in fraction of twopi [0,1] for complete segment 

        **/ 
        //printf("//storch::generate T_DISC gs.type %d gs.mode %d  \n", gs.type, gs.mode ); 

        p.wavelength = gs.wavelength ; 
        p.time = gs.time ; 
        p.mom = gs.mom ; 

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
        float u_zenith  = gs.zenith.x  + curand_uniform(&rng)*(gs.zenith.y-gs.zenith.x)   ;
        float u_azimuth = gs.azimuth.x + curand_uniform(&rng)*(gs.azimuth.y-gs.azimuth.x) ;
#else
        float u_zenith  = gs.zenith.x  + srng::uniform(&rng)*(gs.zenith.y-gs.zenith.x)   ;
        float u_azimuth = gs.azimuth.x + srng::uniform(&rng)*(gs.azimuth.y-gs.azimuth.x) ;
#endif

        float r = gs.radius*u_zenith ;

        float phi = 2.f*M_PIf*u_azimuth ; 
        float sinPhi = sinf(phi); 
        float cosPhi = cosf(phi);
        // __sincosf(phi,&sinPhi,&cosPhi);   // HMM: think thats an apple extension 

        p.pos.x = r*cosPhi ;
        p.pos.y = r*sinPhi ; 
        p.pos.z = 0.f ;   
        // 3D rotate the positions to make their disc perpendicular to p.mom for a nice beam   
        smath::rotateUz(p.pos, p.mom) ; 
        p.pos = p.pos + gs.pos ; // translate position after orienting the disc 

        p.pol.x = sinPhi ;
        p.pol.y = -cosPhi ; 
        p.pol.z = 0.f ;    
        // p.pol.z zero in initial frame, so rotating the frame to arrange 
        // z to be in p.mom direction makes pol transverse to mom
        // NOTICE : ARBITRARY PHASE OF POLARIZATION CHOICE HERE 
        smath::rotateUz(p.pol, p.mom) ; 
    }
    else if( gs.type == T_SPHERE )
    {
        /**
        T_SPHERE
             generates positions on a sphere of gs.radius and radial momentum direction
             outwards(inwards) for gs.radius +ve(-ve) 
        **/
 

        p.wavelength = gs.wavelength ; 
        p.time = gs.time ; 

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
        float u_zenith  = gs.zenith.x  + curand_uniform(&rng)*(gs.zenith.y-gs.zenith.x)   ;
        float u_azimuth = gs.azimuth.x + curand_uniform(&rng)*(gs.azimuth.y-gs.azimuth.x) ;
#else
        float u_zenith  = gs.zenith.x  + srng::uniform(&rng)*(gs.zenith.y-gs.zenith.x)   ;
        float u_azimuth = gs.azimuth.x + srng::uniform(&rng)*(gs.azimuth.y-gs.azimuth.x) ;
#endif

        float phi = 2.f*M_PIf*u_azimuth ;  // azimuth range 0->2pi 
        float sinPhi = sinf(phi); 
        float cosPhi = cosf(phi);
 
        float cosTheta = 1.f - 2.0f*u_zenith  ;   // polar range 0->pi
        float sinTheta = sqrtf( 1.0f - cosTheta*cosTheta ); 

        float flip = copysignf( 1.f, gs.radius ); 

        // gs.radius -ve(+ve) => flip -1.f(+1.f)  
        // below flips direction and not position for outwards/inwards control
  
        p.mom.x = flip*sinTheta*cosPhi ;  
        p.mom.y = flip*sinTheta*sinPhi ; 
        p.mom.z = flip*cosTheta ; 

        float radius = fabs(gs.radius); 
        p.pos.x = sinTheta*cosPhi*radius ; 
        p.pos.y = sinTheta*sinPhi*radius ; 
        p.pos.z = cosTheta*radius ; 

        // float frac_twopi = 0.0f ;    // tangent vectors up the sphere (towards +Z pole, increasing theta)
        // float frac_twopi = 0.5f ;    // tangent vectors down the sphere (towards -Z pole, decreasing theta) 
        // float frac_twopi = 0.25f ;   // tangents around the sphere in direction of increasing phi
        // float frac_twopi = 0.75f ;      // tangents around the sphere in direction of decreasing phi 
        // NOTICE : ARBITRARY PHASE OF POLARIZATION CHOICE HERE : VARIOUS TANGENTS TO THE SPHERE
        float frac_twopi = gs.distance ;  // repurpose the distance, as frac_twopi 

        float phase = 2.f*M_PIf*frac_twopi ; 
        p.pol.x = cosf(phase) ; 
        p.pol.y = sinf(phase) ; 
        p.pol.z = 0.f ;   
        // p.pol.z zero in initial frame, so rotating the frame to arrange 
        // z to be in p.mom direction makes pol transverse to mom
        smath::rotateUz(p.pol, p.mom); 
    }
    else if( gs.type == T_SPHERE_MARSAGLIA )
    {
        /**
        T_SPHERE_MARSAGLIA
             generates positions on a sphere of gs.radius and radial momentum direction
             outwards(inwards) for gs.radius +ve(-ve) 

             uses Marsaglia rejection sampling to get points on unit sphere

             using zenith/azimuth does restrict the range, but in a funny tent shape 

        **/
        p.wavelength = gs.wavelength ; 
        p.time = gs.time ; 
  
        float u0_zenith, u1_azimuth  ; 
        float u, v, b, a  ; 

        do
        {
#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
            u0_zenith  = gs.zenith.x  + curand_uniform(&rng)*(gs.zenith.y-gs.zenith.x)   ;   // aka polar theta 
            u1_azimuth = gs.azimuth.x + curand_uniform(&rng)*(gs.azimuth.y-gs.azimuth.x) ;   // aka azimuth phi
#else
            u0_zenith  = gs.zenith.x  + srng::uniform(&rng)*(gs.zenith.y-gs.zenith.x)   ;
            u1_azimuth = gs.azimuth.x + srng::uniform(&rng)*(gs.azimuth.y-gs.azimuth.x) ;
#endif
            u = 2.f*u0_zenith - 1.f ; 
            v = 2.f*u1_azimuth - 1.f ; 
            b = u*u + v*v ; 
        }    
        while( b > 1.f ) ;  
        a = 2.f*sqrtf( 1.f - b );   

#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
        //printf("//storch::generate T_SPHERE gs.radius %10.4f gs.distance %10.4f \n", gs.radius, gs.distance ); 
#endif       
        float radius = fabs(gs.radius) ; 
        float flip = copysignf( 1.f, gs.radius ); 

        // gs.radius -ve(+ve) => flip -1.f(+1.f)  
        // want to flip direction but not position and avoid extra storage 
        p.mom.x = flip*a*u ; 
        p.mom.y = flip*a*v ; 
        p.mom.z = flip*(2.f*b - 1.f) ; 

        p.pos.x = a*u*radius ;  
        p.pos.y = a*v*radius ; 
        p.pos.z = (2.f*b-1.f)*radius ; 

        // float frac_twopi = 0.0f ;    // tangent vectors up the sphere (towards +Z pole, increasing theta)
        // float frac_twopi = 0.5f ;    // tangent vectors down the sphere (towards -Z pole, decreasing theta) 
        // float frac_twopi = 0.25f ;   // tangents around the sphere in direction of increasing phi
        // float frac_twopi = 0.75f ;      // tangents around the sphere in direction of decreasing phi 
        // NOTICE : ARBITRARY PHASE OF POLARIZATION CHOICE HERE : VARIOUS TANGENTS TO THE SPHERE
        float frac_twopi = gs.distance ;  // repurpose the distance, as frac_twopi 

        float phase = 2.f*M_PIf*frac_twopi ; 
        p.pol.x = cosf(phase) ; 
        p.pol.y = sinf(phase) ; 
        p.pol.z = 0.f ;   
        // p.pol.z zero in initial frame, so rotating the frame to arrange 
        // z to be in p.mom direction makes pol transverse to mom
        smath::rotateUz(p.pol, p.mom); 
    }
    else if( gs.type == T_LINE )
    {
        /**
        T_LINE
            photons start at positions (varying by photon_id) 
            along a line from -gs.radius to +gs.radius 
            wavelength and time are fixed. 
            The position and polarization are oriented
            such that the local frame z is in the gs.mom direction.  

        **/
        p.wavelength = gs.wavelength ; 
        p.time = gs.time ; 
        p.mom = gs.mom ; 

        float frac = float(photon_id)/float(gs.numphoton) ;  // 0->~1 
        float sfrac = 2.f*(frac-0.5f) ;     // -1 -> ~1
        float r = gs.radius*sfrac ;        // -gs.radius -> gs.radius  (NB gets offset by gs.pos too) 

        p.pos.x = r ;
        p.pos.y = 0.f ; 
        p.pos.z = 0.f ;   

        smath::rotateUz(p.pos, p.mom) ; 
        p.pos = p.pos + gs.pos ; // translate position after orienting the line

        p.pol.x = 0.f ;
        p.pol.y = -1.f ; 
        p.pol.z = 0.f ;    
        smath::rotateUz(p.pol, p.mom) ; 
    }
    else if( gs.type == T_POINT )
    {
        /**
        T_POINT
             all photons start at gs.pos, local frame z is rotate into 
             gs.mom direction as is local -Y polarization direction
        **/
        p.wavelength = gs.wavelength ; 
        p.time = gs.time ; 
        p.mom = gs.mom ; 

        p.pos.x = 0.f ;
        p.pos.y = 0.f ; 
        p.pos.z = 0.f ;   

        smath::rotateUz(p.pos, p.mom) ; 
        p.pos = p.pos + gs.pos ; // translate position after orienting the line

        p.pol.x = 0.f ;
        p.pol.y = -1.f ; 
        p.pol.z = 0.f ;    
        smath::rotateUz(p.pol, p.mom) ; 
    }
    else if( gs.type == T_CIRCLE )
    {
        /**
        T_CIRCLE
             phi position around circle of radius |gs.radius| based on photon_id 
             at position gs.pos

             gs.radius>0(<0)
                 local mom is radially outwards(inwards) in XZ plane

             local -Y pol direction is oriented according to the 
             local radial mom direction  

        **/
        p.wavelength = gs.wavelength ; 
        p.time = gs.time ; 
        float f = float(photon_id)/float(gs.numphoton) ;      // 0->~1 
        float frac = gs.azimuth.x*(1.f-f) + gs.azimuth.y*(f) ; // gs.azimuth.x -> gs.azimuth.y which defaults to 0->1 

        float phi = 2.f*M_PIf*frac ; 
        float sinPhi = sinf(phi); 
        float cosPhi = cosf(phi);

        float r = gs.radius < 0.f ? -gs.radius : gs.radius ; 

        // -ve radius for inwards rays 
        // +ve radius or zero for outwards rays 

        p.mom.x = gs.radius < 0.f ? -cosPhi : cosPhi ; 
        p.mom.y = 0.f ; 
        p.mom.z = gs.radius < 0.f ? -sinPhi : sinPhi ; 

        p.pos.x = r*cosPhi ;
        p.pos.y = 0.f ; 
        p.pos.z = r*sinPhi ;   
        // smath::rotateUz(p.pos, p.mom) ;  // dont do that that  
        p.pos = p.pos + gs.pos ; // translate position after orienting the line

        /*
        printf("// T_CIRCLE frac %10.4f gs.radius %10.4f r %10.4f  p.mom (%10.4f %10.4f %10.4f) p.pos (%10.4f %10.4f %10.4f) \n", 
           frac, gs.radius, r, p.mom.x, p.mom.y, p.mom.z, p.pos.x, p.pos.y, p.pos.z ); 
        */

        p.pol.x = 0.f ;
        p.pol.y = -1.f ; 
        p.pol.z = 0.f ;    
        smath::rotateUz(p.pol, p.mom) ; 
    }
    else if( gs.type == T_RECTANGLE )
    {
        /**
        DIVIDE TOTAL PHOTON SLOTS INTO FOUR SIDES 
        ie side is 0,0,0...,1,1,1...,2,2,2,..,3,3,3...

              +------3:top-------+- gs.zenith.y
              |                  |
              |                  |
              0:left             1:right
              |                  |
              |                  |
              +---2:bottom-------+- gs.zenith.x
              |                  |
              gs.azimuth.x       gs.azimuth.y

        **/ 

        p.wavelength = gs.wavelength ; 
        p.time = gs.time ; 

        int side_size = gs.numphoton/4 ; 
        int side = photon_id/side_size ;  
        int side_offset = side*side_size ; 
        int side_index = photon_id - side_offset ; // index within the side
        float frac = float(side_index)/float(side_size) ;  // 0->~1  within the side

        if( side == 0 || side == 1 ) // left or right 
        {
            p.pos.x = side == 0 ? gs.azimuth.x : gs.azimuth.y ; 
            p.pos.y = 0.f ; 
            p.pos.z = (1.f-frac)*gs.zenith.x + frac*gs.zenith.y ; 

            p.mom.x = side == 0 ? 1.f : -1.f ;   // inwards
            p.mom.y = 0.f ; 
            p.mom.z = 0.f ; 
        }
        else if( side == 2 || side == 3)  // bottom or top
        {
            p.pos.x = (1.f-frac)*gs.azimuth.x + frac*gs.azimuth.y ; 
            p.pos.y = 0.f ; 
            p.pos.z = side == 2 ? gs.zenith.x : gs.zenith.y ;   

            p.mom.x = 0.f ; 
            p.mom.y = 0.f ; 
            p.mom.z = side == 2 ? 1.f : -1.f ;   
        }
        p.pos = p.pos + gs.pos ; // translate position, often gs.pos is origin anyhow

        p.pol.x = 0.f ;
        p.pol.y = -1.f ;    // point out the XZ plane, so its transverse
        p.pol.z = 0.f ;    
        smath::rotateUz(p.pol, p.mom) ; 
    }
    p.zero_flags(); 
    p.set_flag(TORCH); 
}




/**
* qtorch : union between quad6 and specific genstep types for easy usage and yet no serialize/deserialize needed
**/

union qtorch
{
   quad6  q ; 
   storch t ; 
};   



