#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QSCINT_METHOD __device__ 
#else
   #define QSCINT_METHOD 
#endif 


struct qscint
{
    curandState*        s ; 
    cudaTextureObject_t texObj ; 

    // gs0
    int    Id ; 
    int    ParentId ; 
    int    Spare0 ; 
    int    NumPhotons ; 

    // gs1
    float3 X0 ; 
    float  T0 ; 

    // gs2 
    float3 DeltaPosition ; 
    float  StepLength ;

    // gs3
    int   Code ; 
    float Charge ;
    float Spare1 ;  
    float MidVelocity ; 

    // gs4
    float Spare2 ; 
    float Spare3 ; 
    float Spare4 ; 
    float Spare5 ; 
 
    // gs5 
    float ScintillationTime ; 
    float Spare6 ; 
    float Spare7 ; 
    float Spare8 ; 


    QSCINT_METHOD void  init( const float4* gs,  unsigned gs_id, curandState* s_ , cudaTextureObject_t texObj_ ); 
    QSCINT_METHOD void  fabricate();
    QSCINT_METHOD float wavelength(); 
    QSCINT_METHOD void  photon(quad4* photon, unsigned id, bool reemission );
};


QSCINT_METHOD void qscint::fabricate()
{
    X0 = make_float3( 1000.f, 1000.f, 1000.f ); 
    T0 = 0.f ; 

    DeltaPosition = make_float3( 1.f, 1.f, 1.f ); 
    StepLength = 100.f ;

    Charge = 1.f ; 
    MidVelocity = 300.f ; 

    ScintillationTime = 100.f ; 
}

QSCINT_METHOD void qscint::init( const quad6* gs, unsigned gs_id, curandState* s_, cudaTextureObject_t texObj_ )
{
    s = s_ ; 
    texObj = texObj_ ; 

    const float4& gs0 = gs + gs_id*6 + 0 ;  
    const float4& gs1 = gs + gs_id*6 + 1 ;  
    const float4& gs2 = gs + gs_id*6 + 2 ;  
    const float4& gs3 = gs + gs_id*6 + 3 ;  
    const float4& gs4 = gs + gs_id*6 + 4 ;  
    const float4& gs5 = gs + gs_id*6 + 5 ;  




}


QSCINT_METHOD float qscint::wavelength() 
{
    float u0 = curand_uniform(s); 
    return tex2D<float>(texObj, u0, 0.f);    
}

/**
qscint::photon
----------------

One of the random throws is conditional on reemission/Charge
for simpler alignment could change that in both simulations.

**/

QSCINT_METHOD void qscint::photon(quad4* photon, unsigned id, bool reemission )
{
    quad4& p = photon[id] ;     
    if(!reemission) p.zero(); 

    float u0 = curand_uniform(s) ; 
    float u1 = curand_uniform(s) ; 
    float u2 = curand_uniform(s) ;   
    float u3 = curand_uniform(s) ;   

    float wavelength = tex2D<float>(texObj, u0, 0.f);
    float weight = 1.f ; 

    float ct = 1.0f - 2.0f*u1 ;                 // -1.: 1. 
    float st = sqrtf( (1.0f-ct)*(1.0f+ct)) ; 
    float phi = 2.f*M_PIf*u2 ;

    float sp = sinf(phi); 
    float cp = cosf(phi); 

    float3 dir0 = make_float3( st*cp, st*sp,  ct ); 
    float3 pol0 = make_float3( ct*cp, ct*sp, -st );
    float3 perp = cross( dir0, pol0 ); 
    
    float az =  2.f*M_PIf*u3 ; 
    float sz = sin(az);
    float cz = cos(az);

    float3 pol1 = normalize( cz*pol0 + sz*perp ) ; 

    float fraction = ( reemission || Charge == 0.f ) ? 1.f : curand_uniform(s) ;   
    float timeBase = reemission ? p.q0.f.w : T0 + fraction*StepLength/MidVelocity ; 
    float u4 = curand_uniform(s) ; 

    if(!reemission)
    {
        p.q0.f.x = X0.x + fraction*DeltaPosition.x ; 
        p.q0.f.y = X0.y + fraction*DeltaPosition.y ; 
        p.q0.f.z = X0.z + fraction*DeltaPosition.z ; 
    }
    p.q0.f.w = timeBase - ScintillationTime*logf(u4) ;

    p.q1.f.x = dir0.x ; 
    p.q1.f.y = dir0.y ; 
    p.q1.f.z = dir0.z ; 
    p.q1.f.w = weight ;  

    p.q2.f.x = pol1.x ; 
    p.q2.f.y = pol1.y ; 
    p.q2.f.z = pol1.z ; 
    p.q2.f.w = wavelength ; 

    p.q3.u.z = id ; 
}

