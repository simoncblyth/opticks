

#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QSCINT_METHOD __device__
#else
   #define QSCINT_METHOD 
#endif 

/**
After DsG4Scintillation genloop 
**/


QSCINT_METHOD void qscint_wavelength(float* wavelength, unsigned id, curandState *s, cudaTextureObject_t texObj ) 
{
    float u0 = curand_uniform(s); 
    wavelength[id] = tex2D<float>(texObj,  u0, 0.f);    
}


QSCINT_METHOD void qscint_photon(quad4* photon, unsigned id, curandState *s, cudaTextureObject_t texObj ) 
{
    quad4& p = photon[id] ;     
    p.zero(); 

    float u0 = curand_uniform(s) ; 
    float u1 = curand_uniform(s) ; 
    float u2 = curand_uniform(s) ;   
    float u3 = curand_uniform(s) ;   

    float wavelength = tex2D<float>(texObj,  u0, 0.f);
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



