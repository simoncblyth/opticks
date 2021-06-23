#pragma once

#include "quad.h"
#include "random.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define GENSTEP_METHOD __device__
#else
   #define GENSTEP_METHOD 
#endif 



struct Genstep_DsG4Scintillation_r4695
{
    int Id    ;               // (0)
    int ParentId ;
    int MaterialIndex  ;
    int NumPhotons ;

    float3 x0 ;               // (1)
    float  t0 ;

    float3 DeltaPosition ;    //  (2)
    float  step_length ;

    int   code;                 // (3)
    float charge ;
    float weight ;
    float meanVelocity ; 

    /// the above first 4 quads are common to both CerenkovStep and ScintillationStep 

    int   ScintillationType ;   // (4) 
    float f41_spare ;    
    float f42_spare ;    
    float f43_spare ;

    float ScintillationTime ;    // (5)
    float f51_spare ;
    float f52_spare ;
    float f53_spare ;

    // above are loaded parameters, below are derived from them
    float3 p0 ;

    GENSTEP_METHOD void load(const float4* scintillation, unsigned int offset, unsigned int genstep_id) 
    {
        union quad ipmn, ccwv, ssss ;
     
        ipmn.f = scintillation[offset+0];     
        //int gentype = ipmn.i.x ;  

        Id = genstep_id ;    // usurp the encumbent gentype
        ParentId = ipmn.i.y ; 
        MaterialIndex = ipmn.i.z ; 
        NumPhotons = ipmn.i.w ; 

        float4 xt0 = scintillation[offset+1];
        x0 = make_float3(xt0.x, xt0.y, xt0.z );
        t0 = xt0.w ; 
        
        float4 dpsl = scintillation[offset+2] ;
        DeltaPosition = make_float3(dpsl.x, dpsl.y, dpsl.z );
        step_length = dpsl.w ; 

        ccwv.f = scintillation[offset+3] ;
        code = ccwv.i.x ;
        charge = ccwv.f.y ;
        weight = ccwv.f.z ;
        meanVelocity = ccwv.f.w ;

        ssss.f = scintillation[offset+4] ;
        ScintillationType = ssss.i.x ; 
        f41_spare = ssss.f.y ; 
        f42_spare = ssss.f.z ; 
        f43_spare = ssss.f.w ; 

        float4 ssoo = scintillation[offset+5] ;
        ScintillationTime = ssoo.x ; 
        f51_spare = ssoo.y ; 
        f52_spare = ssoo.z ; 
        f53_spare = ssoo.w ; 

        // derived qtys 
        p0 = normalize(DeltaPosition);
    }

    GENSTEP_METHOD void dump()
    {
#ifdef WITH_PRINT
        rtPrintf("Genstep_G4Scintillation_1042 ss.Id %d ParentId %d MaterialIndex %d NumPhotons %d \n", 
           Id, 
           ParentId, 
           MaterialIndex, 
           NumPhotons 
           );


        rtPrintf("x0 %f %f %f  t0 %f \n", 
           x0.x, 
           x0.y, 
           x0.z, 
           t0 
           );

        rtPrintf("DeltaPosition %f %f %f  step_length %f  \n", 
           DeltaPosition.x, 
           DeltaPosition.y, 
           DeltaPosition.z,
           step_length
           ); 

        rtPrintf("code %d  charge %f weight %f preVelocity %f \n", 
           code,
           charge,
           weight,
           meanVelocity
          );

        rtPrintf("ScintillationType %d f41_spare %f f42_spare %f f43_spare %f \n", 
           ScintillationType,
           f41_spare,
           f42_spare,
           f43_spare
          );

        rtPrintf("ScintillationTime %f  f51_spare %f f52_spare %f f53_spare %f \n", 
           ScintillationTime,
           f51_spare,
           f52_spare,
           f53_spare
          );

        rtPrintf("p0 %f %f %f  \n", 
           p0.x, 
           p0.y, 
           p0.z
           );

#endif
    }

    GENSTEP_METHOD void check()
    {
#ifdef WITH_PRINT
        float nmlo = boundary_sample_domain(0.f);
        float nmmi = boundary_sample_domain(0.5f);   
        float nmhi = boundary_sample_domain(1.0f);

        rtPrintf("check sample wavelength lo/mi/hi   %f %f %f \n", nmlo,nmmi,nmhi);   
#endif
    }

    GENSTEP_METHOD void debug()
    {
        dump();
        check();
    }

    GENSTEP_METHOD void generate_photon(Photon& p, curandState& rng)
    {
        p.wavelength = reemission_lookup(curand_uniform(&rng));

        p.direction = uniform_sphere(&rng);

        p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));  // pol is random vector orthogonal to dir

        float fraction = charge == 0.f ? 1.f : curand_uniform(&rng) ;    

        float delta = fraction * step_length ;  

        float deltaTime = delta/meanVelocity - ScintillationTime*logf(curand_uniform(&rng)) ;  

        p.time = t0 + deltaTime ; 

        p.position = x0 + fraction * DeltaPosition ;  

        p.weight = weight ;

        p.flags.u.x = ScintillationType ;   // scnt:overwritten later
        p.flags.u.y = 0 ;
        p.flags.u.z = 0 ;
        p.flags.u.w = 0 ;
    }


};





