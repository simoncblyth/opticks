#pragma once

// mm/ns
#define SPEED_OF_LIGHT 299.792458f

// /usr/local/env/chroma_env/src/chroma/chroma/cuda/photon.h

__device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
{
    float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));
    float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));

    // absorption 
    //   #. advance .time and .position to absorption point
    //   #. if BULK_REEMIT(CONTINUE) change .direction .polarization .wavelength
    //   #. if BULK_ABSORB(BREAK)  .last_hit_triangle -1  
    //
    if (absorption_distance <= scattering_distance) 
    {
        if (absorption_distance <= s.distance_to_boundary) 
        {
            p.time += absorption_distance/(SPEED_OF_LIGHT/s.material1.x);  // x:refractive_index
            p.position += absorption_distance*p.direction;

            float uniform_sample_reemit = curand_uniform(&rng);
            if (uniform_sample_reemit < s.material1.w)   //w:reemission_prob
            {
                // no materialIndex input to reemission_lookup as both scintillators share same CDF 
                // non-scintillators have zero reemission_prob
                p.wavelength = reemission_lookup(curand_uniform(&rng));
                p.direction = uniform_sphere(&rng);
                p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));
                p.flags.i.x = 0 ;  // no-boundary-yet for new direction
                p.flags.i.w |= BULK_REEMIT;
                return CONTINUE;
            }   // photon is reemitted isotropically
            else 
            {
                p.flags.i.w |= BULK_ABSORB;
                return BREAK;
            }   // photon is absorbed in material1
        }
    }
    //  RAYLEIGH_SCATTER(CONTINUE)  .time .position advanced to scatter point .direction .polarization twiddled 
    //
    else 
    {
        if (scattering_distance <= s.distance_to_boundary) 
        {
            p.time += scattering_distance/(SPEED_OF_LIGHT/s.material1.x);  // x:refractive_index
            p.position += scattering_distance*p.direction;

            rayleigh_scatter(p, rng);

            p.flags.i.w |= RAYLEIGH_SCATTER;
            p.flags.i.x = 0 ;  // no-boundary-yet for new direction

            return CONTINUE;
        } // photon is scattered in material1
    }     // if scattering_distance < absorption_distance


    //  Survive to boundary(PASS)  .position .time advanced to boundary 
    //
    p.position += s.distance_to_boundary*p.direction;
    p.time += s.distance_to_boundary/(SPEED_OF_LIGHT/s.material1.x);   // x:refractive_index

    return PASS;

} // propagate_to_boundary


__device__ void propagate_at_boundary( Photon& p, State& s, curandState &rng)
{

    float eta = s.material1.x/s.material2.x ; // x:refractive_index  

    // normal incidence photons do not have a unique plane of incidence, 
    // using polarisation direction will yield normal_coefficient = 1.0f 
    // so will be treated as S polarized 
    //
    //     c1 = 1
    //     c2 = 1
    //
    //
   
    float3 incident_plane_normal = fabs(s.cos_theta) < 1e-6f ? p.polarization : normalize(cross(p.direction, s.surface_normal)) ;

    float normal_coefficient = dot(p.polarization, incident_plane_normal);

    const float c1 = -dot(p.direction, s.surface_normal );    // as-is (unflipped) geometry surface normal 

    const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ; 

    const float c2 = c2c2 < 0.f ? 0.f : sqrtf(c2c2) ;        // TIR for <0.f

    // NB artificially setting c2 to 0.f for TIR results in reflection_coefficient = 1.0f  for both S and P cases

    const float eta_c1 = eta * c1 ; 
    const float eta_c2 = eta * c2 ; 

    bool s_polarized = curand_uniform(&rng) < normal_coefficient*normal_coefficient ;

    const float reflection_coefficient = s_polarized ? 
                         (eta_c1 - c2)/(eta_c1 + c2 )  
                      :
                         (c1 - eta_c2)/(c1 + eta_c2)  ; 

    p.polarization = s_polarized ? 
                          incident_plane_normal
                       :
                          normalize(cross(incident_plane_normal, p.direction));
    
                    
    // S polarized : E field perpendicular to plane of incidence
    // P polarized : E field within plane of incidence 


    bool reflect = curand_uniform(&rng) < reflection_coefficient*reflection_coefficient ;

    p.direction = reflect ? 
                      p.direction + 2.0f*c1*s.surface_normal 
                    : eta*p.direction + (eta_c1 - c2)*s.surface_normal ;   
    
    
   

}




