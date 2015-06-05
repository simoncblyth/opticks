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


//
//  fresnel reflect/transmit conventional directions
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                     s1
//                   +----+          
//                    \   .   /      ^
//               c1   i\  .  / r    /|\
//                      \ . /        |                      
//         material1     \./         | n
//         ---------------+----------+----------
//         material2      .\
//                        . \
//                   c2   .  \ t
//                        .   \
//                        +----+
//                          s2
//   i, incident photons 
//      pointing down to interface (from material1 towards material2)
//
//   n, surface normal (s.surface_normal)
//      pointing up from interface (from material2 back into material1)
//      Orientation is arranged by flipping geometric normal 
//      based on photon direction.
//
//   t, transmitted photons
//      from interface into material2
//
//   r, reflected photons
//      from interface back into material1
//
//   c1, costheta_1 
//      cosine of incident angle,  c1 = dot(-i, n) = - dot(i, n)
//      arranged to be positive via normal flipping 
//      and corresponding flip of which material is labelled 1 and 2 
//     
//
//  polarisation
//  ~~~~~~~~~~~~~~~
//                    
//   S polarized : E field perpendicular to plane of incidence
//   P polarized : E field within plane of incidence 
//
//
// normal incidence photons
// ~~~~~~~~~~~~~~~~~~~~~~~~~~ 
// 
// * no unique plane of incidence, 
// * artifically setting incident_plane_normal to initial p.polarisation yields normal_coefficient = 1.0f 
//   so always treated as S polarized 
//   
//
//   initial momentum dir
//            -s.surface_normal 
//
//   final momentum dir (c1 = 1.f)
//            -s.surface_normal + 2.0f*c1*s.surface_normal  = -p.direction 
//                                                    
//
//  minimise use of trancendental functions 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Obtain c2c2 from Snells Law without lots of expensive function calls.
//  
//        n1 s1 = n2 s2
//
//           s2 = eta * s1       eta = n1/n2
//
//
//   
//         c2c2 = 1 - s2s2 
//              = 1 - eta eta s1 s1  
//              = 1 - eta eta (1 - c1c1) 
//
//
//         c2c2 - 1 = (c1c1 - 1) eta eta
//
//        
//
//  TIR
//  ~~~~
//
//  Total internal reflection, occurs when c2c2 < 0.f  (c2 imaginary)
//
//  Handled by: 
//
//  * artificially setting c2 = 0.f 
//  * results in reflection_coefficient = 1.0f so will always reflect for both S and P cases
//
//
//


__device__ void propagate_at_boundary( Photon& p, State& s, curandState &rng)
{
    // see g4op- for comparison of Geant4/Chroma/OptiX-refract

    float eta = s.material1.x/s.material2.x ;    // eta = n1/n2   x:refractive_index  PRE-FLIPPED

    float3 incident_plane_normal = fabs(s.cos_theta) < 1e-6f ? p.polarization : normalize(cross(p.direction, s.surface_normal)) ;

    float normal_coefficient = dot(p.polarization, incident_plane_normal);  // fraction of E vector perpendicular to plane of incidence, ie S polarization

    const float c1 = -dot(p.direction, s.surface_normal ); // c1 arranged to be +ve   

    const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ; 

    bool tir = c2c2 < 0.f ; 

    const float c2 = tir ? 0.f : sqrtf(c2c2) ;   // c2 chosen +ve, set to 0.f for TIR

    const float eta_c1 = eta * c1 ; 

    const float eta_c2 = eta * c2 ;    

    bool s_polarized = curand_uniform(&rng) < normal_coefficient*normal_coefficient ;

    const float reflection_coefficient = s_polarized 
                      ? 
                         (eta_c1 - c2)/(eta_c1 + c2 )  
                      :
                         (c1 - eta_c2)/(c1 + eta_c2)  
                      ; 

    p.polarization = s_polarized 
                       ? 
                          incident_plane_normal
                       :
                          normalize(cross(incident_plane_normal, p.direction))
                       ;
    
    bool reflect = curand_uniform(&rng) < reflection_coefficient*reflection_coefficient ;

    p.direction = reflect 
                    ? 
                       p.direction + 2.0f*c1*s.surface_normal 
                    : 
                       eta*p.direction + (eta_c1 - c2)*s.surface_normal
                    ;   


    p.flags.i.w |= reflect ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT ;
    p.flags.i.w |= s_polarized ? BOUNDARY_SPOL : BOUNDARY_PPOL ;
    p.flags.i.w |= tir ? BOUNDARY_TIR : BOUNDARY_TIR_NOT ; 

}




