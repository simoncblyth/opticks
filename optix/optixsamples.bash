# === func-gen- : optix/optixsamples fgp optix/optixsamples.bash fgn optixsamples fgh optix
optixsamples-src(){      echo optix/optixsamples.bash ; }
optixsamples-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(optixsamples-src)} ; }
optixsamples-vi(){       vi $(optixsamples-source) ; }
optixsamples-env(){      olocal- ; }
optixsamples-usage(){ cat << EOU


OPTIX SAMPLES
===============

glass
------

triangle_mesh_iterative.cu::

     55   float3 p0 = vertex_buffer[ v_idx.x ];
     56   float3 p1 = vertex_buffer[ v_idx.y ];
     57   float3 p2 = vertex_buffer[ v_idx.z ];
     58 
     59   // Intersect ray with triangle
     60   float3 n;
     61   float  t, beta, gamma;
     62   if( intersect_triangle( ray, p0, p1, p2, n, t, beta, gamma ) ) {
     63 
     64     if(  rtPotentialIntersection( t ) ) {
     ..
     67       float3 geo_n = normalize( n );
     ..
     89 
     90       refine_and_offset_hitpoint( ray.origin + t*ray.direction, ray.direction,
     91                                   geo_n, p0,
     92                                   back_hit_point, front_hit_point );
     93 
     94       rtReportIntersection(material_buffer[primIdx]); 


intersection_refinement.h::


     70 // Refine the hit point to be more accurate and offset it for reflection and
     71 // refraction ray starting points.
     72 static
     73 __device__ __inline__ void refine_and_offset_hitpoint( const optix::float3& original_hit_point, const optix::float3& direction,
     74                                                        const optix::float3& normal, const optix::float3& p,
     75                                                        optix::float3& back_hit_point,
     76                                                        optix::float3& front_hit_point )
     77 {
     78   using namespace optix;
     79 
     80   // Refine hit point
     81   float  refined_t          = intersectPlane( original_hit_point, direction, normal, p );
     //
     //    refined_t is a correction to t, correction applied below 
     //
     82   float3 refined_hit_point  = original_hit_point + refined_t*direction;
     83 
     84   // Offset hit point
     85   if( dot( direction, normal ) > 0.0f ) {
     //
     //     direction and normal on same side so coming from behind
     //    -direction and normal on opposite side so coming from back
     //
     86     back_hit_point  = offset( refined_hit_point,  normal );
     87     front_hit_point = offset( refined_hit_point, -normal );
     88   } else {
     //
     //     -direction and normal on same side so:
     //     back => -normal direction, front => +normal          
     //
     89     back_hit_point  = offset( refined_hit_point, -normal );
     90     front_hit_point = offset( refined_hit_point,  normal );
     91   }
     92 }


     40 // Offset the hit point using integer arithmetic
     41 static __device__ __inline__ optix::float3 offset( const optix::float3& hit_point, const optix::float3& normal )
     42 {     
     43   using namespace optix;
     44   
     45   const float epsilon = 1.0e-4f;
     46   const float offset  = 4096.0f*2.0f;
     47 
     48   float3 offset_point = hit_point;
     49   if( (__float_as_int( hit_point.x )&0x7fffffff)  < __float_as_int( epsilon ) ) {  
     50     offset_point.x += epsilon * normal.x;
     51   } else {
     52     offset_point.x = __int_as_float( __float_as_int( offset_point.x ) + int(copysign( offset, hit_point.x )*normal.x) );
     53   }
     //
     //      if abs(hit_point.x) < epsilon: offset_point.x += epsilon * normal.x
     //      else:
     //           offset_point.x = offset_point.x +  sign(hit_point.x)*offset*normal.x
     //      
     //      fixed offset less objectionable when consider that this is all in object space
     //      BUT checking the coordinates of the object, the offset is huge ? so am missing smth 
     //
     //      trying more reasonable offsets works also:
     //

     47   //const float offset  = 1.0e6f; 
     48   //const float offset  = 4096.0f*2.0f;  // original
     49   //const float offset  = 1000.0f ;      // looks like original
     50   const float offset  = 10.0f ;        // like original
     51   //const float offset  = 2.0f ;           // OK, a few grubby areas 
     52   //const float offset  = 1.0f;          // dark blotchy mess

     //      
     //
     ..
     67   return offset_point;
     68 }






     26 // Plane intersection -- used for refining triangle hit points.  Note
     27 // that this skips zero denom check (for rays perpindicular to plane normal)
     28 // since we know that the ray intersects the plane.
     29 static
     30 __device__ __inline__ float intersectPlane( const optix::float3& origin,
     31                                             const optix::float3& direction,
     32                                             const optix::float3& normal,
     33                                             const optix::float3& point )
     //
     //
     34 {
     35   // Skipping checks for non-zero denominator since we know that ray intersects this plane
     36   return -( optix::dot( normal, origin-point ) ) / optix::dot( normal, direction );
     37 
     38 }
     //
     //   
     //   ratio of projections of (point-origin) and (direction) onto normal direction, 
     //   giving a correction to t ? 
     //
     //         origin + t * direction 
     //
     //   but point is an arbitraily chosen vertex from the three of the intersected triangle 
     //   and "origin" is in fact the unrefined hit point so "point-origin" should be small
     //   and the  
     //   maybe the intention was to use the barycentric point for this 
     //
     //


glass.cu::

     87 RT_PROGRAM void closest_hit_radiance()
     88 {
     89   // intersection vectors
     90   const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
     91   const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
     92   const float3 bhp = rtTransformPoint(RT_OBJECT_TO_WORLD, back_hit_point);
     93   const float3 i = ray.direction;                                            // incident direction
     94         float3 t;                                                            // transmission direction
     95         float3 r;                                                            // reflection direction
     96 
     97   float reflection = 1.0f;
     98   float3 result = make_float3(0.0f);
     99  
    100   const int depth = prd_radiance.depth;
    101 
    102   float3 beer_attenuation;
    103   if(dot(n, ray.direction) > 0) {
    104     // Beer's law attenuation
    105     beer_attenuation = exp(extinction_constant * t_hit);
    106   } else {
    107     beer_attenuation = make_float3(1);
    108   }
    109 
    110   // refraction
    111   if (depth < min(refraction_maxdepth, max_depth))
    112   {
    113     if ( refract(t, i, n, refraction_index) )
    114     {
    115       // check for external or internal reflection
    116       float cos_theta = dot(i, n);
    117       if (cos_theta < 0.0f)
    118         cos_theta = -cos_theta;
    ///
    ///         canonical situation with -i and n on same side means the reflection is external
    ///
    119       else
    ///
    ///         internal reflection  
    ///
    120         cos_theta = dot(t, n);
    121 
    122       reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);
    ///
    ///          larger angle, smaller cos_theta increases reflection
    ///
    123 
    124       float importance = prd_radiance.importance * (1.0f-reflection) * optix::luminance( refraction_color * beer_attenuation );
    ///
    ///           less reflection, more transmission and more importance retained in the recursion
    ///           http://en.wikipedia.org/wiki/Fresnel_equations
    ///

    125       float3 color = cutoff_color;
    126       if ( importance > importance_cutoff ) {
    127         color = TraceRay(bhp, t, depth+1, importance);
    128       }
    129       result += (1.0f - reflection) * refraction_color * color;
    130     }
    131     // else TIR
    132   } // else reflection==1 so refraction has 0 weight
    133 
    134   // reflection
    135   float3 color = cutoff_color;
    136   if (depth < min(reflection_maxdepth, max_depth))
    137   {
    138     r = reflect(i, n);
    139  
    140     float importance = prd_radiance.importance * reflection * optix::luminance( reflection_color * beer_attenuation );
    141     if ( importance > importance_cutoff ) {
    142       color = TraceRay( fhp, r, depth+1, importance );
    143     }
    144   }
    145   result += reflection * reflection_color * color;
    146 
    147   result = result * beer_attenuation;
    148 
    149   prd_radiance.result = result;
    150 }
    ///
    ///
    069 static __device__ __inline__ float3 TraceRay(float3 origin, float3 direction, int depth, float importance )
    070 {
    071   optix::Ray ray = optix::make_Ray( origin, direction, radiance_ray_type, 0.0f, RT_DEFAULT_MAX );
    ///   
    ///     NB tmin 0.0f, as the offsetting is done already via bhp, fhp
    /// 
    072   PerRayData_radiance prd;
    073   prd.depth = depth;
    074   prd.importance = importance;
    075 
    076   rtTrace( top_object, ray, prd );
    077   return prd.result;
    078 }



compare approach to Chroma propagation

* optix glass is splitting the ray into a transmitted and reflected, chroma 
  decides which to do by curand_uniform

* propagate_to_boundary 

::

    411 __noinline__ __device__ void
    412 propagate_at_boundary(Photon &p, State &s, curandState &rng)
    413 {
    414     float incident_angle = get_theta(s.surface_normal,-p.direction);
    415     float refracted_angle = asinf(sinf(incident_angle)*s.refractive_index1/s.refractive_index2);

    /// TODO: compare with G4 



EOU
}
optixsamples-dir(){ echo $(local-base)/env/optix/optix-optixsamples ; }
optixsamples-cd(){  cd $(optixsamples-dir); }
optixsamples-mate(){ mate $(optixsamples-dir) ; }
optixsamples-get(){
   local dir=$(dirname $(optixsamples-dir)) &&  mkdir -p $dir && cd $dir

}
