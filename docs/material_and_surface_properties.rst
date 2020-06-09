Material and Surface Properties
===================================

Opticks handling of material and surface properties is all about 
preparing GPU textures : which means they have to be standardized
to fit into arrays with specific properties and common 
wavelength domain.   

Materials must have a non-zero REEMISSIONPROB 
for it to be regarded as a scintillator and 
collected by GScintillatorLib.

The below guide to the relevant classes is intended to 
show how material/surface properties get translated into 
GPU textures and how those are accessed GPU side.


ggeo/GScintillatorLib 
     
     scintillator lib is populated by GGeo::prepareScintillatorLib  (invoked by GGeo::prepare)
     from raw materials with the 3 properties :
          SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB

     For a material to be regarded as a scintillator it must have all 
     these three properties. 

     Also GScintillatorLib prepares the reemission inverse CDF array. 


optixrap/OScintillatorLib 
    Converts the GScintillatorLib array into the reemission GPU texture.
    Lookups on the reemission texture with uniform randoms generates wavelengths
    according to the desired distribution.


ggeo/GMaterialLib,GMaterial

    materials get standardized into  
    8 properties and a common wavelength domain,  
    as they are destined to get interleaved into the 
    boundary GPU texture.

    For a material to be a scintillator must have non-zero "reemission_prob:REEMISSIONPROB,”  

    The 8 material properties (3 are spare).

::

      63 const char* GMaterialLib::keyspec =
      64 "refractive_index:RINDEX,"
      65 "absorption_length:ABSLENGTH,"
      66 "scattering_length:RAYLEIGH,"
      67 "reemission_prob:REEMISSIONPROB,"
      68 "group_velocity:GROUPVEL,"
      69 "extra_y:EXTRA_Y,"
      70 "extra_z:EXTRA_Z,"
      71 "extra_w:EXTRA_W,"


ggeo/GSurfaceLib,GSurface

    surface properties are standardized into 
    8 properties (4 are spare)  

::

      51 const char* GSurfaceLib::detect            = "detect" ;
      52 const char* GSurfaceLib::absorb            = "absorb" ;
      53 const char* GSurfaceLib::reflect_specular  = "reflect_specular" ;
      54 const char* GSurfaceLib::reflect_diffuse   = "reflect_diffuse" ;
      55 
      56 const char* GSurfaceLib::extra_x          = "extra_x" ;
      57 const char* GSurfaceLib::extra_y          = "extra_y" ;
      58 const char* GSurfaceLib::extra_z          = "extra_z" ;
      59 const char* GSurfaceLib::extra_w          = "extra_w" ;
      60 

      87 const char* GSurfaceLib::keyspec =
      88 "detect:EFFICIENCY,"
      89 "absorb:DUMMY,"
      90 "reflect_specular:REFLECTIVITY,"
      91 "reflect_diffuse:REFLECTIVITY,"
      92 ;


ggeo/GBnd
    A boundary is the 4 integers (omat,osur,isur,imat)
    representing materials and surfaces on either side
    of a transition been materials.

::

       omat:outer-material
       osur:outer-surface
       isur:inner-surface
       imat:inner-material

ggeo/GBndLib  
    collects all unique boundaries GBnd for the entire geometry, 
    boundary indices for all bits of the geometry are stored with the geometry 

    interleaves the standardized material and surface property arrays 
    from GMaterialLib and GSurfaceLib into the boundary dynamic array  

optixrap/OBndLib

    converts GBndLib into the GPU boundary texture and uploads its content to GPU 

optixrap/cu/boundary_lookup.h
      static __device__ __inline__ float4 boundary_lookup( float nm, unsigned int line, unsigned int k)


 
optixrap/cu/state.h

     fill_state does the boundary lookups     

     All pieces of geometry have a boundary index assigned
     to it, which means that on intersection with a ray you get the 
     boundary index which via offsets into the boundary texture
     enable you to access the material and surface properties 
     omat/osur/isur/imat relevant to that intersect.

     This approach copies material and surface properties
     multiple times, but the boundary texture is very small compared
     to what GPUs are designed to handle so its not a problem and 
     it simplifies property lookup.

::

     23 struct State
     24 {  
     25    unsigned int flag ;   
     26    float4 material1 ;    // refractive_index/absorption_length/scattering_length/reemission_prob
     27    float4 m1group2  ;    // group_velocity/spare1/spare2/spare3
     28    float4 material2 ;    
     29    float4 surface    ;   //  detect/absorb/reflect_specular/reflect_diffuse
     30    float3 surface_normal ;
     31    float cos_theta ; 
     32    float distance_to_boundary ;
     33    uint4 optical ;   // x/y/z/w index/type/finish/value  
     34    uint4 index ;     // indices of m1/m2/surf/sensor
     35    uint4 identity ;  //  node/mesh/boundary/sensor indices of last intersection
     36    float ureflectcheat ;
     37 };
     38 
     ..
         
     48 __device__ void fill_state( State& s, int boundary, uint4 identity, float wavelength )
     49 {
     50     // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
     51     // >0 outward going photon
     52     // <0 inward going photon
     53     //
     54     // NB the line is above the details of the payload (ie how many float4 per matsur) 
     55     //    it is just 
     56     //                boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 
     57     //
     58 
     59     int line = boundary > 0 ? (boundary - 1)*BOUNDARY_NUM_MATSUR : (-boundary - 1)*BOUNDARY_NUM_MATSUR  ;
     60 
     61     // pick relevant lines depening on boundary sign, ie photon direction relative to normal
     62     // 
     63     int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;
     64     int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;
     65     int su_line = boundary > 0 ? line + ISUR : line + OSUR ;
     66 
     67     //  consider photons arriving at PMT cathode surface
     68     //  geometry normals are expected to be out of the PMT 
     69     //
     70     //  boundary sign will be -ve : so line+3 outer-surface is the relevant one
     71 
     72     s.material1 = boundary_lookup( wavelength, m1_line, 0);
     73     s.m1group2  = boundary_lookup( wavelength, m1_line, 1);
     74 
     75     s.material2 = boundary_lookup( wavelength, m2_line, 0);
     76     s.surface   = boundary_lookup( wavelength, su_line, 0);
     77 
     78     s.optical = optical_buffer[su_line] ;   // index/type/finish/value
     79 



optixrap/cu/generate.cu

     fill_state is the first thing down after an OptiX ray trace intersection

::

    564         rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );
    565 
    566         if(prd.boundary == 0)
    567         {
    568             s.flag = MISS ;  // overwrite CERENKOV/SCINTILLATION for the no hitters
    569             // zero out no-hitters to avoid leftovers 
    570             s.index.x = 0 ;
    571             s.index.y = 0 ;
    572             s.index.z = 0 ;
    573             s.index.w = 0 ;
    574             break ;
    575         }
    576         // initial and CONTINUE-ing records
    577 
    578         // use boundary index at intersection point to do optical constant + material/surface property lookups 
    579         fill_state(s, prd.boundary, prd.identity, p.wavelength );




optixrap/cu/wavelength_lookup.h 

    reemission_lookup is used depending on reemission_prob and a random throw,
    note use of the state  s.material1.w  


optixrap/cu/propagate.h

::

    122 
    123     if (absorption_distance <= scattering_distance)
    124     {
    125         if (absorption_distance <= s.distance_to_boundary)
    126         {
    127             p.time += absorption_distance/speed ;
    128             p.position += absorption_distance*p.direction;
    129 
    130             const float& reemission_prob = s.material1.w ;
    131             float u_reemit = reemission_prob == 0.f ? 2.f : curand_uniform(&rng);  // avoid consumption at absorption when not scintillator
    132 
    133             if (u_reemit < reemission_prob)
    134             {
    135                 // no materialIndex input to reemission_lookup as both scintillators share same CDF 
    136                 // non-scintillators have zero reemission_prob
    137                 p.wavelength = reemission_lookup(curand_uniform(&rng));
    138                 p.direction = uniform_sphere(&rng);
    139                 p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));
    140                 p.flags.i.x = 0 ;   // no-boundary-yet for new direction
    141 
    142                 s.flag = BULK_REEMIT ;
    143                 return CONTINUE;
    144             }
    145             else
    146             {
    147                 s.flag = BULK_ABSORB ;
    148                 return BREAK;
    149             }
    150         }
    151         //  otherwise sail to boundary  
    152     }
    153     else



