OSensorLib_canonical
=======================

Objective
-----------

Bring OSensorLib into canonical workflow providing angle dependent 
efficiency for all sensors. Must also work when non angle dependent 
efficiency has been set. 

Use this for on GPU efficiency culling that sets a new SENSOR_COLLECT in the photon flags.   

Arrange that this can be tested with G4OKTest using geometry loaded from 
cache and separately persisted SensorLib. 


DONE
--------

1. made a github 0.1.0-rc1 tag so can update junoenv opticks 
2. adopt 1-based  sensorIndex with 0 for none, to save half the bits by going unsigned
3. removed cos_theta from prd 

 
TODO
------

* persist the real JUNO SensorLib from junoenv running for use in G4OKTest 



DONE: adopt 1-based  sensorIndex with 0 for none, to save half the bits by going unsigned
--------------------------------------------------------------------------------------------

* currently using -1 for unset sensorIndex forces used of unsigned and wastes almost half the dynamic range

::

    In [2]: (0x1 << 16) - 1
    Out[2]: 65535

    In [3]: (0x1 << 15) - 1
    Out[3]: 32767            


Should save the bits anyhow, but where is the 2 bytes constraint ?


At geometry level using 32 bit::

    259 glm::uvec4 GVolume::getIdentity() const
    260 {
    261     glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ;
                                                                                ^^^^^^^^^^^^^^^^
    262     return id ;
    263 }

But constrained to 16 bit in photon flags::

    oxrap/cu/generate.cu

    234 
    235 #define FLAGS(p, s, prd) \
    236 { \
    237     p.flags.u.x = ( ((prd.boundary & 0xffff) << 16) | (s.identity.w & 0xffff) )  ;  \
    238     p.flags.u.y = s.identity.x ;  \
    239     p.flags.u.w |= s.flag ; \
    240 } \
    241 

    okc/OpticksPhotonFlags.cc

    085 int OpticksPhotonFlags::SensorIndex(const float& x, const float& , const float& , const float& ) // static
     86 {
     87     uif_t uif ;
     88     uif.f = x ;
     89     unsigned lo = uif.u & 0xffff  ;
     90     return lo <= 0x7fff  ? lo : lo - 0x10000 ;  // twos-complement see SPack::unsigned_as_int 
     91 }



Adaptng to 1-based sensorIndex
----------------------------------


Where sensorIndex used::

    epsilon:opticks blyth$ opticks-fl sensorIndex

    ./ggeo/GVolume.cc
    ./opticksgeo/SensorLib.hh
    ./opticksgeo/SensorLib.cc
    ./ggeo/GNodeLib.cc
    ./ggeo/GNodeLib.hh
    ./ggeo/GPho.cc
    ./optickscore/OpticksPhotonFlags.hh
    ./optickscore/OpticksPhotonFlags.cc
    ./optickscore/tests/OpticksPhotonFlagsTest.cc
    ./extg4/X4PhysicalVolume.cc
    ./sysrap/tests/SPackTest.cc
    ./ggeo/GGeoTest.cc
    ./optixrap/cu/generate.cu

          adapted to 1-based 

    ./ggeo/GGeo.hh
    ./ggeo/GGeo.cc 
    ./ggeo/GMergedMesh.cc
    ./g4ok/G4Opticks.cc
    ./g4ok/G4Opticks.hh

          unchanged, as just passes through/along  

    ./cfg4/CWriter.cc
    ./ggeo/GMesh.txt
          no change needed

    ./npy/HitsNPY.cpp
    ./ana/debug/genstep_sequence_material_mismatch.py
          looks obsolete/ancient  



Mysteriously the changes induces some opticks-t fails

* :doc:`G4StepNPY_checkGencodes_mismatch_assert.rst`




Need to communicate f_theta f_phi from closest hit to raygen via PRD
----------------------------------------------------------------------


Before adding, can PRD be slimmed ?::

     25 struct PerRayData_propagate
     26 {    
     27     float3 surface_normal ; 
     28     float distance_to_boundary ;
     29     int   boundary ; 
     30     uint4 identity ;
     31     float cos_theta ;
     32 };
     33      
     34 /**     
     35 
     36 surface_normal
     37     essential for propagate.h eg for reflection
     38 
     39 distance_to_boundary 
     40     rtIntersectionDistance is not available in raygen, so need in PRD 
     41     to pass from closest hit to raygen 
     42 
     43 boundary
     44     currently occupying 32 bits when 16 bits would be fine
     45 
     46 identity
     47     16 bytes, but so useful : in principal could just use 4 bytes of nodeIndex and look 
     48     up the identity from identity buffers 
     49 
     50 cos_theta
     51     sign is definitely needed, but is the value ? Actually the sign info may already be  
     52     carried in the sign of the 1-based boundary index 
     53 
     54     Value seems only used to special case normal incidence in propagate_at_boundary
     55 
     56 
     57 
     58 **/





::


     52 RT_PROGRAM void closest_hit_propagate()
     53 {
     54      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     55      float cos_theta = dot(n,ray.direction);

     //    cos_theta -1. ray direction opposite to outwards surface normal (impinging)
     //    cos_theta  1. ray direction parallel to outwards surface normal (coming from within) 
     //    cos_theta  0. ray direction perpendicular to normal : grazing incidence 


     56     
     57      prd.cos_theta = cos_theta ;
     58      prd.distance_to_boundary = t ;   // huh: there is an standard attrib for this
     59 
     60      unsigned boundaryIndex = ( instanceIdentity.z & 0xffff ) ;
     61      prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;
     62      prd.identity = instanceIdentity ; 
     63      prd.surface_normal = cos_theta > 0.f ? -n : n ;   
     64 
     65 //#define WITH_PRINT_IDENTITY_CH 1
     66 #ifdef WITH_PRINT_IDENTITY_CH
     67      rtPrintf("// material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (%8d %8d %8d %8d) \n", 
     68         instanceIdentity.x, 
     69         instanceIdentity.y, 
     70         instanceIdentity.z, 
     71         instanceIdentity.w) ;  
     72 
     73      rtPrintf("// material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (%8d %8d %8d %8d) \n", 
     74         prd.identity.x, 
     75         prd.identity.y, 
     76         prd.identity.z, 
     77         prd.identity.w) ;  
     78 #endif
     79 
     80 }



::

    403 
    404 __device__ void propagate_at_boundary( Photon& p, State& s, curandState &rng)
    405 {
    406     float eta = s.material1.x/s.material2.x ;    // eta = n1/n2   x:refractive_index  PRE-FLIPPED
    407 
    408     float3 incident_plane_normal = fabs(s.cos_theta) < 1e-6f ? p.polarization : normalize(cross(p.direction, s.surface_normal)) ;
    409 
    410     float normal_coefficient = dot(p.polarization, incident_plane_normal);  // fraction of E vector perpendicular to plane of incidence, ie S polarization
    41



Slimming PRD ? Can some of the full 16 bytes of identity be removed ?
------------------------------------------------------------------------

::

    s.identity.x   nodeIndex
    s.identity.y   tripletIdentity (a more meaningfull way to identify the volume, but duplicates nodeIndex)
    s.identity.z   shape (packed shape and boundary)
    s.identity.w   sensorIndex 


    235 #define FLAGS(p, s, prd) \
    236 { \
    237     p.flags.u.x = ( ((prd.boundary & 0xffff) << 16) | (s.identity.w & 0xffff) )  ;  \
    238     p.flags.u.y = s.identity.x ;  \
    239     p.flags.u.w |= s.flag ; \
    240 } \







