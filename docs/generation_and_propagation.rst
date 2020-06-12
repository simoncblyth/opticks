Generation and propagation
=============================

The below provides annotated extracts (a readers digest) of crucial parts of 
some of the NVIDIA OptiX programs used by Opticks.  The intention is to show the broad outlines
of how the programs are used to implement the optical photon simulation. For details consult the sources. 


NVIDIA OptiX programs
------------------------- 

RG : Ray Generation 
    entry point into the ray tracing pipeline, invoked by the system in parallel for each user-defined work assignment
EX : Exception 
    invoked for conditions such as stack overflow and other errors
CH : Closest hit  
    Called when a traced ray finds the closest intersection point, such as for material shading
AH : Any hit 
    Called when a traced ray finds a new, potentially closest, intersection point, such as for shadow computation
    **not used by Opticks, disfavored for performance reasons**
IN : Intersection 
    Implements a ray-primitive intersection test, invoked during traversal
BB : Bounding box 
    Computes a primitive’s world space bounding box, called when the system builds a new acceleration structure over the geometry
MI : Miss 
    Called when a traced ray misses all scene geometry
AT : Attribute
    Called after intersection with a built-in triangle. Used to provide triangle-specific attributes to the any-hit and closest-hit program.
    **not used by Opticks**


Ray Generation
------------------

optixrap/cu/generate.cu::

    412 RT_PROGRAM void generate()
    413 {

    ///
    ///   Associate photon with its genstep, via seed buffer
    ///

    414     unsigned long long photon_id = launch_index.x ;
    421     unsigned int genstep_id = seed_buffer[photon_id] ;  
    427     unsigned int genstep_offset = genstep_id*GNUMQUAD ;
    428 
    429     union quad ghead ;           // union f:float4/i:int4/u:uint4
    430     ghead.f = genstep_buffer[genstep_offset+0];  
    431     int gencode = ghead.i.x ;   // integer bits from float buffer
    ...
    443     curandState rng = rng_states[photon_id];
    444 
    445     State s ;  
    446     Photon p ; 
    449

    /// 
    ///  Load CerenkovStep or ScintillationStep param from genstep buffer and generates a photon 
    ///

    450     if(gencode == CERENKOV)  
    451     {   
    452         CerenkovStep cs ;
    453         csload(cs, genstep_buffer, genstep_offset, genstep_id);
    457         generate_cerenkov_photon(p, cs, rng );
    458         s.flag = CERENKOV ;
    459     }
    460     else if(gencode == SCINTILLATION)
    461     {
    462         ScintillationStep ss ;
    463         ssload(ss, genstep_buffer, genstep_offset, genstep_id);
    467         generate_scintillation_photon(p, ss, rng );
    468         s.flag = SCINTILLATION ;
    469     }
    470     else if(gencode == TORCH)
    ...
    480     else if(gencode == EMITSOURCE)
    ...

    ///
    ///  Bounce loop : propagating around geometry 
    /// 

    514     int bounce = 0 ;
    ...
    544     PerRayData_propagate prd ;
    545 
    546     while( bounce < bounce_max )
    547     {
    552         bounce++;   // increment at head, not tail, as CONTINUE skips the tail
    553 
    554         // closest hit program sets these, see material1_propagate.cu:closest_hit_propagate
    555         prd.distance_to_boundary = -1.f ;
    558         prd.identity.z = 0 ; // boundaryIndex, 0-based 
    560         prd.boundary = 0 ;   // signed, 1-based
    561 
    564         rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );


    ///         Closest hit program (material1_propagate.cu:closest_hit_propagate) invoked by the ray trace
    ///         communicates back here to the ray generation program via the prd (PerRayData_propagate).


    565 
    566         if(prd.boundary == 0)
    567         {
    568             s.flag = MISS ;  // overwrite CERENKOV/SCINTILLATION for the no hitters
    574             break ;
    575         }  

    ///         fill_state 
    ///              uses the boundary index to lookup wavelength dependent material and surface properties
    ///              (eg scattering_length, absorption_length, reemission_prob, reflectivity) from the boundary texture.
    /// 
    ///         NB the above rtTrace is the only geometry query : 
    ///         this works as all properties necessary for the propagation 
    ///         are "hung" on the boundaries.
    ///

    579         fill_state(s, prd.boundary, prd.identity, p.wavelength );
    580 
    581         s.distance_to_boundary = prd.distance_to_boundary ;
    582         s.surface_normal = prd.surface_normal ;
    583         s.cos_theta = prd.cos_theta ;
    ...
    607         command = propagate_to_boundary( p, s, rng );
    608         if(command == BREAK)    break ;           // BULK_ABSORB
    609         if(command == CONTINUE) continue ;        // BULK_REEMIT/BULK_SCATTER
    610         // PASS : survivors will go on to pick up one of the below flags, 
    611 
    612

    ///
    ///         s.optical.x > 0 indicates there are surface properties (eg detect "EFFICIENCY") 
    ///         for this boundary 
    ///
    613         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    614         {
    615             command = propagate_at_surface(p, s, rng);
    616             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    617             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    618         }
    619         else
    620         {
    622             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    624         }
    625 
    626     }   // bounce < bounce_max



optixrap/cu/propagate.h::

    ///
    ///   Choosing history is simple when only a few possibilites.
    ///   The ray trace to find closest boundary is done at every step in order
    ///   to get the current material/surface properties in this material m1 and 
    ///   the next material m2.
    ///   


    078 __device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
    ...
    112     float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));   // .z:scattering_length
    113     float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));   // .y:absorption_length
    ...
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
    154     {
    ...



Boundary assignment during X4PhysicalVolume::convertStructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Boundaries are central to the Opticks geometry model which 
is boundary based, unlike Geant4 which is volume based.
Boundaries are formed during the X4PhysicalVolume::convertStructure recursive traversal
from a physical volume PV and its parent PV.

A GBnd instance holds four indices (omat, osur, isur, imat) representing.

* outer material
* outer surface
* inner surface
* inner material

Adding GBnd to GBndLib only returns a new boundary index if that GBnd has not been added previously.
All structural volumes (GVolume) have a boundary index assigned, and this boundary is 
passed through to the GPU geometry via the identityBuffer. 

The upshot is that at any ray trace intersect the boundary index is retrieved which allows rapid access
to material and surface properties via lookups on the 2d boundary texture, the dimensions being 
the wavelength and the boundary index.

::

    0880 void X4PhysicalVolume::convertStructure()
     881 {
     882     LOG(info) << "[ creating large tree of GVolume instances" ;
     ...
     889     const G4VPhysicalVolume* pv = m_top ;
     890     GVolume* parent = NULL ;
     891     const G4VPhysicalVolume* parent_pv = NULL ;
     892     int depth = 0 ;
     893     bool recursive_select = false ;
     ...
     898     m_root = convertStructure_r(pv, parent, depth, parent_pv, recursive_select );
     899 


     954 GVolume* X4PhysicalVolume::convertStructure_r(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const parent_pv, bool& recursive_select )
     955 {
     ...
     960      GVolume* volume = convertNode(pv, parent, depth, parent_pv, recursive_select );
     961 
     967      m_ggeo->add(volume); // collect in nodelib
     968 
     969      const G4LogicalVolume* const lv = pv->GetLogicalVolume();
     970  
     971      for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
     972      {
     973          const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
     974          convertStructure_r(child_pv, volume, depth+1, pv, recursive_select );
     975      }
     976 
     977      return volume   ;
     978 }


    1151 GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
    1152 {
    ....
    1159     unsigned boundary = addBoundary( pv, pv_p );
    ...
    1292     GVolume* volume = new GVolume(ndIdx, gtransform, mesh );
    ...
    1305     volume->setBoundary( boundary );
    1309 
    1310     volume->setLocalTransform(ltriple);
    1311     volume->setGlobalTransform(gtriple);
    ....
    1320     volume->setPVName( pvName.c_str() );
    1321     volume->setLVName( lvName.c_str() );
    ....
    1326     if(parent)
    1327     {
    1328          parent->addChild(volume);
    1329          volume->setParent(parent);
    1330     }
    ...
    1339     return volume ;
    1340 }



    0989 unsigned X4PhysicalVolume::addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p )
     990 {
     991     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
     992     const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;
     993 
     994     const G4Material* const imat_ = lv->GetMaterial() ;
     995     const G4Material* const omat_ = lv_p ? lv_p->GetMaterial() : imat_ ;  // top omat -> imat 
     996 
     997     const char* omat = X4::BaseName(omat_) ;
     998     const char* imat = X4::BaseName(imat_) ;
     ...
    1002     // look for a border surface defined between this and the parent volume, in either direction
    1003     bool first_priority = true ;
    1004     const G4LogicalSurface* const isur_ = findSurface( pv  , pv_p , first_priority );
    1005     const G4LogicalSurface* const osur_ = findSurface( pv_p, pv   , first_priority );
    ...
    1088     unsigned boundary = 0 ; 
    1089     if( g_sslv == NULL && g_sslv_p == NULL  )   // no skin surface on this or parent volume, just use bordersurface if there are any
    1090     {
    1091         const char* osur = X4::BaseName( osur_ ); 
    1092         const char* isur = X4::BaseName( isur_ ); 
    1093         boundary = m_blib->addBoundary( omat, osur, isur, imat ); 
    1094     }
    ....
    1112     return boundary ; 

    /// m_blib (ggeo/GBndLib)
    ///     GBndLib::addBoundary 
    ///          looks up indices of material and surfaces from the names,   
    ///          and stores 4 integers (omat,osur,isur,imat) returning a boundary 
    ///          index for each unique quadruplet of indices 
    ///          

              

Intersection
--------------

Calls to rtTrace traverse the BVH acceleration structure to find
bounding boxes that are intersected by the ray. For the closest 
of these the intersection 





Closest Hit
-------------


optixrap/cu/material1_propagate.cu::
 
     20 #include <optix.h>
     21 #include "PerRayData_propagate.h"
     22 #include "wavelength_lookup.h"
     23 
     24 //attributes set by TriangleMesh.cu:mesh_intersect 
     25 
     26 rtDeclareVariable(float3,  geometricNormal, attribute geometric_normal, );
     27 rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
     28 
     29 rtDeclareVariable(PerRayData_propagate, prd, rtPayload, );
     30 rtDeclareVariable(optix::Ray,           ray, rtCurrentRay, );
     31 rtDeclareVariable(float,                  t, rtIntersectionDistance, );
     32 
     33 
     34 RT_PROGRAM void closest_hit_propagate()
     35 {
     36      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     37      float cos_theta = dot(n,ray.direction);
     38 
     39      prd.cos_theta = cos_theta ;
     40      prd.distance_to_boundary = t ;   // huh: there is an standard attrib for this
     41      unsigned int boundaryIndex = instanceIdentity.z ;
     42      prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;
     43      prd.identity = instanceIdentity ;
     44      prd.surface_normal = cos_theta > 0.f ? -n : n ;
     45 }
     46
     47 // prd.boundary 
     48 //    * 1-based index with cos_theta signing, 0 means miss
     49 //    * sign identifies which of inner/outer-material is material1/material2 
     50 //    * by virtue of zero initialization, a miss leaves prd.boundary at zero
     51 //
     52 //  cos_theta > 0.f
     53 //        outward going photons, with p.direction in same hemi as the geometry normal
     54 //
     55 //  cos_theta < 0.f  
     56 //        inward going photons, with p.direction in opposite hemi to geometry normal
     57 //
     58 // surface_normal oriented to point from material2 back into material1 
     59 //


optixrap/OGeo.cc::

     506 optix::Material OGeo::makeMaterial()
     507 {
     ...
     513     optix::Material material = m_context->createMaterial();
     514     material->setClosestHitProgram(OContext::e_radiance_ray, m_ocontext->createProgram("material1_radiance.cu", "closest_hit_radiance"));
     515     material->setClosestHitProgram(OContext::e_propagate_ray, m_ocontext->createProgram("material1_propagate.cu", "closest_hit_propagate"));
     516     return material ;
     517 }

Opticks uses only a single optix::Material, that is associated to the closest hit program in OGeo::makeMaterial.
Renderers typically use optix::Material is to "shade" the appearance of different geometry depending 
on material type, eg wood, metal, plastic, etc..

As Opticks needs only the distance to the intersection and surface normal at the intersection, 
there is no need for multiple optix::Material.  The different properties of materials and surfaces
are carried in the boundary index. 






