how_to_integrate_triangulated_geom_with_analytic
==================================================

Progress Overview
-------------------

1. DONE: unified SOPTIX_BuildInput.h base
2. DONE: use that in CSGOptix/SBT WITH_SOPTIX_ACCEL
3. DONE: found empty SScene in current geom, so kludged with SScene::Load 

4. trying all triangulated with CSGOptiX/SBT at stage of needing to 
   arrange SBT/hitgroup to make the ana/tri branch  

   * need type enum in hitgroup data that branches in the ptx
   * checking SDK examples, SDK/cuda/LocalGeometry.h:getLocalGeometry is closest  

   * need to incorporate some of sysrap/SOPTIX.cu into CSGOptiX/CSGOptiX7.cu and
     effect the switch 



Cutting Edge
---------------

* tri+ana WITH_SOPTIX_ACCEL now runs, but with TRIMESH configured get black render where triangles should be::

    TRIMESH=1 ~/o/CSGOptiX/cxr_min.sh


HOW TO DEBUG
----------------

Setup small (eg single G4Orb) CSGFoundry+SScene test geometry and do some PIDX ray trace debug dumping
comparing::

    TRIMESH=1 ~/o/CSGOptiX/cxr_min.sh
    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh 



SDK/cuda/GeometryData.h
--------------------------

::

    055 struct GeometryData
     56 {
     57     enum Type
     58     {
     59         TRIANGLE_MESH         = 0,
     60         SPHERE                = 1,
     61         LINEAR_CURVE_ARRAY    = 2,
     62         QUADRATIC_CURVE_ARRAY = 3,
     63         CUBIC_CURVE_ARRAY     = 4,
     64         CATROM_CURVE_ARRAY    = 5,
     65     };
     66 
     67     // The number of supported texture spaces per mesh.
     68     static const unsigned int num_textcoords = 2;
     69 
     70     struct TriangleMesh
     71     {
     72         GenericBufferView  indices;
     73         BufferView<float3> positions;
     74         BufferView<float3> normals;
     75         BufferView<Vec2f>  texcoords[num_textcoords]; // The buffer view may not be aligned, so don't use float2
     76         BufferView<Vec4f>  colors;                    // The buffer view may not be aligned, so don't use float4
     77     };
     78 
     79 
     80     struct Sphere
     81     {
     82         float3 center;
     83         float  radius;
     84     };
     85 
     86 
     87     struct Curves
     88     {
     89         BufferView<float2> strand_u;     // strand_u at segment start per segment
     90         GenericBufferView  strand_i;     // strand index per segment
     91         BufferView<uint2>  strand_info;  // info.x = segment base
     92                                          // info.y = strand length (segments)
     93     };
     94 
     95 
     96     Type  type;
     97 
     98     union
     99     {
    100         TriangleMesh triangle_mesh;
    101         Sphere       sphere;
    102         Curves       curves;
    103     };
    104 };



::

    [blyth@localhost SDK]$ find . -type f -exec grep -H GeometryData.h {} \;
    ./sutil/CMakeLists.txt:    ${SAMPLES_CUDA_DIR}/GeometryData.h
    ./cuda/whitted.h:#include <cuda/GeometryData.h>
    ./cuda/sphere.h:#include "GeometryData.h"
    ./cuda/LocalGeometry.h:#include <cuda/GeometryData.h>
    ./optixHair/optixHair.cu:#include <cuda/GeometryData.h>
    ./optixWhitted/optixWhitted.h:#include <cuda/GeometryData.h>
    [blyth@localhost SDK]$ 


cuda/whitted.h::

     44 struct HitGroupData
     45 {
     46     GeometryData geometry_data;
     47     MaterialData material_data;
     48 };

cuda/sphere.h::

     29 #pragma once
     30 
     31 #include "GeometryData.h"
     32 
     33 namespace sphere {
     34 
     35 const unsigned int NUM_ATTRIBUTE_VALUES = 4u;
     36 
     37 struct SphereHitGroupData
     38 {
     39     GeometryData::Sphere sphere;
     40 };
     41 
     42 }  // namespace sphere


cuda/LocalGeometry.h::

     59 SUTIL_HOSTDEVICE LocalGeometry getLocalGeometry( const GeometryData& geometry_data )
     60 {
     61     LocalGeometry lgeom;
     62     switch( geometry_data.type )
     63     {
     64         case GeometryData::TRIANGLE_MESH:
     65         {
     66             const GeometryData::TriangleMesh& mesh_data = geometry_data.triangle_mesh;
     67 
     68             const unsigned int prim_idx = optixGetPrimitiveIndex();
     69             const float2       barys    = optixGetTriangleBarycentrics();
     70 




::

    216 extern "C" __global__ void __closesthit__curve_strand_u()
    217 {
    218     const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    219 
    220     const whitted::HitGroupData* hitGroupData = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    221     const GeometryData&          geometryData = reinterpret_cast<const GeometryData&>( hitGroupData->geometry_data );
    222 
    223     const float3 normal     = computeNormal( optixGetPrimitiveType(), primitiveIndex );
    224     const float3 colors[2]  = {make_float3( 1, 0, 0 ), make_float3( 0, 1, 0 )};
    225     const float  u          = getStrandU( geometryData, primitiveIndex );
    226     const float3 base_color = colors[0] * u + colors[1] * ( 1 - u );
    227 
    228     const float3 hitPoint = getHitPoint();
    229     const float3 result   = shade( hitGroupData, hitPoint, normal, base_color );
    230 
    231     whitted::setPayloadResult( result );




SDK hitKind
------------

::

     622 /// Returns the 8 bit hit kind associated with the current hit.
     623 /// 
     624 /// Use optixGetPrimitiveType() to interpret the hit kind.
     625 /// For custom intersections (primitive type OPTIX_PRIMITIVE_TYPE_CUSTOM),
     626 /// this is the 7-bit hitKind passed to optixReportIntersection(). 
     627 /// Hit kinds greater than 127 are reserved for built-in primitives.
     628 ///
     629 /// Available only in AH and CH.
     630 static __forceinline__ __device__ unsigned int optixGetHitKind();
     631 
     632 /// Function interpreting the result of #optixGetHitKind().
     633 static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType( unsigned int hitKind );
     634 
     635 /// Function interpreting the result of #optixGetHitKind().
     636 static __forceinline__ __device__ bool optixIsFrontFaceHit( unsigned int hitKind );
     637 
     638 /// Function interpreting the result of #optixGetHitKind().
     639 static __forceinline__ __device__ bool optixIsBackFaceHit( unsigned int hitKind );
     640 
     641 /// Function interpreting the hit kind associated with the current optixReportIntersection.
     642 static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType();
     643 
     644 /// Function interpreting the hit kind associated with the current optixReportIntersection.
     645 static __forceinline__ __device__ bool optixIsFrontFaceHit();
     646 
     647 /// Function interpreting the hit kind associated with the current optixReportIntersection.
     648 static __forceinline__ __device__ bool optixIsBackFaceHit();
     649 
     650 /// Convenience function interpreting the result of #optixGetHitKind().
     651 static __forceinline__ __device__ bool optixIsTriangleHit();
     652 
     653 /// Convenience function interpreting the result of #optixGetHitKind().
     654 static __forceinline__ __device__ bool optixIsTriangleFrontFaceHit();
     655 
     656 /// Convenience function interpreting the result of #optixGetHitKind().
     657 static __forceinline__ __device__ bool optixIsTriangleBackFaceHit();
     658 


::

    [blyth@localhost include]$ find . -type f -exec grep -H GetPrimitiveType {} \;
    ./internal/optix_7_device_impl.h:static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType(unsigned int hitKind)
    ./internal/optix_7_device_impl.h:static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType()
    ./internal/optix_7_device_impl.h:    return optixGetPrimitiveType( optixGetHitKind() );
    ./optix_7_device.h:/// Use optixGetPrimitiveType() to interpret the hit kind.
    ./optix_7_device.h:static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType( unsigned int hitKind );
    ./optix_7_device.h:static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType();
    ./optix_7_types.h:/// It is preferred to use optixGetPrimitiveType(), together with
    [blyth@localhost include]$ 


    1174 static __forceinline__ __device__ unsigned int optixGetHitKind()
    1175 {
    1176     unsigned int u0;
    1177     asm( "call (%0), _optix_get_hit_kind, ();" : "=r"( u0 ) : );
    1178     return u0;
    1179 }
    1180 
    1181 static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType(unsigned int hitKind)
    1182 {
    1183     unsigned int u0;
    1184     asm( "call (%0), _optix_get_primitive_type_from_hit_kind, (%1);" : "=r"( u0 ) : "r"( hitKind ) );
    1185     return (OptixPrimitiveType)u0;
    1186 }
    1187 
    1188 static __forceinline__ __device__ bool optixIsBackFaceHit( unsigned int hitKind )
    1189 {
    1190     unsigned int u0;
    1191     asm( "call (%0), _optix_get_backface_from_hit_kind, (%1);" : "=r"( u0 ) : "r"( hitKind ) );
    1192     return (u0 == 0x1);
    1193 }
    1194 
    1195 static __forceinline__ __device__ bool optixIsFrontFaceHit( unsigned int hitKind )
    1196 {
    1197     return !optixIsBackFaceHit( hitKind );
    1198 }
    1199 
    1200 
    1201 static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType()
    1202 {
    1203     return optixGetPrimitiveType( optixGetHitKind() );
    1204 }
    1205 
    1206 static __forceinline__ __device__ bool optixIsBackFaceHit()
    1207 {
    1208     return optixIsBackFaceHit( optixGetHitKind() );
    1209 }
    1210 
    1211 static __forceinline__ __device__ bool optixIsFrontFaceHit()
    1212 {
    1213     return optixIsFrontFaceHit( optixGetHitKind() );
    1214 }
    1215 
    1216 static __forceinline__ __device__ bool optixIsTriangleHit()
    1217 {
    1218     return optixIsTriangleFrontFaceHit() || optixIsTriangleBackFaceHit();
    1219 }
    1220 
    1221 static __forceinline__ __device__ bool optixIsTriangleFrontFaceHit()
    1222 {
    1223     return optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE;
    1224 }
    1225 
    1226 static __forceinline__ __device__ bool optixIsTriangleBackFaceHit()
    1227 {
    1228     return optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE;
    1229 }
    1230 





OptixPrimitiveType
----------------------

::

     400 /// Builtin primitive types
     401 ///
     402 typedef enum OptixPrimitiveType
     403 {
     404     /// Custom primitive.
     405     OPTIX_PRIMITIVE_TYPE_CUSTOM                        = 0x2500,
     406     /// B-spline curve of degree 2 with circular cross-section.
     407     OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE       = 0x2501,
     408     /// B-spline curve of degree 3 with circular cross-section.
     409     OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE           = 0x2502,
     410     /// Piecewise linear curve with circular cross-section.
     411     OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR                  = 0x2503,
     412     /// CatmullRom curve with circular cross-section.
     413     OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM              = 0x2504,
     414     OPTIX_PRIMITIVE_TYPE_SPHERE                        = 0x2506,
     415     /// Triangle.
     416     OPTIX_PRIMITIVE_TYPE_TRIANGLE                      = 0x2531,
     417 } OptixPrimitiveType;
     418 
     419 /// Builtin flags may be bitwise combined.
     420 ///
     421 /// \see #OptixPipelineCompileOptions::usesPrimitiveTypeFlags
     422 typedef enum OptixPrimitiveTypeFlags
     423 {
     424     /// Custom primitive.
     425     OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM                  = 1 << 0,
     426     /// B-spline curve of degree 2 with circular cross-section.
     427     OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE = 1 << 1,
     428     /// B-spline curve of degree 3 with circular cross-section.
     429     OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE     = 1 << 2,
     430     /// Piecewise linear curve with circular cross-section.
     431     OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR            = 1 << 3,
     432     /// CatmullRom curve with circular cross-section.
     433     OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM        = 1 << 4,
     434     OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE                  = 1 << 6,
     435     /// Triangle.
     436     OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE                = 1 << 31,
     437 } OptixPrimitiveTypeFlags;
     438 



SBT hitgroup needs tri/ana generalization 
--------------------------------------------


sysrap/SOPTIX.cu::

    229 extern "C" __global__ void __closesthit__ch()
    230 {
    231     const SOPTIX_HitgroupData* hit_group_data = reinterpret_cast<SOPTIX_HitgroupData*>( optixGetSbtDataPointer() );
    232     const SOPTIX_TriMesh& mesh = hit_group_data->mesh ;
    233 
    234     //printf("//__closesthit__ch\n"); 
    235 
    236     const unsigned prim_idx = optixGetPrimitiveIndex();
    237     const float2   barys    = optixGetTriangleBarycentrics();

    /// BUILTIN TRI INTERSECT IS USED : SO NO __intersection__is



CSGOptiX/CSGOptiX7.cu::

    494 extern "C" __global__ void __closesthit__ch()
    495 {
    496     unsigned iindex = optixGetInstanceIndex() ;
    497     unsigned identity = optixGetInstanceId() ;
    498 
    499 #ifdef WITH_PRD
    500     quad2* prd = getPRD<quad2>();
    501 
    502     prd->set_identity( identity ) ;
    503     prd->set_iindex(   iindex ) ;
    504     float3* normal = prd->normal();
    505     *normal = optixTransformNormalFromObjectToWorldSpace( *normal ) ;
    506 
    ...

    541 extern "C" __global__ void __intersection__is()
    542 {    
    543     HitGroupData* hg  = (HitGroupData*)optixGetSbtDataPointer();
    544     int nodeOffset = hg->nodeOffset ; 
    545 
    546     const CSGNode* node = params.node + nodeOffset ;  // root of tree
    547     const float4* plan = params.plan ;
    548     const qat4*   itra = params.itra ;


CSGOptiX/Binding.h::

    020 struct HitGroupData   // effectively Prim 
     21 {
     22     int numNode ;   
     23     int nodeOffset ;
     24 };  
     25     
     26     
     27 #if defined(__CUDACC__) || defined(__CUDABE__)
     28 #else
     29 #include <optix_types.h>
     30     
     31 template <typename T>
     32 struct SbtRecord
     33 {
     34     __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
     35     T data;
     36 };
     37     
     38 typedef SbtRecord<RaygenData>     Raygen ;
     39 typedef SbtRecord<MissData>       Miss ;
     40 typedef SbtRecord<HitGroupData>   HitGroup ;
     41 
     42 #endif


sysrap/SOPTIX_Binding.h::

    012 template <typename T>
     13 struct SOPTIX_Record
     14 {   
     15     __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
     16     T data;
     17 };

     27 struct SOPTIX_TriMesh 
     28 {   
     29     uint3*  indice ;
     30     float3* vertex ; 
     31     float3* normal ;  
     32 };  
     33 


     34 struct SOPTIX_HitgroupData 
     35 {   
     36     SOPTIX_TriMesh mesh ;
     37 };  
     38     
     39     
     40 typedef SOPTIX_Record<SOPTIX_RaygenData>   SOPTIX_RaygenRecord;
     41 typedef SOPTIX_Record<SOPTIX_MissData>     SOPTIX_MissRecord;
     42 typedef SOPTIX_Record<SOPTIX_HitgroupData> SOPTIX_HitgroupRecord;
     43     










need to incorporate some of sysrap/SOPTIX.cu into CSGOptiX/CSGOptiX7.cu
--------------------------------------------------------------------------




What level of ana/tri split ? CSGSolid
----------------------------------------

1:1 CSGSolid:GAS

As each GAS must be either analytic or triangulated 
have to split at CSGSolid level. 

That means if have a G4VSolid (eg the guide tube torus) 
that must be triangulated then must arrange for the corresponding 
CSGPrim to be isolated into its own CSGSolid. 

Initially can just assert that selected CSGPrim must be isolated, 
as that will be the case for the guide tube. 


Recent addition triangulated geom is dev in sysrap/SOPTIX,SScene
--------------------------------------------------------------------

* :doc:`sysrap/SOPTIX`


High level geometry workflow
------------------------------


::

    227 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    228 {
    229     LOG(LEVEL) << "[ G4VPhysicalVolume world " << world ;
    230     assert(world);
    231     wd = world ;
    232 
    233     assert(sim && "sim instance should have been grabbed/created in ctor" );
    234     stree* st = sim->get_tree();
    235 
    236     tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    237     LOG(LEVEL) << "Completed U4Tree::Create " ;
    238 
    239     sim->initSceneFromTree(); // not so easy to do at lower level  
    240 
    241 
    242     CSGFoundry* fd_ = CSGFoundry::CreateFromSim() ; // adopts SSim::INSTANCE  
    243     setGeometry(fd_);
    244 
    245     LOG(info) << Desc() ;
    246 
    247     LOG(LEVEL) << "] G4VPhysicalVolume world " << world ;
    248 }



::

     079 /**
      80 CSGFoundry::CSGFoundry
      81 ------------------------
      82 
      83 HMM: the dependency between CSGFoundry and SSim is a bit mixed up
      84 because of the two possibilities:
      85 
      86 1. "Import" : create CSGFoundry from SSim/stree using CSGImport
      87 2. "Load"   : load previously created and persisted CSGFoundry + SSim from file system 
      88 
      89 sim(SSim) used to be a passive passenger of CSGFoundry but now that CSGFoundry 
      90 can be CSGImported from SSim it is no longer so passive. 
      91 
      92 **/
      93 
      94 CSGFoundry::CSGFoundry()
      95     :
      96     d_prim(nullptr),
      97     d_node(nullptr),
      98     d_plan(nullptr),
      99     d_itra(nullptr),
     100     sim(SSim::Get()),
     101     import(new CSGImport(this)),




Workflow : how to add tri ?
-------------------------------

SSim
   holds stree(ana) and SScene(tri)

CSGFoundry 
   has sim member giving access to both stree and SScene

CSGFoundry::CreateFromSim/CSGFoundry::importSim
   populates CSGFoundry from stree 


* HMM: simpler to have parallel ana+tri throughout the geometry workflow with the 
  ana/tri switch done at the GAS handle creation stage 

* ana at all stages is very small, so no resource issue, 
  tri could be large for the remainder instance : so want to 
  do ana/tri switch before GPU (hmm might not be so easy with SOPTIX)

  * this might need SOPTIX_MeshGroup reworking to defer uploads : unless
    just deferred usage of that until GAS-handle stage  
 


DONE : made a more vertical API for tri/ana integration
--------------------------------------------------------

::

   SOPTIX_MeshGroup* Create( OptixDeviceContext& ctx, const SMeshGroup* mg );

   SMeshGroup* mg = scene->meshgroup[i] ;  
   SOPTIX_MeshGroup* xmg = SOPTIX_MeshGroup::Create( ctx, mg ) ; 
   xmg->gas->handle  



NEXT: name based ana/tri control 
-------------------------------------



Analytic in stree/CSG/CSGOptiX 
---------------------------------

::

     551 void CSGOptiX::initGeometry()
     552 {
     553     LOG(LEVEL) << "[" ;
     554     params->node = foundry->d_node ;
     555     params->plan = foundry->d_plan ;
     556     params->tran = nullptr ;
     557     params->itra = foundry->d_itra ;
     558 
     559     bool is_uploaded =  params->node != nullptr ;
     560     LOG_IF(fatal, !is_uploaded) << "foundry must be uploaded prior to CSGOptiX::initGeometry " ;
     561     assert( is_uploaded );
     562 
     563 #if OPTIX_VERSION < 70000
     564     six->setFoundry(foundry);
     565 #else
     566     LOG(LEVEL) << "[ sbt.setFoundry " ;
     567     sbt->setFoundry(foundry);
     568     LOG(LEVEL) << "] sbt.setFoundry " ;
     569 #endif
     570     const char* top = Top();
     571     setTop(top);
     572     LOG(LEVEL) << "]" ;
     573 }


::

   CSGOptiX::initGeometry
   SBT::setFoundry
   SBT::createGeom
   SBT::createGAS_Standard



Where+how to ana/tri branch ?
-------------------------------

EMM is integer based.  Need name based gas_idx control for greater longevity. 

::

     261 void SBT::createGAS_Standard()
     262 {
     263     unsigned num_solid = foundry->getNumSolid();   // STANDARD_SOLID
     264     for(unsigned i=0 ; i < num_solid ; i++)
     265     {
     266         unsigned gas_idx = i ;
     267 
     268         bool enabled = SGeoConfig::IsEnabledMergedMesh(gas_idx) ;
     269         bool enabled2 = emm & ( 0x1 << gas_idx ) ;
     270         bool enabled_expect = enabled == enabled2 ;
     271         assert( enabled_expect );
     272         if(!enabled_expect) std::raise(SIGINT);
     273 
     274         if( enabled )
     275         {
     276             LOG(LEVEL) << " emm proceed " << gas_idx ;
     277             createGAS(gas_idx);
     278         }
     279         else
     280         {
     281             LOG(LEVEL) << " emm skip " << gas_idx ;
     282         }
     283     } 
     284     LOG(LEVEL) << descGAS() ;
     285 }  


Commonality between ana and tri is the handle
---------------------------------------------------

* HMM: SOPTIX side "gas" is SOPTIX_Accel instance
* WIP: maybe standardize by using the handle in the  vgas map ?

  * NOPE: NEED NUMBER OF buildInputs FOR SBT MECHANICS
  * added reference to the vector in SOPTIX_Accel MAYBE NEEDS TO BE pointer to vector on heap ?


::

   00305 void SBT::createGAS(unsigned gas_idx)
     306 {
     307     CSGPrimSpec ps = foundry->getPrimSpec(gas_idx);
     308     GAS gas = {} ;
     309     GAS_Builder::Build(gas, ps);
     310     vgas[gas_idx] = gas ;
     311 }

   0005 struct AS
      6 {
      7     CUdeviceptr             d_buffer;
      8     OptixTraversableHandle  handle ;
      9 };


* IAS_Builder::CollectInstances sets gas.handle into OptixInstance



Should CSGOptiX adopt some of SOPTIX ? 
---------------------------------------------

SOPTIX_Accel
    builds acceleration structure GAS or IAS from the buildInputs

    * could replace:: 

       GAS_Builder::BoilerPlate 
       IAS_Builder::Build


HMM: many of the CSGOptiX::initXXX and SBT.h PIP.h could be 
replaced by SOPTIX but not much motivation unless can show better
performance.  


Need to check perf as make such changes
------------------------------------------



