triangulated_optional_geometry
===============================


Whats needed ?
-----------------

high level approach
~~~~~~~~~~~~~~~~~~~~~

* mesh/triangles(vertices+indices) need to be at GAS level 
  and use instancing transforms, avoiding repetition


optional control of mesh usage in geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* need user input control that opts for triangles for particular GAS (or LVID) ? 

  * BUT: the GAS splits are the result of factorization, so in principal
    do not know the indices to pick beforehand 
  * having triangle opted LVID could force solid into separate GAS


creating and persisting mesh triangles (vertices+indices) : DONE: FIRST ATTEMPT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* CURRENTLY NOT SUPPRESSING POINTER IN SOLID NAMES FOR NPFOLD KEYS ? 
  POTENTIALLY JUST ISSUE FOR TESTING THAT RUNS FROM GDML NOT LIVE GEOMETRY

* U4Tree/stree needs to use Geant4 polygonization (U4Mesh with U4MESH_EXTRA) 
  to persist the triangles(vertices+indices) (can do this for all solids, as not uploaded)

  * each solid yields an NPFold, eg key name sWorld
    and place to hang the NPFold maybe: "SSim/stree/mesh/sWorld/..." 

  * U4Tree::initSolid looks the place to use U4Mesh::MakeFold


uploading mesh triangles/vertices 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* CSGFoundry model (as it is uploaded) needs to optionally upload only needed triangles
  (vertices+indices), and hold onto device pointers/counts/offsets etc 
  in "CSGMeshSpec" analogous to "CSGPrimSpec"

HMM: SDK/optixMeshViewer uploads just one buffer obtained from GLTF and uses BufferViews 
to access whats needed with appropropriate offsets : makes sense to do similar, but 
manually as only need vertices+indices ?

CSGPrimSpec::

    epsilon:opticks blyth$ opticks-fl CSGPrimSpec
    ./CSGOptiX/GAS_Builder.cc
    ./CSGOptiX/GAS_Builder.h
    ./CSGOptiX/SBT.cc
         SBT::createGAS gets CSGPrimSpec from CSGFoundry 
         and uses GAS_Builder::Build to create GAS

    ./CSG/CSGPrimSpec.cc

    ./CSG/CSGPrim.cc
         CSGPrim::MakeSpec creates CSGPrimSpec

    ./CSG/CSGFoundry.h
         CSGFoundry::getPrimSpecDevice uses above MakeSpec, 
         d_prim with CSGSolid::primOffset CSGSolid::numPrim


HMM: is there need for CSGMesh as well as CSGMeshSpec ? 
Or could add method to CSGSolid/CSGPrim ?

HMM: need to assume all CSGPrim in the CSGSolid 
make the same analytic/triangulated choice, need to enforce that ? 



create OptiX acceleration structures using the mesh triangles/vertices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

* CSGOptiX needs to use "CSGMeshSpec" to construct triangulated GAS in GAS_Builder


how to combine customPrimitives with triangulated ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Seems no __intersection__ function needed for builtin triangles::

    epsilon:SDK blyth$ find . -type f -exec grep -Hl __intersection__ {} \;
    ./optixSimpleMotionBlur/optixSimpleMotionBlur.cu
    ./optixSimpleMotionBlur/optixSimpleMotionBlur.cpp
    ./optixCutouts/optixCutouts.cpp
    ./optixCallablePrograms/optixCallablePrograms.cpp
    ./cuda/sphere.cu
    ./optixWhitted/optixWhitted.cpp
    ./optixWhitted/geometry.cu
    ./optixDemandTexture/optixDemandTexture.cu
    ./optixDemandTexture/optixDemandTexture.cpp
    ./optixVolumeViewer/optixVolumeViewer.cpp
    ./optixVolumeViewer/volume.cu
    ./optixDynamicMaterials/optixDynamicMaterials.cu
    ./optixDynamicMaterials/optixDynamicMaterials.cpp
    ./optixCustomPrimitive/optixCustomPrimitive.cpp
    epsilon:SDK blyth$ 




intersection with the triangulated geometry 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* how to branch on geometry type 


/Developer/OptiX_750/SDK/cuda/whitted.cu::

    152 extern "C" __global__ void __closesthit__radiance()
    153 {
    154     const whitted::HitGroupData* hit_group_data = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    155     const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );



/Developer/OptiX_750/SDK/cuda/whitted.h::

     44 struct HitGroupData
     45 {
     46     GeometryData geometry_data;
     47     MaterialData material_data;
     48 };
        

/Developer/OptiX_750/SDK/cuda/GeometryData.h::

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
    ...        
     96     Type  type;
     97 
     98     union
     99     {
    100         TriangleMesh triangle_mesh;
    101         Sphere       sphere;
    102         Curves       curves;
    103     };
    104 };


/Developer/OptiX_750/SDK/cuda/LocalGeometry.h::

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
     71             uint3 tri = make_uint3(0u, 0u, 0u);
     72             if( mesh_data.indices.elmt_byte_size == 4 )
     73             {
     74                 const uint3* indices = reinterpret_cast<uint3*>( mesh_data.indices.data );
     75                 tri = indices[ prim_idx ];
     76             }
     77             else if( mesh_data.indices.elmt_byte_size == 2 )
     78             {
     79                 const unsigned short* indices = reinterpret_cast<unsigned short*>( mesh_data.indices.data );
     80                 const unsigned short  idx0    = indices[prim_idx * 3 + 0];
     81                 const unsigned short  idx1    = indices[prim_idx * 3 + 1];
     82                 const unsigned short  idx2    = indices[prim_idx * 3 + 2];
     83                 tri                           = make_uint3( idx0, idx1, idx2 );
     84             }
     85             else
     86             {
     87                 const unsigned int base_idx = prim_idx * 3;
     88                 tri = make_uint3( base_idx + 0, base_idx + 1, base_idx + 2 );
     89             }
     90 
     91             const float3 P0 = mesh_data.positions[ tri.x ];
     92             const float3 P1 = mesh_data.positions[ tri.y ];
     93             const float3 P2 = mesh_data.positions[ tri.z ];
     94             lgeom.P = ( 1.0f-barys.x-barys.y)*P0 + barys.x*P1 + barys.y*P2;
     95             lgeom.P = optixTransformPointFromObjectToWorldSpace( lgeom.P );
     96 


::

    epsilon:SDK blyth$ optix7-f optixGetTriangleBarycentrics
    ./optixCutouts/optixCutouts.cu:        const float2 barycentrics    = optixGetTriangleBarycentrics();
    ./optixTriangle/optixTriangle.cu:    const float2 barycentrics = optixGetTriangleBarycentrics();
    ./cuda/LocalGeometry.h:            const float2       barys    = optixGetTriangleBarycentrics();
    ./optixNVLink/optixNVLink.cu:        const float2 barycentrics = optixGetTriangleBarycentrics();
    epsilon:SDK blyth$ 





