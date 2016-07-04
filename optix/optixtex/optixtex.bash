# === func-gen- : optix/optixtex/optixtex fgp optix/optixtex/optixtex.bash fgn optixtex fgh optix/optixtex
optixtex-src(){      echo optix/optixtex/optixtex.bash ; }
optixtex-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(optixtex-src)} ; }
optixtex-vi(){       vi $(optixtex-source) ; }
optixtex-env(){      olocal- ; }
optixtex-usage(){ cat << EOU

OptiX Textures
===============

Have to think thru entire trace to determine appropriate repr
---------------------------------------------------------------

* http://bps11.idav.ucdavis.edu/talks/12-userDefinedRayTracingPipelines-Parker-BPS2011.pdf

::

    RT_PROGRAM void mesh_intersect(int primIdx) {

    ￼  uint3 v_idx = index_buffer[primIdx];   // look up face indices

       float3 p0 = vertex_buffer[v_idx].p0;   // vertices of the triangle
       float3 p1 = vertex_buffer[v_idx].p1
       float3 p2 = vertex_buffer[v_idx].p2;

       // Intersect ray with triangle 
       float3 e0 = p1 - p0;
       float3 e1 = p0 - p2;


Compare with chroma/cuda/photon.h::

    174 __device__ void
    175 fill_state(State &s, Photon &p, Geometry *g)
    176 {
    177 
    178     p.last_hit_triangle = intersect_mesh(p.position, p.direction, g,
    179                                          s.distance_to_boundary,
    180                                          p.last_hit_triangle);
    181 
    182     if (p.last_hit_triangle == -1) {
    183         s.material1_index = 999;
    184         s.material2_index = 999;
    185         p.history |= NO_HIT;
    186         return;
    187     }
    188 
    189     Triangle t = get_triangle(g, p.last_hit_triangle);
    190 
    191     unsigned int material_code = g->material_codes[p.last_hit_triangle];
    192 
    193     int inner_material_index = convert(0xFF & (material_code >> 24));
    194     int outer_material_index = convert(0xFF & (material_code >> 16));
    195     s.surface_index = convert(0xFF & (material_code >> 8));
    196 


Tempting to use material/surface index buffers in OptiX but, OptiX
has geometry/material association via geometry instance ? But for use 
of a single mesh with multiple substances will need a substance buffer 
to pluck the appropriate for the triangle.


OptiX Materials very different cf Chroma
------------------------------------------

OptiX materials

* associated closestHit program (shader)
* context containing any buffers, textures, 

Contrast with Chroma materials, which are dumb
struct instances.


Multi Material Examples
--------------------------

finding multi material geometryInstance usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:OptiX_370b2_sdk blyth$ find . -name '*.cu' -exec grep -H rtReportIntersection {} \; | grep -v 0

    ./cuda/triangle_mesh.cu:      rtReportIntersection(material_buffer[primIdx]);
    ./displacement/geometry_programs.cu:        rtReportIntersection(material_buffer[primIdx]);
    ./glass/triangle_mesh_iterative.cu:      rtReportIntersection(material_buffer[primIdx]);
    ./hybridShadows/triangle_mesh_fat.cu:      rtReportIntersection(material_buffer[primIdx]);
    ./isgReflections/triangle_mesh_fat.cu:      rtReportIntersection(material_buffer[primIdx]);
    ./isgShadows/triangle_mesh_fat.cu:      rtReportIntersection(material_buffer[primIdx]);
    ./primitiveIndexOffsets/triangle_mesh.cu:      rtReportIntersection(material_buffer[primIdx]);
    ./progressivePhotonMap/triangle_mesh.cu:      rtReportIntersection(material_buffer[primIdx]);
    ./swimmingShark/sphere_list.cu:      if(rtReportIntersection(material_buffer[primIdx]))
    ./swimmingShark/sphere_list.cu:        rtReportIntersection(material_buffer[primIdx]);
    ./whirligig/sphere_list.cu:      if(rtReportIntersection(material_buffer[primIdx]))
    ./whirligig/sphere_list.cu:        rtReportIntersection(material_buffer[primIdx]);

    ./shadeTree/sphere_array.cu:      if(rtReportIntersection( primIdx % (int)material_count ))
    ./shadeTree/sphere_array.cu:        rtReportIntersection( primIdx % (int)material_count );


material_buffer[primIdx]
~~~~~~~~~~~~~~~~~~~~~~~~~~~


sutil/OptixMeshImpl.cpp::

    769 optix::Buffer OptixMeshImpl::getGeoMaterialBuffer( optix::Geometry geo )
    770 {
    771   return geo["material_buffer"]->getBuffer();
    772 }
    773 
    774 void OptixMeshImpl::setGeoMaterialBuffer( optix::Geometry geo, optix::Buffer buf )
    775 {
    776   geo["material_buffer"]->setBuffer( buf );
    777 }
    ... 
    ...
    ...
    311     optix::Buffer mbuffer
    312       = m_mesh.m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT,
    313                                         group.num_triangles );
    314 
    315     group_geo_info.mbuffer_data
    316       = static_cast<unsigned int*>( mbuffer->map() );
    317     for( int j = 0; j < group.num_triangles; ++j ) {
    318       group_geo_info.mbuffer_data[j] = 0; // See above TODO
    319                                           //   What TODO? HAS LORE BEEN LOST??
    320     }
    321     mbuffer->unmap();
    322     group_geo_info.mbuffer_data = 0;
    323 
    ...
    ...
    ...
    333     m_mesh.setGeoMaterialBuffer( geo, mbuffer );



/usr/local/env/cuda/OptiX_370b2_sdk/sutil/MeshBash.h::

    121 typedef std::map<std::string, MeshGroup> MeshGroupMap;
    ...
    188   template <class Functor>
    189   void forEachGroup( Functor functor ) const;
    ...
    425 template <class Functor>
    426 void MeshBase::forEachGroup( Functor functor ) const
    427 {
    428   MeshGroupMap::const_iterator groups_end = m_mesh_groups.end();
    429   for( MeshGroupMap::const_iterator groups_iter = m_mesh_groups.begin();
    430        groups_iter != groups_end; )
    431   {
    432     functor( (groups_iter++)->second );
    433       // The post-increment ++ ensures the iterator is updated before functor()
    434       // runs, since functor may invalidate the iterator
    435   }
    436 }



/usr/local/env/cuda/OptiX_370b2_sdk/shadeTree/
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different material for each sphere, shadeTree.cpp:: 

    398   Program sphere_shader = m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "materials.cu" ), "shade_tree_material" );
    ...
    418   uint nmaterials = nspheres.x * nspheres.y;
    419   if (nmaterials > 1000)  nmaterials = 1000;
    420   std::vector< Material > sphere_mats( nmaterials );
    421   for( uint i=0; i < nmaterials; ++i ) {
    422     sphere_mats[i] = m_context->createMaterial();
    423     sphere_mats[i]->setClosestHitProgram( 0, sphere_shader );
    424     const int prog_id = i % m_colorPrograms.size();
    425     sphere_mats[i]["colorShader"]->set( m_colorPrograms[prog_id]);
    426     sphere_mats[i]["normalShader"]->set( m_normalPrograms[ m_normalProgId[prog_id] ]);
    ///
    ///     programs can be lodged into variables of the material : allowing "composite"  shading
    ///
    427   }
    428   sphere["material_count"]->setUint( nmaterials );
    ...
    ...
    475   // Place geometry into hierarchy
    476   std::vector<GeometryInstance> gis;
    477   gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );
    478   gis.push_back( m_context->createGeometryInstance( sphere,        sphere_mats.begin(), sphere_mats.end() ) );


Same shader code for each material, but very different effect due to 
the composable colorShader and normalShader. 

sphere_array.cu::

    059 void intersect_sphere( float4 sphere, int primIdx )
    060 {
    ...
    101       if(rtReportIntersection( primIdx % (int)material_count ))


Hand off to the appropriate material closest hit material program+context 
depends on the material index reported in the intersect code.


Multiple Texture Handling
----------------------------

Optix 3.0 has Bindless Textures as a new feature
The rayDifferentials sample shows how to do this:

* https://devtalk.nvidia.com/default/topic/523651/optix/best-way-to-store-multiple-textures-/

::

    TextureSampler samplers[16];
    ...
    Buffer tex_ids = matl->getContext()->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_INT, num_levels);
    m_context["tex_ids"]->set( tex_ids );
    int *ids = (int*)tex_ids->map();
    for(int l = 0; l < num_levels; ++l) ids[l] = samplers[l]->getId();
    tex_ids->unmap();


TextureSampler setup
---------------------

Configure sampler, associate buffer ../sutil/HDRLoader.cpp::

    226 optix::TextureSampler loadHDRTexture( optix::Context context,
    227                                       const std::string& filename,
    228                                       const optix::float3& default_color )
    229 {


Sampler/Material/Geometry association
---------------------------------------

* samplers associated to material via tex_ids and params lodged as variables of the material
* material associated to geometry via createGeometryInstance

rayDifferentials.cpp::

    332   // Floor geometry
    333   std::string pgram_ptx( ptxpath( "rayDifferentials", "parallelogram_differentials.cu" ) );
    334   Geometry parallelogram = m_context->createGeometry();
    335   parallelogram->setPrimitiveCount( 1u );
    336   parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( pgram_ptx, "bounds" ) );
    337   parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( pgram_ptx, "intersect" ) );
    ...
    383   // Checker material for floor
    384   Program check_ch = m_context->createProgramFromPTXFile( ptxpath( "rayDifferentials", "flat_tex_mip.cu" ), "closest_hit_radiance" );
    385   Program check_ah = m_context->createProgramFromPTXFile( ptxpath( "rayDifferentials", "flat_tex_mip.cu" ), "any_hit_shadow" );
    386   Material floor_matl = m_context->createMaterial();
    387   floor_matl->setClosestHitProgram( 0, check_ch );
    388   floor_matl->setAnyHitProgram( 1, check_ah );
    390   floor_matl["Kd1"]->setFloat( 0.8f, 0.3f, 0.15f);
    ...
    400   floor_matl["reflectivity2"]->setFloat( 0.0f, 0.0f, 0.0f);
    401 
    402   loadTextures(floor_matl);
    403
    404   // Create GIs for each piece of geometry
    405   std::vector<GeometryInstance> gis;
    ...
    410   gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl,  &floor_matl+1 ) );
    411 


Samplers into material context variable 
-----------------------------------------

::

    225 void RayDifferentialsScene::loadTextures(Material matl)
    226 {
    227   TextureSampler tex0 = loadTexture(matl->getContext(), m_tex_filename, make_float3(0.f));
    228   RTsize width0, height0;
    229   tex0->getBuffer(0,0)->getSize(width0, height0);
    230 
    231   tex0->setWrapMode( 0, RT_WRAP_REPEAT );
    232   tex0->setWrapMode( 1, RT_WRAP_REPEAT );
    233 
    ...
    245   matl["tex0_dim"]->setInt((int)width0, (int)height0);
    246 
    247   int num_levels = 1;
    248   int dim = (int)width0;
    249   matl["tex0"]->set(tex0);
    250   Buffer previous_tex = tex0->getBuffer(0,0);
    251  
    252   TextureSampler samplers[16];
    253   samplers[0] = tex0;
    254 
    255   do {
    256     dim >>= 1;
    257 
    258     TextureSampler sampler = matl->getContext()->createTextureSampler();
    ...
    301     sampler->setBuffer(0,0,buffer);
    302     samplers[num_levels] = sampler;
    303 
    304     num_levels++;
    305     previous_tex = buffer;
    306   } while(dim != 1);
    307 
    308   matl["num_mip_levels"]->setInt(num_levels);
    309 
    310   // Get texture IDs for all levels and store them in a buffer so we can use them on the device
    311   Buffer tex_ids = matl->getContext()->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_INT, num_levels);
    312   m_context["tex_ids"]->set( tex_ids );
    313   int *ids = (int*)tex_ids->map();
    314   for(int l = 0; l < num_levels; ++l)
    315     ids[l] = samplers[l]->getId();
    316 
    317   tex_ids->unmap();
    318 }


Geometry Instances with multiple materials
--------------------------------------------

OptiX_Programming_Guide_3.7.0.pdf p26:

    The number of materials that must be assigned to a geometry instance is
    determined by the highest material index that may be reported by an
    intersection program of the referenced geometry.

p46::

    __device__ bool rtReportIntersection( unsigned int material )

    This function takes an unsigned int specifying the index of a material 
    that must be associated with an any hit and closest hit program. 
    This material index can be used to support primitives of several different 
    materials flattened into a single Geometry object. 
    Traversal then immediately invokes the corresponding any hit program.

Access on device
------------------

Within the closest hit, there is no hunting for the
material its already associated with the material variables in scope.
The work of finding the appropriate material is done already 
in the intersection program.

rayDifferentials/flat_tex_mip.cu::

    051 rtDeclareVariable(float3, texcoord, attribute texcoord, );
    ...
    057 rtDeclareVariable(int, num_mip_levels, , );
    058 rtDeclareVariable(int2, tex0_dim, , );
    059 rtDeclareVariable(float2, tex_scale, , );
    060 
    061 
    062 rtBuffer<int, 1> tex_ids;
    ...
    074 static __inline__ __device__ float4 get_color(int i)
    075 {
    ...
    099     float2 uv = make_float2(texcoord)*tex_scale;
    100     return rtTex2D<float4>(tex_ids[i], uv.x, uv.y);
    ...  
    102 }
    105 RT_PROGRAM void closest_hit_radiance()
    106 {
    ...
    143   float4 color;
    144   if(lod >= num_mip_levels-1) {
    145     color = get_color(num_mip_levels-1);
    146   } else if( lod <= 0 ) {
    147     color = get_color(0);
    148   } else {
    149     int lod_lo = floorf(lod);
    150     int lod_hi = ceilf(lod);
    151     float t = lod - (float)((int)lod);
    152     color = (1.f-t)*get_color(lod_lo) + t*get_color(lod_hi);
    153   }
    154 
    155   prd_radiance.result = make_float3(color);
    156 }





EOU
}
optixtex-dir(){ echo $(local-base)/env/optix/optixtex/optix/optixtex-optixtex ; }
optixtex-cd(){  cd $(optixtex-dir); }
optixtex-mate(){ mate $(optixtex-dir) ; }
optixtex-get(){
   local dir=$(dirname $(optixtex-dir)) &&  mkdir -p $dir && cd $dir

}
