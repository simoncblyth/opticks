# === func-gen- : optix/optixtex/optixtex fgp optix/optixtex/optixtex.bash fgn optixtex fgh optix/optixtex
optixtex-src(){      echo optix/optixtex/optixtex.bash ; }
optixtex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optixtex-src)} ; }
optixtex-vi(){       vi $(optixtex-source) ; }
optixtex-env(){      elocal- ; }
optixtex-usage(){ cat << EOU

OptiX Textures
===============


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
