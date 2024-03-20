optix7_sdk_notes
=================


sutil::Scene
-------------

* Scene is very GLTF influenced, but still has useful patterns to follow. 

  * stree.h acts as the Opticks equiv to a loaded GLTF 

* Scene mixes holding the GPU side geometry buffers with OptiX stuff

  * splitting those seem natural 
  * need equiv of the non-OptiX parts of sutil::Scene before can look at tri building 

    * (SScene.h using SMesh that does equiv of GLTF loading from the stree.h)


* HMM: regards to triangles can the buffers be shared between OpenGL and CUDA/OptiX ? 



sutil/Scene::
          
    053 class Scene
     54 {
     55 public:
     56     SUTILAPI Scene();
     57     SUTILAPI ~Scene();
     58 
     59     struct Instance
     60     {
     61         Matrix4x4                         transform;
     62         Aabb                              world_aabb;
     63 
     64         int                               mesh_idx;
     65     };
     66 
     67     struct MeshGroup
     68     {
     69         std::string                       name;
     70 
     71         std::vector<GenericBufferView>    indices;
     //                     BufferView<unsigned> 

     72         std::vector<BufferView<float3> >  positions;
     73         std::vector<BufferView<float3> >  normals;
     74         std::vector<BufferView<Vec2f> >   texcoords[GeometryData::num_textcoords];
     75         std::vector<BufferView<Vec4f> >   colors;

     // BufferView hold device pointers 
     // uploads happen at Scene::addBuffer
     //
     // HMM: does the MeshGroup avoid the need to mergemeshes for OptiX ? 
     // BUT mybe need that for OpenGL ? Or not, they are views into one buf after all.
     // That one buf could be used by OpenGL  

     76 
     77         std::vector<int32_t>              material_idx;
     78 
     79         OptixTraversableHandle            gas_handle = 0;
     80         CUdeviceptr                       d_gas_output = 0;
     81 
     82         Aabb                              object_aabb;
     83     };
    ...
    ...
    134     std::vector<Camera>                      m_cameras;

    135     std::vector<std::shared_ptr<Instance> >  m_instances;
    136     std::vector<std::shared_ptr<MeshGroup> > m_meshes;

    137     std::vector<MaterialData>                m_materials;
    138     std::vector<CUdeviceptr>                 m_buffers;        
    ///     direct from GLTF buffers : curious not to include size 

    139     std::vector<cudaTextureObject_t>         m_samplers;
    140     std::vector<cudaArray_t>                 m_images;
    141     sutil::Aabb                              m_scene_aabb;
    142 
    143     OptixDeviceContext                   m_context                  = 0;
    144     OptixShaderBindingTable              m_sbt                      = {};
    145     OptixPipelineCompileOptions          m_pipeline_compile_options = {};
    146     OptixPipeline                        m_pipeline                 = 0;
    147     OptixModule                          m_ptx_module               = 0;
    148 
    149     OptixProgramGroup                    m_raygen_prog_group        = 0;
    150     OptixProgramGroup                    m_radiance_miss_group      = 0;
    151     OptixProgramGroup                    m_occlusion_miss_group     = 0;
    152     OptixProgramGroup                    m_radiance_hit_group       = 0;
    153     OptixProgramGroup                    m_occlusion_hit_group      = 0;
    154     OptixTraversableHandle               m_ias_handle               = 0;
    155     CUdeviceptr                          m_d_ias_output_buffer      = 0;
    156 };




cuda/BufferView::

     Holds device pointer and provides strided and template typed element access

     33 template <typename T>
     34 struct BufferView
     35 {
     36     CUdeviceptr    data           CONST_STATIC_INIT( 0 );
     37     unsigned int   count          CONST_STATIC_INIT( 0 );
     38     unsigned short byte_stride    CONST_STATIC_INIT( 0 );
     39     unsigned short elmt_byte_size CONST_STATIC_INIT( 0 );
     40 
     41     SUTIL_HOSTDEVICE bool isValid() const
     42     { return static_cast<bool>( data ); }
     43 
     44     SUTIL_HOSTDEVICE operator bool() const
     45     { return isValid(); }
     46 
     47     SUTIL_HOSTDEVICE const T& operator[]( unsigned int idx ) const
     48     { return *reinterpret_cast<T*>( data + idx*(byte_stride ? byte_stride : sizeof( T ) ) ); }
     49 };
     50 
     51 typedef BufferView<unsigned int> GenericBufferView;
     52 




Cf with CSGOptiX 
--------------------

* hmm need to have something like the Scene created from stree before 
  can 


