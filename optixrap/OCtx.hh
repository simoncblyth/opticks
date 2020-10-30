#pragma once 

/**
OCtx
-----

This is experimenting with a watertight wrapper for the underlying OptiX 
that strictly does not expose any OptiX types in its interface.
Where some identity is needed have adopted the cheat of using "void*". 
Approach inspired by C style opaque pointer

* https://en.wikipedia.org/wiki/Opaque_pointer


Adopting a pre-existing OptiX context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When adopting a pre-existing OptiX context created at C++ level into OCtx 
have to first peel back to the C-API in order to reach the bare pointer::

   optix::Context context = optix::Context::create(); 
   optix::ContextObj* contextObj = context.get(); 
   RTcontext context_ptr = contextObj->get(); 
   void* ptr = (void*)context_ptr ; 
   OCtx octx(ptr); 

   // in brief  
   OCtx octx((void*)context.get()->get()); 

Motivation
~~~~~~~~~~~~~

Attempting to hide the OptiX 6 to OptiX 7 API change is the eventual intention. 
But that is not going to be straightforward as API 7 is so different to API 6.  
Given the differences between the 6 and 7 APIs it makes little
sense to wrap at a low levels. Instead need to aim for a higher level 
wrapping, for example making an instance assembly in one call given an
array of transforms and the geometry pointer etc..
Will need to become proficient in 7 before can attempt to 
fill out an implementation for it.

Googling doesnt yield much help on this:

* :google:`C++ library wrapper that does not expose underlying types`
* :google:`C wrapper that does not expose underlying types`
* https://en.wikipedia.org/wiki/Facade_pattern

**/

#include <vector>
#include "plog/Severity.h"
#include "NGLM.hpp"
#include "OXRAP_API_EXPORT.hh"
class NPYBase ; 
template <typename T> class NPY ;

class OXRAP_API OCtx 
{
    private:
        static OCtx* INSTANCE ; 
        static const plog::Severity LEVEL ; 
    private:
        void* m_context_ptr ;    
    public:
        OCtx(void* context_ptr=NULL); 
        void* init(); 

        static OCtx* Get();

        void* ptr();
        bool has_variable( const char* key );
        void* create_buffer(const NPYBase* arr, const char* key, const char type, const char flag, int item, bool transpose ) const ;
        void* get_buffer( const char* key );
        void desc_buffer( void* buffer_ptr ); 
        void upload_buffer( const NPYBase* arr, void* buffer_ptr, int item ) const ;
        void download_buffer( NPYBase* arr, const char* key, int item);
        void set_raygen_program( unsigned entry_point_index, const char* ptx_path, const char* func );
        void set_exception_program( unsigned entry_point_index, const char* ptx_path, const char* func );
        void set_miss_program( unsigned entry_point_index, const char* ptx_path, const char* func );
        void* create_geometry(unsigned prim_count, const char* ptxpath, const char* bounds_func, const char* intersect_func );
        void* create_material(const char* ptxpath, const char* closest_hit_func, unsigned entry_point_index );
        void* create_geometryinstance(void* geo_ptr, void* mat_ptr);
        void* create_geometrygroup(const void* gi_ptr); 
        void* create_geometrygroup(const std::vector<const void*>& v_gi_ptr);
        void* create_acceleration( const char* accel );
        void set_geometrygroup_acceleration(void* geometrygroup_ptr, void* ac_ptr );
        void set_group_acceleration(void* group_ptr, void* ac_ptr );
        void set_geometrygroup_context_variable( const char* key, void* gg_ptr );
        void compile();
        void validate();
        void launch(unsigned entry_point_index, unsigned width, unsigned height, unsigned depth);
        void launch(unsigned entry_point_index, unsigned width, unsigned height);
        void launch(unsigned entry_point_index, unsigned width);
        void launch_instrumented( unsigned entry_point_index, unsigned width, unsigned height, double& t_prelaunch, double& t_launch  );
        unsigned create_texture_sampler( void* buffer_ptr, const char* config ) const ;
        void set_texture_param( void* buffer_ptr, unsigned tex_id, const char* param_key );
        unsigned upload_2d_texture(const char* param_key, const NPYBase* inp, const char* config, int item);
        void set_geometry_float4( void* geometry_ptr, const char* key, float x, float y, float z, float w );
        void set_geometry_float3( void* geometry_ptr, const char* key, float x, float y, float z);
        void set_context_float4( const char* key, float x, float y, float z, float w );
        void set_context_int4( const char* key, int x, int y, int z, int w );
        void set_context_int( const char* key, int x );
        void set_context_viewpoint( const glm::vec3& eye, const glm::vec3& U,  const glm::vec3& V, const glm::vec3& W, const float scene_epsilon );
        void* create_transform( bool transpose, const float* m44, const float* inverse_m44 );
        void* create_instanced_assembly( const NPY<float>* transforms, const void* geometry_ptr, const void* material_ptr, bool identity_from_transform_03 );
        void* create_single_assembly( const glm::mat4& m4, const void* geometry_ptr, const void* material_ptr, bool identity_from_transform_03);

        void* create_group( const char* key, const void* child_group_ptr );
        void group_add_child_group( void* group_ptr , void* child_group_ptr );
};


