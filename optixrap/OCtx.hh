#pragma once 

/**
OCtx
-----

Try to create a watertight wrapper for the underlying OptiX 
that strictly does not expose any OptiX types in its interface.
Where some identity is needed have adopted the cheat of using "void*". 
Approach inspired by C style opaque pointer

* https://en.wikipedia.org/wiki/Opaque_pointer


Googling doesnt yield much help on this:

* :google:`C++ library wrapper that does not expose underlying types`
* :google:`C wrapper that does not expose underlying types`
* https://en.wikipedia.org/wiki/Facade_pattern

Clearly attempting to hide the OptiX 6 to OptiX 7 API change
is the intention. But that is not going to be straightforward as
API 7 is so different to API 6.  

Given the differences between the 6 and 7 APIs it makes little
sense to wrap at a low levels. Instead need to aim for a higher level 
wrapping, for example making an instance assembly in one call given an
array of transforms and the geometry pointer etc..

Will need to become proficient in 7 before can attempt to 
fill out an implementation for it.

**/

#include <vector>
#include "NGLM.hpp"
#include "OXRAP_API_EXPORT.hh"

struct OCtx ; 
class NPYBase ; 

OCtx* OXRAP_API OCtx_(); 
void* OXRAP_API OCtx_get() ;
bool  OXRAP_API OCtx_has_variable(const char* key);

void* OXRAP_API OCtx_create_buffer(const NPYBase* arr, const char* key, const char type, const char flag, int item=-1); 
void* OXRAP_API OCtx_get_buffer(const char* key); 
void  OXRAP_API OCtx_desc_buffer(void* buffer_ptr); 
void  OXRAP_API OCtx_upload_buffer(const NPYBase* arr, void* buffer_ptr, int item=-1); 
void  OXRAP_API OCtx_download_buffer(    NPYBase* arr, const char* key, int item=-1); 

void  OXRAP_API OCtx_set_raygen_program(    unsigned entry_point_index, const char* ptx_path, const char* func );
void  OXRAP_API OCtx_set_miss_program(      unsigned entry_point_index, const char* ptx_path, const char* func );
void  OXRAP_API OCtx_set_exception_program( unsigned entry_point_index, const char* ptx_path, const char* func );

void* OXRAP_API OCtx_create_geometry(unsigned prim_count, const char* ptxpath, const char* bounds_func, const char* intersect_func ); 
void* OXRAP_API OCtx_create_material(const char* ptxpath, const char* closest_hit_func, unsigned entry_point_index ); 
void* OXRAP_API OCtx_create_geometryinstance(void* geo_ptr, void* mat_ptr ); 
void* OXRAP_API OCtx_create_geometrygroup(const std::vector<void*>& v_gi_ptr); 
void* OXRAP_API OCtx_create_acceleration( const char* accel );
void  OXRAP_API OCtx_set_geometrygroup_acceleration(void* gg_ptr, void* ac_ptr );
void  OXRAP_API OCtx_set_group_acceleration(void* group_ptr, void* ac_ptr );
void  OXRAP_API OCtx_set_geometrygroup_context_variable( const char* key, void* gg_ptr ); 

void  OXRAP_API OCtx_compile(); 
void  OXRAP_API OCtx_validate(); 
void  OXRAP_API OCtx_launch(unsigned entry_point_index, unsigned width, unsigned height, unsigned depth); 
void  OXRAP_API OCtx_launch(unsigned entry_point_index, unsigned width, unsigned height); 
void  OXRAP_API OCtx_launch(unsigned entry_point_index, unsigned width); 
void  OXRAP_API OCtx_launch_instrumented(unsigned entry_point_index, unsigned width, unsigned height, double& t_prelaunch, double& t_launch); 

unsigned OXRAP_API OCtx_create_texture_sampler( void* buffer_ptr, const char* config );
void     OXRAP_API OCtx_set_texture_param(      void* buffer_ptr, unsigned tex_id, const char* param_key ); 
void     OXRAP_API OCtx_upload_2d_texture_layered(const char* param_key, const NPYBase* inp, const char* config, int item=-1);

void  OXRAP_API OCtx_set_geometry_float4(void* geometry_ptr, const char* key, float x, float y, float z, float w );
void  OXRAP_API OCtx_set_geometry_float3(void* geometry_ptr, const char* key, float x, float y, float z);

void  OXRAP_API OCtx_set_context_float4(const char* key, float x, float y, float z, float w );
void  OXRAP_API OCtx_set_context_int4(const char* key, int x, int y, int z, int w );
void  OXRAP_API OCtx_set_context_viewpoint( const glm::vec3& eye, const glm::vec3& U,  const glm::vec3& V, const glm::vec3& W, const float scene_epsilon ); 

void* OXRAP_API OCtx_create_transform( bool transpose, const float* m44, const float* inverse_m44 ); 
void* OXRAP_API OCtx_create_instanced_assembly( NPYBase* transforms, const void* geometry_ptr, const void* material_ptr ); 

void* OXRAP_API OCtx_create_group( const char* key, const void* child_group_ptr); 
void  OXRAP_API OCtx_group_add_child_group( void* group_ptr , void* child_group_ptr ); 



