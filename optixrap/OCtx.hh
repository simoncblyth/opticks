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

void* OXRAP_API OCtx_create_buffer(const NPYBase* arr, const char* key, const char type, const char flag); 
void* OXRAP_API OCtx_get_buffer(const char* key); 
void  OXRAP_API OCtx_desc_buffer(void* buffer_ptr); 
void  OXRAP_API OCtx_upload_buffer(const NPYBase* arr, void* buffer_ptr); 
void  OXRAP_API OCtx_download_buffer(    NPYBase* arr, const char* key); 

void  OXRAP_API OCtx_set_raygen_program(    unsigned entry_point_index, const char* ptx_path, const char* func );
void  OXRAP_API OCtx_set_exception_program( unsigned entry_point_index, const char* ptx_path, const char* func );

void* OXRAP_API OCtx_create_geometry(unsigned prim_count, const char* ptxpath, const char* bounds_func, const char* intersect_func ); 
void* OXRAP_API OCtx_create_material(const char* ptxpath, const char* closest_hit_func, unsigned entry_point_index ); 
void* OXRAP_API OCtx_create_geometry_instance(void* geo_ptr, void* mat_ptr ); 
void* OXRAP_API OCtx_create_geometry_group(const std::vector<void*>& v_gi_ptr); 
void* OXRAP_API OCtx_create_acceleration( const char* accel );
void  OXRAP_API OCtx_set_acceleration(void* gg_ptr, void* ac_ptr );
void  OXRAP_API OCtx_set_geometry_group_context_variable( const char* key, void* gg_ptr ); 

void  OXRAP_API OCtx_compile(); 
void  OXRAP_API OCtx_validate(); 
void  OXRAP_API OCtx_launch(unsigned entry_point_index, unsigned width, unsigned height, unsigned depth); 
void  OXRAP_API OCtx_launch(unsigned entry_point_index, unsigned width, unsigned height); 
void  OXRAP_API OCtx_launch(unsigned entry_point_index, unsigned width); 

unsigned OXRAP_API OCtx_create_texture_sampler( void* buffer_ptr, const char* config );
void     OXRAP_API OCtx_set_texture_param(      void* buffer_ptr, unsigned tex_id, const char* param_key ); 

void  OXRAP_API OCtx_set_float4(const char* key, float x, float y, float z, float w );
void  OXRAP_API OCtx_set_int4(const char* key, int x, int y, int z, int w );
void  OXRAP_API OCtx_set_viewpoint( const glm::vec3& eye, const glm::vec3& U,  const glm::vec3& V, const glm::vec3& W, const float scene_epsilon ); 

