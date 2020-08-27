#include "plog/Severity.h"
#include "NPYBase.hpp"
#include "OCtx.hh"
#include "PLOG.hh"

#include "optix.h"
#define OPTIX_VERSION_MAJOR (OPTIX_VERSION / 10000)
#define OPTIX_VERSION_MINOR ((OPTIX_VERSION % 10000) / 100)
#define OPTIX_VERSION_MICRO (OPTIX_VERSION % 100)

#if OPTIX_VERSION_MAJOR >= 7

#else

#include <optixu/optixpp_namespace.h>
#include "OFormat.hh"
#include "OTex.hh"

#endif

/**




**/


#if OPTIX_VERSION_MAJOR >= 7
struct OCtx 
{
    static OCtx* INSTANCE ; 
    static const plog::Severity LEVEL ; 

    // TODO: equivalent calls with OptiX 7 API 
    //  but probably there are no equivalents for many ..
    //  need to develop some expertise to see the commonality 
    //  and find possibly some higher level API that can be common 

    OCtx()
    {
        INSTANCE = this ; 
    }
};

#else
struct OCtx 
{
    static OCtx* INSTANCE ; 
    static const plog::Severity LEVEL ; 
    optix::Context context ;    
    OCtx()
    {
        INSTANCE = this ; 
        context = optix::Context::create();
        context->setRayTypeCount(1);
        context->setExceptionEnabled( RT_EXCEPTION_ALL , true );
        context->setPrintEnabled(1);
        context->setPrintBufferSize(4096);
        context->setEntryPointCount(1);
    }
    void* get()
    {
        RTcontext ctxPtr = context->get(); 
        return ctxPtr ;   
        // recovery: optix::Context context = optix::Context::take((RTcontext)get()) ;
    }
    bool has_variable( const char* key )
    {
        optix::Variable var = context->queryVariable(key); 
        return var.get() != NULL ; 
    }
    void* create_buffer(const NPYBase* arr, const char* key, const char type, const char flag)
    {
        unsigned buffer_type = 0 ; 
        switch(type)
        {
            case 'O': buffer_type = RT_BUFFER_OUTPUT        ; break ; 
            case 'I': buffer_type = RT_BUFFER_INPUT         ; break ; 
            case 'B': buffer_type = RT_BUFFER_INPUT_OUTPUT  ; break ; 
        }
        unsigned buffer_flag = 0 ;
        switch(flag)
        {
            case 'L': buffer_flag = RT_BUFFER_LAYERED       ; break ; 
            case 'G': buffer_flag = RT_BUFFER_GPU_LOCAL     ; break ; 
            case 'C': buffer_flag = RT_BUFFER_COPY_ON_DIRTY ; break ; 
        }

        unsigned buffer_desc = buffer_type | buffer_flag ; 
        optix::Buffer buf = context->createBuffer(buffer_desc) ;  

        unsigned multiplicity = arr->getShape(-1) ;  // last shape dimension -> multiplicity -> buffer format
        assert( multiplicity == 1 || multiplicity == 2 || multiplicity == 3 || multiplicity == 4 ); 
        RTformat format = OFormat::ArrayType(arr); 
        buf->setFormat( format ); 

        LOG(LEVEL) << " arr " << arr->getShapeString() ; 

        unsigned nd = arr->getDimensions(); 
        if( nd == 2 )
        {
            unsigned width = arr->getShape(0); 
            buf->setSize(width);   
            LOG(LEVEL) << " width " << width ; 
        }
        else if( nd == 3 )
        {
            unsigned height = arr->getShape(0); 
            unsigned width = arr->getShape(1) ; 
            buf->setSize(width, height);   
            LOG(LEVEL) 
                << " height " << height 
                << " width " << width 
                ; 
        }
        else if( nd == 4 )
        {
            unsigned depth = arr->getShape(0) ; // when layered the depth is the number of layers
            unsigned height = arr->getShape(1); 
            unsigned width = arr->getShape(2) ; 
            buf->setSize(width, height, depth);   
            LOG(LEVEL) 
                << " depth " << depth 
                << " height " << height 
                << " width " << width 
                ; 
        }
        if(key != NULL)
        {
            LOG(info) << " placing buffer into context with key " << key ; 
            context[key]->setBuffer(buf); 
        }

        optix::BufferObj* bufObj = buf.get();  
        RTbuffer bufPtr = bufObj->get(); 
        void* ptr = bufPtr ; 

        if(type == 'I' || type == 'B')
        {
            upload_buffer( arr, ptr ); 
        }
        return ptr ; 
    } 
    void* get_buffer( const char* key )
    { 
        assert( has_variable(key) ); 
        optix::Buffer buf = context[key]->getBuffer(); 
        optix::BufferObj* bufObj = buf.get();  
        RTbuffer bufPtr = bufObj->get(); 
        void* ptr = bufPtr ; 
        return ptr ;  
    }
    void desc_buffer( void* buffer_ptr )
    {
        RTbuffer bufPtr = (RTbuffer)buffer_ptr ;  // recovering the buffer from the void* ptr 
        optix::Buffer buf = optix::Buffer::take(bufPtr) ;
        unsigned nd = buf->getDimensionality(); 

        RTsize depth(0);  
        RTsize height(0);  
        RTsize width(0);  

        if( nd == 1 ){
            buf->getSize(width); 
            LOG(info) << " dimensionality " << nd << " width " << width ; 
        } else if( nd == 2 ){
            buf->getSize(width, height); 
            LOG(info) << " dimensionality " << nd << " width " << width << " height " << height ; 
        } else if( nd == 3 ){
            buf->getSize(width, height, depth);
            LOG(info) << " dimensionality " << nd << " width " << width << " height " << height << " depth " << depth ; 
        } else {
            assert(0); 
        }
    }
    void upload_buffer( const NPYBase* arr, void* buffer_ptr )
    {
        RTbuffer bufPtr = (RTbuffer)buffer_ptr ;  // recovering the buffer from the void* ptr 
        optix::Buffer buf = optix::Buffer::take(bufPtr) ;

        void* buf_data = buf->map() ; 
        arr->write_(buf_data); 
        buf->unmap(); 
    }
    void download_buffer( NPYBase* arr, const char* key )
    {
        bool exists = has_variable( key ); 
        if(!exists) LOG(fatal) << "no buffer in context with key " << key  ; 
        assert(exists); 

        optix::Buffer buf = context[key]->getBuffer(); 
        void* buf_data = buf->map() ; 
        arr->read(buf_data); 
        buf->unmap(); 
    }



    void set_raygen_program( unsigned entry_point_index, const char* ptx_path, const char* func )
    {
        optix::Program program = context->createProgramFromPTXFile( ptx_path, func ) ; 
        context->setRayGenerationProgram( entry_point_index,  program );
    }
    void set_exception_program( unsigned entry_point_index, const char* ptx_path, const char* func )
    {
        optix::Program program = context->createProgramFromPTXFile( ptx_path, func ) ; 
        context->setExceptionProgram( entry_point_index,  program );
    }



    void* create_geometry(unsigned prim_count, const char* ptxpath, const char* bounds_func, const char* intersect_func )
    {
        optix::Geometry geom = context->createGeometry();
        geom->setPrimitiveCount( prim_count );
        LOG(info) << "[ ptxpath " << ptxpath ; 
        optix::Program bd = context->createProgramFromPTXFile( ptxpath, bounds_func ) ;  
        optix::Program in = context->createProgramFromPTXFile( ptxpath, intersect_func ) ;  
        geom->setBoundingBoxProgram(bd);
        geom->setIntersectionProgram(in);
        geom["sphere"]->setFloat( 0, 0, 0, 1.5 );
        LOG(info) << "] ptxpath " << ptxpath ; 

        optix::GeometryObj* geomObj = geom.get(); 
        RTgeometry geomPtr = geomObj->get();  
        void* ptr = geomPtr ;  
        return ptr ;
    }
    void* create_material(const char* ptxpath, const char* closest_hit_func, unsigned entry_point_index )
    {
        optix::Material mat = context->createMaterial();
        LOG(info) << "[ compile ch " ;
        optix::Program ch = context->createProgramFromPTXFile( ptxpath, closest_hit_func ) ;     
        LOG(info) << "] compile ch " ;  
        mat->setClosestHitProgram( entry_point_index, ch );

        optix::MaterialObj* matObj = mat.get();
        RTmaterial matPtr = matObj->get(); 
        void* ptr = matPtr ; 
        return ptr ; 
    }
    void* create_geometry_instance(void* geo_ptr, void* mat_ptr)
    {
         optix::Geometry geo = optix::Geometry::take((RTgeometry)geo_ptr);   
         optix::Material mat = optix::Material::take((RTmaterial)mat_ptr);   
         optix::GeometryInstance gi = context->createGeometryInstance( geo, &mat, &mat+1 ) ;

         optix::GeometryInstanceObj* giObj = gi.get(); 
         RTgeometryinstance giPtr = giObj->get(); 
         void* ptr = giPtr ; 
         return ptr ; 
    }
    void* create_geometry_group(const std::vector<void*>& v_gi_ptr)
    {
        optix::GeometryGroup gg = context->createGeometryGroup();
        unsigned ngi = v_gi_ptr.size(); 
        gg->setChildCount(ngi);
        for(unsigned i=0 ; i < ngi ; i++)
        {
            void* giPtr = v_gi_ptr[0]; 
            optix::GeometryInstance gi = optix::GeometryInstance::take((RTgeometryinstance)giPtr); 
            gg->setChild( i, gi );
        }
        optix::GeometryGroupObj* ggObj = gg.get(); 
        RTgeometrygroup ggPtr = ggObj->get(); 
        void* ptr = ggPtr ; 
        return ptr ; 
    }
    void* create_acceleration( const char* accel )
    {
        optix::Acceleration acc = context->createAcceleration(accel); 
        optix::AccelerationObj* accObj = acc.get(); 
        RTacceleration accPtr = accObj->get(); 
        void* ptr = accPtr ; 
        return ptr ; 
    }
    void set_acceleration(void* gg_ptr, void* ac_ptr )
    {
        optix::GeometryGroup gg = optix::GeometryGroup::take((RTgeometrygroup)gg_ptr); 
        optix::Acceleration  ac = optix::Acceleration::take((RTacceleration)ac_ptr) ; 
        gg->setAcceleration(ac); 
    }
    void set_geometry_group_context_variable( const char* key, void* gg_ptr )
    {
        optix::GeometryGroup gg = optix::GeometryGroup::take((RTgeometrygroup)gg_ptr); 
        context[key]->set(gg) ;  
    }



    void compile()
    {
        context->compile(); 
    }
    void validate()
    {
        context->validate(); 
    }
    void launch(unsigned entry_point_index, unsigned width, unsigned height, unsigned depth)
    {
        context->launch(entry_point_index, width, height, depth); 
    }
    void launch(unsigned entry_point_index, unsigned width, unsigned height)
    {
        context->launch(entry_point_index, width, height); 
    }
    void launch(unsigned entry_point_index, unsigned width)
    {
        context->launch(entry_point_index, width); 
    }



    unsigned create_texture_sampler( void* buffer_ptr, const char* config )
    {
        RTbuffer bufPtr = (RTbuffer)buffer_ptr ;  
        optix::Buffer buffer = optix::Buffer::take(bufPtr) ;
        unsigned nd = buffer->getDimensionality() ;    
        LOG(info) << "[ creating tex_sampler with buffer of dimensionality nd " << nd ;  
        optix::TextureSampler tex = context->createTextureSampler(); 

        //RTwrapmode wrapmode = RT_WRAP_REPEAT ; 
        RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_EDGE ; 
        //RTwrapmode wrapmode = RT_WRAP_MIRROR ;
        //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_BORDER ; 
        tex->setWrapMode(0, wrapmode);
        tex->setWrapMode(1, wrapmode);
        //tex->setWrapMode(2, wrapmode);   corresponds to layer?

        RTfiltermode filtermode = RT_FILTER_NEAREST ;  // RT_FILTER_LINEAR 
        RTfiltermode minification = filtermode ; 
        RTfiltermode magnification = filtermode ; 
        RTfiltermode mipmapping = RT_FILTER_NONE ; 

        tex->setFilteringModes(minification, magnification, mipmapping);

        RTtextureindexmode indexmode = (RTtextureindexmode)OTex::IndexMode(config) ;  
        LOG(info) << "tex.setIndexingMode [" << OTex::IndexModeString(indexmode) << "]" ; 
        tex->setIndexingMode( indexmode );  

        //RTtexturereadmode readmode = RT_TEXTURE_READ_NORMALIZED_FLOAT ; // return floating point values normalized by the range of the underlying type
        RTtexturereadmode readmode = RT_TEXTURE_READ_ELEMENT_TYPE ;  // return data of the type of the underlying buffer
        // when the underlying type is float the is no difference between RT_TEXTURE_READ_NORMALIZED_FLOAT and RT_TEXTURE_READ_ELEMENT_TYPE

        tex->setReadMode( readmode ); 
        tex->setMaxAnisotropy(1.0f);
        LOG(LEVEL) << "] creating tex_sampler " ;  

        unsigned deprecated0 = 0 ; 
        unsigned deprecated1 = 0 ; 
        tex->setBuffer(deprecated0, deprecated1, buffer); 

        unsigned tex_id = tex->getId() ; 
        return tex_id ; 
    }
    void set_texture_param( void* buffer_ptr, unsigned tex_id, const char* param_key )
    {
        RTbuffer bufPtr = (RTbuffer)buffer_ptr ;  
        optix::Buffer buffer = optix::Buffer::take(bufPtr) ;
        unsigned nd = buffer->getDimensionality() ;    

        RTsize depth(0);  
        RTsize height(0);  
        RTsize width(0);  

        switch(nd){
            case 1: buffer->getSize(width)               ; break ; 
            case 2: buffer->getSize(width,height)        ; break ; 
            case 3: buffer->getSize(width,height,depth)  ; break ;
            default: assert(0)                           ; break ; 
        }
        optix::int4 param = optix::make_int4(width, height, depth, tex_id); 
        context[param_key]->setInt(param);
        LOG(info) << param_key << " ( " << param.x << " " << param.y << " " << param.z << " " << param.w << " " << " ) " << " ni/nj/nk/tex_id " ; 
    }


    void set_float4( const char* key, float x, float y, float z, float w )
    {
        context[key]->setFloat(optix::make_float4(x, y, z, w));
    }
    void set_int4( const char* key, int x, int y, int z, int w )
    {
        context[key]->setInt(optix::make_int4(x, y, z, w));
    }
    void set_viewpoint( const glm::vec3& eye, const glm::vec3& U,  const glm::vec3& V, const glm::vec3& W, const float scene_epsilon )
    {
        context[ "scene_epsilon"]->setFloat( scene_epsilon );
        context[ "eye"]->setFloat( eye.x, eye.y, eye.z  );  
        context[ "U"  ]->setFloat( U.x, U.y, U.z  );  
        context[ "V"  ]->setFloat( V.x, V.y, V.z  );  
        context[ "W"  ]->setFloat( W.x, W.y, W.z  );  
        context[ "radiance_ray_type"   ]->setUint( 0u );        
    }


};

#endif

const plog::Severity OCtx::LEVEL = PLOG::EnvLevel("OCtx", "DEBUG"); 
OCtx* OCtx::INSTANCE = NULL ; 
OCtx* OCtx_()
{
    if(OCtx::INSTANCE == NULL) OCtx::INSTANCE = new OCtx() ; 
    return OCtx::INSTANCE ;  
}
void* OCtx_get() { return OCtx_()->get() ;  }
bool  OCtx_has_variable( const char* key){ return OCtx_()->has_variable(key) ; }

void* OCtx_create_buffer(const NPYBase* arr, const char* key, char type, char flag) { return OCtx_()->create_buffer(arr, key, type, flag); }
void* OCtx_get_buffer(const char* key) { return OCtx_()->get_buffer(key);  }
void  OCtx_desc_buffer(void* ptr) { OCtx_()->desc_buffer(ptr); }
void  OCtx_upload_buffer(const NPYBase* arr, void* buffer_ptr) { OCtx_()->upload_buffer(arr, buffer_ptr); }
void  OCtx_download_buffer(NPYBase* arr, const char* key) { OCtx_()->download_buffer(arr, key); }

void  OCtx_set_raygen_program( unsigned entry_point_index, const char* ptx_path, const char* func ) { OCtx_()->set_raygen_program(entry_point_index, ptx_path, func ); }
void  OCtx_set_exception_program( unsigned entry_point_index, const char* ptx_path, const char* func ) { OCtx_()->set_exception_program(entry_point_index, ptx_path, func ); }

void* OCtx_create_geometry(unsigned prim_count, const char* ptxpath, const char* bounds_func, const char* intersect_func ) { return OCtx_()->create_geometry(prim_count, ptxpath, bounds_func, intersect_func);  }
void* OCtx_create_material(const char* ptxpath, const char* closest_hit_func, unsigned entry_point_index ) { return OCtx_()->create_material(ptxpath, closest_hit_func, entry_point_index); } 
void* OCtx_create_geometry_instance(void* geo_ptr, void* mat_ptr ) {   return OCtx_()->create_geometry_instance(geo_ptr, mat_ptr); }
void* OCtx_create_geometry_group(const std::vector<void*>& v_gi_ptr) { return OCtx_()->create_geometry_group(v_gi_ptr); } 
void* OCtx_create_acceleration( const char* accel ) {                  return OCtx_()->create_acceleration( accel ); } 
void  OCtx_set_acceleration(void* gg_ptr, void* ac_ptr ) {                        OCtx_()->set_acceleration(gg_ptr, ac_ptr); }
void  OCtx_set_geometry_group_context_variable( const char* key, void* gg_ptr ) { OCtx_()->set_geometry_group_context_variable(key, gg_ptr); }

void  OCtx_compile() { OCtx_()->compile(); }
void  OCtx_validate() { OCtx_()->validate(); }
void  OCtx_launch(unsigned entry_point_index, unsigned width, unsigned height, unsigned depth) { OCtx_()->launch(entry_point_index, width, height, depth); }
void  OCtx_launch(unsigned entry_point_index, unsigned width, unsigned height) {                 OCtx_()->launch(entry_point_index, width, height); }
void  OCtx_launch(unsigned entry_point_index, unsigned width) {                                  OCtx_()->launch(entry_point_index, width); }

unsigned OCtx_create_texture_sampler( void* buffer_ptr, const char* config ) { return OCtx_()->create_texture_sampler(buffer_ptr, config); }
void     OCtx_set_texture_param( void* buffer_ptr, unsigned tex_id, const char* param_key ) { OCtx_()->set_texture_param(buffer_ptr, tex_id, param_key); }

void  OCtx_set_float4(const char* key, float x, float y, float z, float w ) { OCtx_()->set_float4(key, x, y, z, w); }
void  OCtx_set_int4(const char* key, int x, int y, int z, int w ) { OCtx_()->set_int4(key, x, y, z, w); }
void  OCtx_set_viewpoint( const glm::vec3& eye, const glm::vec3& U,  const glm::vec3& V, const glm::vec3& W, const float scene_epsilon ) { OCtx_()->set_viewpoint( eye, U, V, W, scene_epsilon ); }


