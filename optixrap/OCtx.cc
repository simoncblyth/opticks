#include <chrono>
#include "plog/Severity.h"

#include "SPack.hh"
#include "NPY.hpp"
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
#include "OBuffer.hh"
#include "OTex.hh"

#endif

const plog::Severity OCtx::LEVEL = PLOG::EnvLevel("OCtx", "DEBUG"); 
OCtx* OCtx::INSTANCE = NULL ; 

OCtx::OCtx(void* ptr)
    :
    m_context_ptr(ptr ? ptr :  init())
{
}

void* OCtx::init()
{
    optix::Context context = optix::Context::create();
    context->setRayTypeCount(1);
    context->setExceptionEnabled( RT_EXCEPTION_ALL , true );
    context->setPrintEnabled(1);
    context->setPrintBufferSize(4096);
    context->setEntryPointCount(1);

    optix::ContextObj* contextObj = context.get(); 
    RTcontext contextPtr = contextObj->get(); 
    void* ptr = contextPtr ; 
    return ptr ; 
}

OCtx* OCtx::Get()  // static 
{
    if(INSTANCE == NULL) INSTANCE = new OCtx() ; 
    return INSTANCE  ;  
}

void* OCtx::ptr()
{
    return m_context_ptr ; 
}

bool OCtx::has_variable( const char* key )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Variable var = context->queryVariable(key); 
    return var.get() != NULL ; 
}

/**
OCtx::create_buffer
--------------------

Creates GPU buffer with dimensions and elements to match that of the 
argument NPY array (or sub-array  when item argument is > -1).
The last dimension of the array determines the format multiplicity 
of the buffer which must be 1,2,3 or 4.

For 'I' or 'B' type input buffers the array is also 
uploaded to the GPU buffer.

arr
    NPYBase array, for output buffers (type 'O') the array need not be allocated yet. 
    For input (type 'I') or input_output (type 'B') buffers the array must 
    be allocated and populated already. 

    Array dimensions of zero are accepted and treated just like non-zero values, 
    being transposed as controlled by the transpose argument.  
    This allows empty arrays to yield empty buffers.

key 
    name of the buffer for GPU declaration and usage 
type
    O:OUTPUT, I:INPUT, B:INPUT_OUTPUT
flag
    L:LAYERED, G:GPU_LOCAL, C:COPY_ON_DIRTY 
    These are rarely used, typically leave as ' ' 
item      
    for item -1 the entire array becomes the buffer, for item values 0,1,2 etc..
    only a single item from the array goes into the buffer.

transpose
    when *false* the buffer dimensions match the array, this 
    will typically result in array content being scrambled with 
    a "fold-over" structure due to serialization order mismatch.
    
    when *true* the buffer dimensions are transposed compared to the array, 

    For 1d arrays *transpose* does nothing. 

    NPY arrays use row-major serialization order whereas OptiX buffers
    use column-major serialization order. To avoid array content being 
    order scrambled it is necessary to use transpose_dimensions = true.
    For details on this see tests/OCtx{2,3}dTest.cc  

    For 1d/2d/3d array or subarray (excluding the final element dimension), 
    my convention is to name the array dimensions as indicated::

         1d:                 (width) 
         2d:         (height, width)
         3d:  (depth, height, width)

    This naming is to match the natural image serialization ordering 
    starting with the top line raster etc.. 

    The sizes of these dimensions are then used to control the buffer dimensions.
    When transpose_dimensions is false the GPU buffer dimensions are not transposed, staying asis.

    When transpose_dimensions is true those yield a GPU buffer of dimensions::

         1d:                (width)
         2d:         (width,height)
         3d:   (width,height,depth) 


    Note that transposing the buffer shape it is also necessary to transpose the shape 
    of the launch when using::
    
        output_buffer[launch_index] = val 



**/

void* OCtx::create_buffer(const NPYBase* arr, const char* key, const char type, const char flag, int item, bool transpose  ) const 
{
    if(transpose == false)
       LOG(warning) << "CAUTION : are using transpose:false, this typically causes array content order scrambling "
       ; 

    LOG(LEVEL) << "[" ; 
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
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
    LOG(LEVEL) << " buffer_desc " << OBuffer::BufferDesc(buffer_desc) ; 

    optix::Buffer buf = context->createBuffer(buffer_desc) ;  

    unsigned multiplicity = arr->getShape(-1) ;  // last shape dimension -> multiplicity -> buffer format

    bool allowed_multiplicity =  multiplicity == 1 || multiplicity == 2 || multiplicity == 3 || multiplicity == 4 ; 

    if(!allowed_multiplicity)
        LOG(fatal) 
            << " FATAL multiplicity is not allowed " << multiplicity 
            << " shape " << arr->getShapeString()
            << " key " << key 
            ; 

    assert( allowed_multiplicity ); 
    RTformat format = OFormat::ArrayType(arr); 
    buf->setFormat( format ); 

    unsigned array_nd = arr->getNumDimensions(); 
    bool subarray = item > -1 ; 
    unsigned buffer_nd = subarray ? array_nd - 2 : array_nd - 1 ; 


    if( buffer_nd == 1 )
    {
        unsigned width = arr->getShape(subarray ? 1 : 0); 
        LOG(LEVEL) 
            << " arr " << arr->getShapeString() 
            << " item " << item 
            << " subarray " << ( subarray ? "Y" : "N" )
            << " array_nd " << array_nd
            << " buffer_nd " << buffer_nd
            << " width " << width 
            << " transpose " << transpose
            ; 

        buf->setSize(width);   
    }
    else if( buffer_nd == 2 )
    {
        unsigned height = arr->getShape(subarray ? 1 : 0 ) ; 
        unsigned width  = arr->getShape(subarray ? 2 : 1 ) ; 
        LOG(LEVEL) 
            << " arr " << arr->getShapeString() 
            << " item " << item 
            << " subarray " << ( subarray ? "Y" : "N" )
            << " array_nd " << array_nd
            << " buffer_nd " << buffer_nd
            << " height " << height 
            << " width " << width 
            << " transpose " << transpose
            ; 

        if( transpose )
        {
            buf->setSize(width, height);   
        }
        else
        {
            buf->setSize(height, width);   
        }
    }
    else if( buffer_nd == 3 )
    {
        unsigned depth  = arr->getShape(subarray ? 1 : 0 ) ;
        unsigned height = arr->getShape(subarray ? 2 : 1 ) ; 
        unsigned width  = arr->getShape(subarray ? 3 : 2 ) ;
 
        LOG(LEVEL) 
            << " arr " << arr->getShapeString() 
            << " item " << item 
            << " subarray " << ( subarray ? "Y" : "N" )
            << " array_nd " << array_nd
            << " buffer_nd " << buffer_nd
            << " depth " << depth 
            << " height " << height 
            << " width " << width 
            << " transpose " << transpose
            ; 
         
        if( transpose )
        {
            buf->setSize(width, height, depth);   
        }
        else
        {
            buf->setSize(depth, height, width);   
        }

    }
    else
    {
         LOG(fatal) 
            << " not enough dimensions "
            << " arr " << arr->getShapeString() 
            << " item " << item 
            << " subarray " << ( subarray ? "Y" : "N" )
            << " array_nd " << array_nd
            << " buffer_nd " << buffer_nd
            << " transpose " << transpose
             ; 
         assert(0); 
    } 

    if(key != NULL)
    {
        LOG(LEVEL) << " placing buffer into context with key " << key ; 
        context[key]->setBuffer(buf); 
    }

    optix::BufferObj* bufObj = buf.get();  
    RTbuffer bufPtr = bufObj->get(); 
    void* ptr = bufPtr ; 

    if(type == 'I' || type == 'B') // input buffer or input-output buffer
    {
        upload_buffer( arr, ptr, item ); 
    }
    LOG(LEVEL) << "]" ; 
    return ptr ; 
} 

void* OCtx::get_buffer( const char* key )
{ 
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    assert( has_variable(key) ); 
    optix::Buffer buf = context[key]->getBuffer(); 
    optix::BufferObj* bufObj = buf.get();  
    RTbuffer bufPtr = bufObj->get(); 
    void* ptr = bufPtr ; 
    return ptr ;  
}

void OCtx::desc_buffer( void* buffer_ptr )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
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

void OCtx::upload_buffer( const NPYBase* arr, void* buffer_ptr, int item ) const 
{
    LOG(LEVEL) << "[ " << item  ; 
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    RTbuffer bufPtr = (RTbuffer)buffer_ptr ;  // recovering the buffer from the void* ptr 
    optix::Buffer buf = optix::Buffer::take(bufPtr) ;

    void* buf_data = buf->map() ; 

    if( item < 0 )
    {
        arr->write_(buf_data); 
    }
    else
    {
        arr->write_item_(buf_data, item); 
    }

    buf->unmap(); 
    LOG(LEVEL) << "]" ; 
}


void OCtx::download_buffer( NPYBase* arr, const char* key, int item)
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    bool exists = has_variable( key ); 
    if(!exists) LOG(fatal) << "no buffer in context with key " << key  ; 
    assert(exists); 

    optix::Buffer buf = context[key]->getBuffer(); 
    void* buf_data = buf->map() ; 

    if( item < 0 )
    {
        arr->read_(buf_data); 
    }
    else
    {
        arr->read_item_(buf_data, item); 
    }

    buf->unmap(); 
}



void OCtx::set_raygen_program( unsigned entry_point_index, const char* ptx_path, const char* func )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Program program = context->createProgramFromPTXFile( ptx_path, func ) ; 
    context->setRayGenerationProgram( entry_point_index,  program );
}


void OCtx::set_exception_program( unsigned entry_point_index, const char* ptx_path, const char* func )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Program program = context->createProgramFromPTXFile( ptx_path, func ) ; 
    context->setExceptionProgram( entry_point_index,  program );
}

void OCtx::set_miss_program( unsigned entry_point_index, const char* ptx_path, const char* func )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Program program = context->createProgramFromPTXFile( ptx_path, func ) ; 
    context->setMissProgram( entry_point_index,  program );
}


void* OCtx::create_geometry(unsigned prim_count, const char* ptxpath, const char* bounds_func, const char* intersect_func )
{
    LOG(LEVEL) << "[" ; 
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Geometry geom = context->createGeometry();
    geom->setPrimitiveCount( prim_count );
    LOG(LEVEL) << "[ ptxpath " << ptxpath ; 
    optix::Program bd = context->createProgramFromPTXFile( ptxpath, bounds_func ) ;  
    optix::Program in = context->createProgramFromPTXFile( ptxpath, intersect_func ) ;  
    geom->setBoundingBoxProgram(bd);
    geom->setIntersectionProgram(in);


    LOG(LEVEL) << "] ptxpath " << ptxpath ; 

    optix::GeometryObj* geomObj = geom.get(); 
    RTgeometry geomPtr = geomObj->get();  
    void* ptr = geomPtr ;  
    LOG(LEVEL) << "]" ; 
    return ptr ;
}

void* OCtx::create_material(const char* ptxpath, const char* closest_hit_func, unsigned entry_point_index )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Material mat = context->createMaterial();
    LOG(LEVEL) << "[ compile ch " ;
    optix::Program ch = context->createProgramFromPTXFile( ptxpath, closest_hit_func ) ;     
    LOG(LEVEL) << "] compile ch " ;  
    mat->setClosestHitProgram( entry_point_index, ch );

    optix::MaterialObj* matObj = mat.get();
    RTmaterial matPtr = matObj->get(); 
    void* ptr = matPtr ; 
    return ptr ; 
}

void* OCtx::create_geometryinstance(void* geo_ptr, void* mat_ptr)
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Geometry geo = optix::Geometry::take((RTgeometry)geo_ptr);   
    optix::Material mat = optix::Material::take((RTmaterial)mat_ptr);   
    optix::GeometryInstance gi = context->createGeometryInstance( geo, &mat, &mat+1 ) ;

    optix::GeometryInstanceObj* giObj = gi.get(); 
    RTgeometryinstance giPtr = giObj->get(); 
    void* ptr = giPtr ; 
    return ptr ; 
}

void* OCtx::create_geometrygroup(const void* gi_ptr)
{
    std::vector<const void*> v_gi_ptr ; 
    v_gi_ptr.push_back(gi_ptr); 
    return create_geometrygroup(v_gi_ptr ); 
}

void* OCtx::create_geometrygroup(const std::vector<const void*>& v_gi_ptr)
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::GeometryGroup gg = context->createGeometryGroup();
    unsigned ngi = v_gi_ptr.size(); 
    gg->setChildCount(ngi);
    for(unsigned i=0 ; i < ngi ; i++)
    {
        const void* giPtr = v_gi_ptr[0]; 
        optix::GeometryInstance gi = optix::GeometryInstance::take((RTgeometryinstance)giPtr); 
        gg->setChild( i, gi );
    }
    optix::GeometryGroupObj* ggObj = gg.get(); 
    RTgeometrygroup ggPtr = ggObj->get(); 
    void* ptr = ggPtr ; 
    return ptr ; 
}


void* OCtx::create_acceleration( const char* accel )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Acceleration acc = context->createAcceleration(accel); 
    optix::AccelerationObj* accObj = acc.get(); 
    RTacceleration accPtr = accObj->get(); 
    void* ptr = accPtr ; 
    return ptr ; 
}

void OCtx::set_geometrygroup_acceleration(void* geometrygroup_ptr, void* ac_ptr )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::GeometryGroup gg = optix::GeometryGroup::take((RTgeometrygroup)geometrygroup_ptr); 
    optix::Acceleration  accel = optix::Acceleration::take((RTacceleration)ac_ptr) ; 
    gg->setAcceleration(accel); 
}

void OCtx::set_group_acceleration(void* group_ptr, void* ac_ptr )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Group         group = optix::Group::take((RTgroup)group_ptr); 
    optix::Acceleration  accel = optix::Acceleration::take((RTacceleration)ac_ptr) ; 
    group->setAcceleration(accel); 
}

void OCtx::set_geometrygroup_context_variable( const char* key, void* gg_ptr )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::GeometryGroup gg = optix::GeometryGroup::take((RTgeometrygroup)gg_ptr); 
    context[key]->set(gg) ;  
}



void OCtx::compile()
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    context->compile(); 
}

void OCtx::validate()
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    context->validate(); 
}

void OCtx::launch(unsigned entry_point_index, unsigned width, unsigned height, unsigned depth)
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    context->launch(entry_point_index, width, height, depth); 
}

void OCtx::launch(unsigned entry_point_index, unsigned width, unsigned height)
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    context->launch(entry_point_index, width, height); 
}

void OCtx::launch(unsigned entry_point_index, unsigned width)
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    context->launch(entry_point_index, width); 
}

void OCtx::launch_instrumented( unsigned entry_point_index, unsigned width, unsigned height, double& t_prelaunch, double& t_launch  )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    LOG(LEVEL) << "[" ; 

    auto t0 = std::chrono::high_resolution_clock::now();

    context->launch( entry_point_index , 0, 0  );

    auto t1 = std::chrono::high_resolution_clock::now();

    context->launch( entry_point_index , width, height  );

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> t_prelaunch_ = t1 - t0;
    std::chrono::duration<double> t_launch_    = t2 - t1;

    t_prelaunch = t_prelaunch_.count() ;
    t_launch = t_launch_.count() ;

    std::cout 
         << " prelaunch " << std::setprecision(4) << std::fixed << std::setw(15) << t_prelaunch 
         << " launch    " << std::setprecision(4) << std::fixed << std::setw(15) << t_launch 
         << std::endl 
         ;   
    LOG(LEVEL) << "]" ; 
}

unsigned OCtx::create_texture_sampler( void* buffer_ptr, const char* config ) const 
{
    LOG(LEVEL) << "["; 
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
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

    bool layered = depth > 0 ; 
    //unsigned array_size = layered ? depth : 1u ; 
    unsigned array_size = 1u ;   // setting to anything other than 1 segments ?

    LOG(LEVEL) 
        << " nd " << nd 
        <<  " (depth, height, width )  (" << depth << " " << height << " " << width << ")" 
        << " layered " << ( layered ? "Y" : "N" )
        << " array_size " << array_size 
        ; 

    optix::TextureSampler tex = context->createTextureSampler(); 

    //RTwrapmode wrapmode = RT_WRAP_REPEAT ; 
    RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_EDGE ; 
    //RTwrapmode wrapmode = RT_WRAP_MIRROR ;
    //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_BORDER ; 
    tex->setWrapMode(0, wrapmode);
    tex->setWrapMode(1, wrapmode);
    //tex->setWrapMode(2, wrapmode);  // is this needed with layerd ?

    RTfiltermode filtermode = RT_FILTER_NEAREST ;  // RT_FILTER_LINEAR 
    RTfiltermode minification = filtermode ; 
    RTfiltermode magnification = filtermode ; 
    RTfiltermode mipmapping = RT_FILTER_NONE ; 

    tex->setFilteringModes(minification, magnification, mipmapping);

    RTtextureindexmode indexmode = (RTtextureindexmode)OTex::IndexMode(config) ;  
    LOG(LEVEL) << "tex.setIndexingMode [" << OTex::IndexModeString(indexmode) << "]" ; 
    tex->setIndexingMode( indexmode );  

    //RTtexturereadmode readmode = RT_TEXTURE_READ_NORMALIZED_FLOAT ; // return floating point values normalized by the range of the underlying type
    RTtexturereadmode readmode = RT_TEXTURE_READ_ELEMENT_TYPE ;  // return data of the type of the underlying buffer
    // when the underlying type is float the is no difference between RT_TEXTURE_READ_NORMALIZED_FLOAT and RT_TEXTURE_READ_ELEMENT_TYPE

    /*
    tex->setMipLevelCount(1u);
    LOG(LEVEL) << "[ setArraySize " << array_size ; 
    tex->setArraySize(array_size);     // deprecated call : setting to anything other than 1 segments 
    LOG(LEVEL) << "] setArraySize " << array_size ; 
    */

    tex->setReadMode( readmode ); 
    tex->setMaxAnisotropy(1.0f);

    unsigned deprecated0 = 0 ; 
    unsigned deprecated1 = 0 ; 
    tex->setBuffer(deprecated0, deprecated1, buffer); 

    unsigned tex_id = tex->getId() ; 
    LOG(LEVEL) << "]"; 
    return tex_id ; 
}

void OCtx::set_texture_param( void* buffer_ptr, unsigned tex_id, const char* param_key )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
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
    LOG(LEVEL) << param_key << " ( " << param.x << " " << param.y << " " << param.z << " " << param.w << " " << " ) " << " ni/nj/nk/tex_id " ; 

    if( param_key != NULL )
    {
        context[param_key]->setInt(param);
    }
}

/**
upload_2d_texture
------------------

Note reversed shape order of the texBuffer->setSize( width, height, depth)
wrt to the shape of the input buffer.  

For example with a landscape input PPM image of height 512 and width 1024 
the natural array shape to use is (height, width, ncomp) ie (512,1024,3) 
This is natural because it matches the row-major ordering of the image data 
in PPM files starting with the top row (with a width) and rastering down 
*height* by rows. 

BUT when specifying the dimensions of the tex buffer it is necessary to use::

     texBuffer->setSize(width, height, depth) 

**/

unsigned OCtx::upload_2d_texture(const char* param_key, const NPYBase* inp, const char* config, int item)
{
    LOG(LEVEL) << "[" ; 
    bool transform = true ; 
    void* buffer_ptr = create_buffer(inp, NULL, 'I', ' ', item, transform ); 
    unsigned tex_id = create_texture_sampler(buffer_ptr, config ); 
    set_texture_param( buffer_ptr, tex_id, param_key );  
    LOG(LEVEL) << "]" ; 
    return tex_id ; 
}


void OCtx::set_geometry_float4( void* geometry_ptr, const char* key, float x, float y, float z, float w )
{
    optix::Geometry geometry = optix::Geometry::take((RTgeometry)geometry_ptr);   
    geometry[key]->setFloat(optix::make_float4(x, y, z, w));  
}

void OCtx::set_geometry_float3( void* geometry_ptr, const char* key, float x, float y, float z)
{
    optix::Geometry geometry = optix::Geometry::take((RTgeometry)geometry_ptr);   
    geometry[key]->setFloat(optix::make_float3(x, y, z));  
}

void OCtx::set_context_float4( const char* key, float x, float y, float z, float w )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    context[key]->setFloat(optix::make_float4(x, y, z, w));
}

void OCtx::set_context_int4( const char* key, int x, int y, int z, int w )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    context[key]->setInt(optix::make_int4(x, y, z, w));
}

void OCtx::set_context_int( const char* key, int x )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    context[key]->setInt(x);
}





void OCtx::set_context_viewpoint( const glm::vec3& eye, const glm::vec3& U,  const glm::vec3& V, const glm::vec3& W, const float scene_epsilon )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    context[ "scene_epsilon"]->setFloat( scene_epsilon );
    context[ "eye"]->setFloat( eye.x, eye.y, eye.z  );  
    context[ "U"  ]->setFloat( U.x, U.y, U.z  );  
    context[ "V"  ]->setFloat( V.x, V.y, V.z  );  
    context[ "W"  ]->setFloat( W.x, W.y, W.z  );  
    context[ "radiance_ray_type"   ]->setUint( 0u );        
}

void* OCtx::create_transform( bool transpose, const float* m44, const float* inverse_m44 )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Transform xform = context->createTransform();
    xform->setMatrix(transpose, m44, inverse_m44);

    optix::TransformObj* xformObj = xform.get(); 
    RTtransform xformPtr = xformObj->get(); 
    void* ptr = xformPtr ; 
    return ptr ; 
}   

void* OCtx::create_single_assembly( const glm::mat4& m4, const void* geometry_ptr, const void* material_ptr )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Geometry geo = optix::Geometry::take((RTgeometry)geometry_ptr);   
    optix::Material mat = optix::Material::take((RTmaterial)material_ptr);   
    optix::Group assembly = context->createGroup();
    assembly->setChildCount(1);
    assembly->setAcceleration( context->createAcceleration( "Trbvh" ) );  

    optix::Acceleration instance_accel = context->createAcceleration( "Trbvh" );

    glm::mat4 m4c(m4);  
    unsigned m4c_03_identity = SPack::uint_from_float(m4c[0].w) ; 
 
    // Setting the right hand column incase of sneaky insertions.
    // My (OpenGL) convention is for translation in m4[3].xyz.
    // These slots are probably ignored by OptiX anyhow.
    m4c[0].w = 0.f ; 
    m4c[1].w = 0.f ; 
    m4c[2].w = 0.f ; 
    m4c[3].w = 1.f ; 

    bool transpose = true ; 
    optix::Transform xform = context->createTransform();
    xform->setMatrix(transpose, glm::value_ptr(m4c), 0); 

    unsigned ichild = 0 ; 
    assembly->setChild(ichild, xform);

    optix::GeometryInstance pergi = context->createGeometryInstance() ;
    pergi->setMaterialCount(1);
    pergi->setMaterial(0, mat );
    pergi->setGeometry(geo);
    pergi["identity"]->setUint(m4c_03_identity);

    optix::GeometryGroup perxform = context->createGeometryGroup();
    perxform->addChild(pergi); 
    perxform->setAcceleration(instance_accel) ; 

    xform->setChild(perxform);
    optix::GroupObj* groupObj = assembly.get(); 
    RTgroup groupPtr = groupObj->get();
    void* ptr = groupPtr ; 
    return ptr ; 
}

/**
OCtx::create_instanced_assembly
--------------------------------

::

   assembly                  (Group) 
      assembly_accel         (Acceleration) 

      xform_0                (Transform)
         perxform_0          (GeometryGroup)
             pergi_0         (GeometryInstance)  
                geo          (Geometry)  
                mat          (Material)
             instance_accel  (Acceleration)   

      xform_1                (Transform)
         perxform_1          (GeometryGroup)
             pergi_1         (GeometryInstance)  
                geo          (Geometry)
                mat          (Material)
             instance_accel  (Acceleration)   
**/

void* OCtx::create_instanced_assembly( const NPY<float>* transforms, const void* geometry_ptr, const void* material_ptr )
{ 
    unsigned num_tr = transforms->getNumItems() ; 
    LOG(LEVEL) 
         << " transforms " << transforms->getShapeString()
         ; 

    const char* accel = "Trbvh" ; 

    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Geometry geo = optix::Geometry::take((RTgeometry)geometry_ptr);   
    optix::Material mat = optix::Material::take((RTmaterial)material_ptr);   
   
    optix::Acceleration instance_accel = context->createAcceleration(accel);
    optix::Acceleration assembly_accel  = context->createAcceleration(accel);

    optix::Group assembly = context->createGroup();
    assembly->setChildCount( num_tr );
    assembly->setAcceleration( assembly_accel );  

    for(unsigned ichild=0 ; ichild < num_tr ; ichild++)
    {
        unsigned m4_03_identity = transforms->getUInt(ichild,0,3,0);      
        glm::mat4 m4 = transforms->getMat4(ichild,-1); 
        unsigned m4_03_check = SPack::uint_from_float(m4[0].w) ; 
        assert( m4_03_identity == m4_03_check );

        bool transpose = true ; 
        // Setting the right hand column incase of sneaky insertions.
        // My (OpenGL) convention is for translation in m4[3].xyz.
        // These slots are probably ignored by OptiX anyhow.
        m4[0].w = 0.f ; 
        m4[1].w = 0.f ; 
        m4[2].w = 0.f ; 
        m4[3].w = 1.f ; 

        optix::Transform xform = context->createTransform();
        xform->setMatrix(transpose, glm::value_ptr(m4), 0); 

        assembly->setChild(ichild, xform);

        optix::GeometryInstance pergi = context->createGeometryInstance() ;
        pergi->setMaterialCount(1);
        pergi->setMaterial(0, mat );
        pergi->setGeometry(geo);

        pergi["identity"]->setUint(m4_03_identity) ;  

        optix::GeometryGroup perxform = context->createGeometryGroup();
        perxform->addChild(pergi); 
        perxform->setAcceleration(instance_accel) ; 

        xform->setChild(perxform);
    }   

    optix::GroupObj* groupObj = assembly.get(); 
    RTgroup groupPtr = groupObj->get();
    void* ptr = groupPtr ; 
    return ptr ; 
}


void* OCtx::create_group( const char* key, const void* child_group_ptr )
{
    optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Group group = context->createGroup() ;
    if(key)
    {
        context[key]->set(group);
    }
    if(child_group_ptr)
    {
         optix::Group child = optix::Group::take((RTgroup)child_group_ptr); 
         group->addChild(child);
    }

    optix::GroupObj* groupObj = group.get(); 
    RTgroup groupPtr = groupObj->get();
    void* ptr = groupPtr ; 
    return ptr ; 
}

void OCtx::group_add_child_group( void* group_ptr , void* child_group_ptr )
{
    //optix::Context context = optix::Context::take((RTcontext)m_context_ptr); 
    optix::Group group = optix::Group::take((RTgroup)group_ptr); 
    optix::Group child = optix::Group::take((RTgroup)child_group_ptr); 
    group->addChild(child);
}


