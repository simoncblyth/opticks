#include <iomanip>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime_api.h>
// brap-
#include "BTimer.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "GLMFormat.hpp"

// okc-
#include "OpticksEntry.hh"
#include "OpticksBufferControl.hh"

// optixrap-
#include "STimes.hh"
#include "SPPM.hh"
#include "OConfig.hh"
#include "OContext.hh"

#include "PLOG.hh"
using namespace optix ; 

const char* OContext::COMPUTE_ = "COMPUTE" ; 
const char* OContext::INTEROP_ = "INTEROP" ; 


const char* OContext::getModeName()
{
    switch(m_mode)
    {
       case COMPUTE:return COMPUTE_ ; break ; 
       case INTEROP:return INTEROP_ ; break ; 
    }
    assert(0);
}



OpticksEntry* OContext::addEntry(char code)
{
    LOG(fatal) << "OContext::addEntry " << code ; 
    //assert(0);
    bool defer = true ; 
    unsigned index ;
    switch(code)
    { 
        case 'G': index = addEntry("generate.cu.ptx", "generate", "exception", defer) ; break ;
        case 'T': index = addEntry("generate.cu.ptx", "trivial",  "exception", defer) ; break ;
        case 'N': index = addEntry("generate.cu.ptx", "nothing",  "exception", defer) ; break ;
        case 'R': index = addEntry("generate.cu.ptx", "tracetest",  "exception", defer) ; break ;
        case 'D': index = addEntry("generate.cu.ptx", "dumpseed", "exception", defer) ; break ;
        case 'S': index = addEntry("seedTest.cu.ptx", "seedTest", "exception", defer) ; break ;
        case 'P': index = addEntry("pinhole_camera.cu.ptx", "pinhole_camera" , "exception", defer);  break;
    }
    return new OpticksEntry(index, code) ; 
}



OContext::OContext(optix::Context context, Mode_t mode, bool with_top, bool verbose) 
    : 
    m_context(context),
    m_mode(mode),
    m_debug_photon(-1),
    m_entry(0),
    m_closed(false),
    m_with_top(with_top),
    m_verbose(verbose)
{
    init();
}

optix::Context OContext::getContext()
{
     return m_context ; 
}

optix::Context& OContext::getContextRef()
{
     return m_context ; 
}


optix::Group OContext::getTop()
{
     return m_top ; 
}
unsigned int OContext::getNumRayType()
{
    return e_rayTypeCount ;
}


void OContext::setDebugPhoton(unsigned int debug_photon)
{
    m_debug_photon = debug_photon ; 
}
unsigned int OContext::getDebugPhoton()
{
    return m_debug_photon ; 
}


OContext::Mode_t OContext::getMode()
{
    return m_mode ; 
}


bool OContext::isCompute()
{
    return m_mode == COMPUTE ; 
}
bool OContext::isInterop()
{
    return m_mode == INTEROP ; 
}


void OContext::setStackSize(unsigned int stacksize)
{
    LOG(debug) << "OContext::setStackSize " << stacksize ;  
    m_context->setStackSize(stacksize);
}

void OContext::setPrintIndex(const std::string& pindex)
{
    LOG(debug) << "OContext::setPrintIndex " << pindex ;  
    if(!pindex.empty())
    {
        glm::ivec3 idx = givec3(pindex);
        LOG(debug) << "OContext::setPrintIndex " 
                  << pindex
                  << " idx " << gformat(idx) 
                   ;  
        m_context->setPrintLaunchIndex(idx.x, idx.y, idx.z);

/*
  rtContextSetPrintLaunchIndex 
  toggles printing for individual computation grid cells. 
  Print statements have no adverse effect on performance while printing is globally disabled, which is the default behavior.
*/

    }
}


void OContext::init()
{
    m_cfg = new OConfig(m_context);

    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(2*2*2*8192);
    //m_context->setPrintLaunchIndex(0,0,0);

    unsigned int num_ray_type = getNumRayType() ;

    m_context->setRayTypeCount( num_ray_type );   // more static than entry type count


    if(m_with_top)
    {
        m_top = m_context->createGroup();
        m_context[ "top_object" ]->set( m_top );
    }

    LOG(debug) << "OContext::init " 
              << " mode " << getModeName()
              << " num_ray_type " << num_ray_type 
              ; 
}


void OContext::cleanUp()
{
    m_context->destroy();
    m_context = 0;
}




optix::Program OContext::createProgram(const char* filename, const char* progname )
{
    return m_cfg->createProgram(filename, progname);
}

unsigned int OContext::addEntry(const char* filename, const char* raygen, const char* exception, bool defer)
{
    return m_cfg->addEntry(filename, raygen, exception, defer ); 
}
unsigned int OContext::addRayGenerationProgram( const char* filename, const char* progname, bool defer)
{
    return m_cfg->addRayGenerationProgram(filename, progname, defer);
}
unsigned int OContext::addExceptionProgram( const char* filename, const char* progname, bool defer)
{
    return m_cfg->addExceptionProgram(filename, progname, defer);
}

void OContext::setMissProgram( unsigned int index, const char* filename, const char* progname, bool defer )
{
    m_cfg->setMissProgram(index, filename, progname, defer);
}



void OContext::close()
{
    if(m_closed) return ; 

    m_closed = true ; 

    unsigned int num = m_cfg->getNumEntryPoint() ;

    LOG(info) << "OContext::close numEntryPoint " << num ; 

    m_context->setEntryPointCount( num );
 
    if(m_verbose) m_cfg->dump("OContext::close");

    m_cfg->apply();
}


void OContext::dump(const char* msg)
{
    m_cfg->dump(msg);
}
unsigned int OContext::getNumEntryPoint()
{
    return m_cfg->getNumEntryPoint();
}




void OContext::launch(unsigned int lmode, unsigned int entry, unsigned int width, unsigned int height, STimes* times )
{
    if(!m_closed) close();

    LOG(debug)<< "OContext::launch" 
              << " entry " << entry 
              << " width " << width 
              << " height " << height 
              ;


    if(times) times->count     += 1 ; 


    if(lmode & VALIDATE)
    {
        double t0, t1 ; 
        t0 = BTimer::RealTime();

        m_context->validate();

        t1 = BTimer::RealTime();
        if(times) times->validate  += t1 - t0 ;
    }

    if(lmode & COMPILE)
    {
        double t0, t1 ; 
        t0 = BTimer::RealTime();

        m_context->compile();

        t1 = BTimer::RealTime();
        if(times) times->compile  += t1 - t0 ;
    }


    if(lmode & PRELAUNCH)
    {
        double t0, t1 ; 
        t0 = BTimer::RealTime();

        m_context->launch( entry, 0, 0); 

        t1 = BTimer::RealTime();
        if(times) times->prelaunch  += t1 - t0 ;
    }


    if(lmode & LAUNCH)
    {
        double t0, t1 ; 
        t0 = BTimer::RealTime();

        m_context->launch( entry, width, height ); 

        t1 = BTimer::RealTime();
        if(times) times->launch  += t1 - t0 ;
    }

}



template <typename T>
void OContext::upload(optix::Buffer& buffer, NPY<T>* npy)
{
    unsigned int numBytes = npy->getNumBytes(0) ;

    OpticksBufferControl ctrl(npy->getBufferControlPtr());
    bool verbose = ctrl("VERBOSE_MODE") ;

    if(ctrl(OpticksBufferControl::OPTIX_OUTPUT_ONLY_))
    { 
         LOG(warning) << "OContext::upload NOT PROCEEDING "
                      << " name " << npy->getBufferName()
                      << " as " << OpticksBufferControl::OPTIX_OUTPUT_ONLY_
                      << " desc " << npy->description("skip-upload") 
                      ;
     
    }
    else if(ctrl("UPLOAD_WITH_CUDA"))
    {
        if(verbose) LOG(info) << npy->description("UPLOAD_WITH_CUDA markDirty") ;

        void* d_ptr = NULL;
        rtBufferGetDevicePointer(buffer->get(), 0, &d_ptr);
        cudaMemcpy(d_ptr, npy->getBytes(), numBytes, cudaMemcpyHostToDevice);
        buffer->markDirty();
    }
    else
    {
        if(verbose) LOG(info) << npy->description("standard OptiX UPLOAD") ;
        memcpy( buffer->map(), npy->getBytes(), numBytes );
        buffer->unmap(); 
    }
}


template <typename T>
void OContext::download(optix::Buffer& buffer, NPY<T>* npy)
{
    assert(npy);
    OpticksBufferControl ctrl(npy->getBufferControlPtr());
    bool verbose = ctrl("VERBOSE_MODE") ;

    bool proceed = false ; 
    if(ctrl(OpticksBufferControl::OPTIX_INPUT_ONLY_))
    {
         proceed = false ; 
         LOG(warning) << "OContext::download NOT PROCEEDING "
                      << " name " << npy->getBufferName()
                      << " as " << OpticksBufferControl::OPTIX_INPUT_ONLY_
                      << " desc " << npy->description("skip-download") 
                      ;
    }
    else if(ctrl(OpticksBufferControl::COMPUTE_MODE_))
    {
         proceed = true ; 
    }
    else if(ctrl(OpticksBufferControl::OPTIX_NON_INTEROP_))
    {   
         proceed = true ;
         LOG(info) << "OContext::download PROCEED for " << npy->getBufferName() << " as " << OpticksBufferControl::OPTIX_NON_INTEROP_  ;
    }   
    
    if(proceed)
    {

        if(verbose)
             LOG(info) << " VERBOSE_MODE "  << " " << npy->description("download") ;

        void* ptr = buffer->map() ; 
        npy->read( ptr );
        buffer->unmap(); 
    }
    else
    {
        if(verbose)
             LOG(info)<< npy->description("download SKIPPED") ;

    }
}



template <typename T>
optix::Buffer OContext::createBuffer(NPY<T>* npy, const char* name)
{
    assert(npy);
    OpticksBufferControl ctrl(npy->getBufferControlPtr());
    bool verbose = ctrl("VERBOSE_MODE") ;

    bool compute = isCompute()  ; 

    if(verbose) 
       LOG(info) << "OContext::createBuffer "
              << std::setw(20) << name 
              << std::setw(20) << npy->getShapeString()
              << " mode : " << ( compute ? "COMPUTE " : "INTEROP " )
              << " BufferControl : " << ctrl.description(name)
              ;

    unsigned int type(0);
    bool noctrl = false ; 
    
    if(      ctrl("OPTIX_INPUT_OUTPUT") )  type = RT_BUFFER_INPUT_OUTPUT ;
    else if( ctrl("OPTIX_OUTPUT_ONLY")  )  type = RT_BUFFER_OUTPUT  ;
    else if( ctrl("OPTIX_INPUT_ONLY")   )  type = RT_BUFFER_INPUT  ;
    else                                   noctrl = true ; 
   
    if(noctrl) LOG(fatal) << "no buffer control for " << name << ctrl.description("") ;
    assert(!noctrl);

 
    if( ctrl("BUFFER_COPY_ON_DIRTY") )     type |= RT_BUFFER_COPY_ON_DIRTY ;


    optix::Buffer buffer ; 

    if( compute )
    {
        buffer = m_context->createBuffer(type);
    }
    else if( ctrl("OPTIX_NON_INTEROP") )
    {
        buffer = m_context->createBuffer(type);
    }
    else
    {
         int buffer_id = npy ? npy->getBufferId() : -1 ;
         if(!(buffer_id > -1))
             LOG(fatal) << "OContext::createBuffer CANNOT createBufferFromGLBO as not uploaded  "
                        << " name " << std::setw(20) << name
                        << " buffer_id " << buffer_id 
                         ; 
         assert(buffer_id > -1 );
         buffer = m_context->createBufferFromGLBO(type, buffer_id);
    } 

    configureBuffer<T>(buffer, npy, name );
    return buffer ; 
}




template <typename T>
unsigned OContext::determineBufferSize(NPY<T>* npy, const char* name)
{
    unsigned int ni = std::max(1u,npy->getShape(0));
    unsigned int nj = std::max(1u,npy->getShape(1));  
    unsigned int nk = std::max(1u,npy->getShape(2));  

    bool seed = strcmp(name, "seed")==0 ;
    RTformat format = getFormat(npy->getType(), seed);
    unsigned int size ; 

    if(format == RT_FORMAT_USER || seed)
    {
        size = ni*nj*nk ; 
    }
    else
    {
        size = npy->getNumQuads() ;  
 
    }
    return size ; 
}


template <typename T>
void OContext::configureBuffer(optix::Buffer& buffer, NPY<T>* npy, const char* name)
{

    bool seed = strcmp(name, "seed")==0 ;

    RTformat format = getFormat(npy->getType(), seed);
    buffer->setFormat(format);  // must set format, before can set ElementSize



    unsigned size = determineBufferSize(npy, name);


    const char* label ; 
    if(     format == RT_FORMAT_USER) label = "USER";
    else if(seed)                     label = "SEED";
    else                              label = "QUAD";



    std::stringstream ss ; 
    ss 
       << std::setw(10) << name
       << std::setw(20) << npy->getShapeString()
       << " " << label 
       << " size " << size ; 
       ;
    std::string hdr = ss.str();

    if(format == RT_FORMAT_USER )
    {
        buffer->setElementSize(sizeof(T));
        LOG(debug) << hdr
                  << " elementsize " << sizeof(T)
                  ;
    }
    else
    {
        LOG(debug) << hdr ;
    }
    

    buffer->setSize(size); 

    //
    // NB in interop mode, the OptiX buffer is just a reference to the 
    // OpenGL buffer object, however despite this the size
    // and format of the OptiX buffer still needs to be set as they control
    // the addressing of the buffer in the OptiX programs 
    //
    //         79 rtBuffer<float4>               genstep_buffer;
    //         80 rtBuffer<float4>               photon_buffer;
    //         ..
    //         85 rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
    //         86 rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 
    //
}


template <typename T>
void OContext::resizeBuffer(optix::Buffer& buffer, NPY<T>* npy, const char* name)
{
    OpticksBufferControl ctrl(npy->getBufferControlPtr());
    bool verbose = ctrl("VERBOSE_MODE") ;

    unsigned size = determineBufferSize(npy, name);
    buffer->setSize(size); 

    if(verbose)
    LOG(info) << "OContext::resizeBuffer " << name << " shape " << npy->getShapeString() << " size " << size  ; 
}





RTformat OContext::getFormat(NPYBase::Type_t type, bool seed)
{
    RTformat format ; 
    switch(type)
    {
        case NPYBase::FLOAT:     format = RT_FORMAT_FLOAT4         ; break ; 
        case NPYBase::SHORT:     format = RT_FORMAT_SHORT4         ; break ; 
        case NPYBase::INT:       format = RT_FORMAT_INT4           ; break ; 
        case NPYBase::UINT:      format = RT_FORMAT_UNSIGNED_INT4  ; break ; 
        case NPYBase::CHAR:      format = RT_FORMAT_BYTE4          ; break ; 
        case NPYBase::UCHAR:     format = RT_FORMAT_UNSIGNED_BYTE4 ; break ; 
        case NPYBase::ULONGLONG: format = RT_FORMAT_USER           ; break ; 
        case NPYBase::DOUBLE:    format = RT_FORMAT_USER           ; break ; 
    }

    if(seed)
    {
         assert(type == NPYBase::UINT);
         format = RT_FORMAT_UNSIGNED_INT ;
         LOG(debug) << "OContext::getFormat override format for seed " ; 
    }
    return format ; 
}




void OContext::snap(const char* path)
{
    optix::Buffer output_buffer = m_context["output_buffer"]->getBuffer() ; 

    RTsize width, height, depth ;
    output_buffer->getSize(width, height, depth);

    LOG(info) 
         << "OContext::snap"
         << " path " << path 
         << " width " << width
         << " width " << (int)width
         << " height " << height
         << " height " << (int)height
         << " depth " << depth
         ;   

    void* ptr = output_buffer->map() ; 

    int ncomp = 4 ;   
    SPPM::write(path,  (unsigned char*)ptr , width, height, ncomp );

    output_buffer->unmap(); 
}


void OContext::save(const char* path)
{
    optix::Buffer output_buffer = m_context["output_buffer"]->getBuffer() ;

    RTsize width, height, depth ;
    output_buffer->getSize(width, height, depth);

    LOG(info)
         << "OContext::save"
         << " width " << width
         << " width " << (int)width
         << " height " << height
         << " height " << (int)height
         << " depth " << depth
         ;

    NPY<unsigned char>* npy = NPY<unsigned char>::make(width, height, 4) ;
    npy->zero();

    void* ptr = output_buffer->map() ;
    npy->read( ptr );

    output_buffer->unmap();

    npy->save(path);
}







template OXRAP_API void OContext::upload<unsigned>(optix::Buffer&, NPY<unsigned>* );
template OXRAP_API void OContext::download<unsigned>(optix::Buffer&, NPY<unsigned>* );
template OXRAP_API void OContext::resizeBuffer<unsigned>(optix::Buffer&, NPY<unsigned>*, const char* );

template OXRAP_API void OContext::upload<float>(optix::Buffer&, NPY<float>* );
template OXRAP_API void OContext::download<float>(optix::Buffer&, NPY<float>* );
template OXRAP_API void OContext::resizeBuffer<float>(optix::Buffer&, NPY<float>*, const char* );

template OXRAP_API void OContext::upload<short>(optix::Buffer&, NPY<short>* );
template OXRAP_API void OContext::download<short>(optix::Buffer&, NPY<short>* );
template OXRAP_API void OContext::resizeBuffer<short>(optix::Buffer&, NPY<short>*, const char* );

template OXRAP_API void OContext::upload<unsigned long long>(optix::Buffer&, NPY<unsigned long long>* );
template OXRAP_API void OContext::download<unsigned long long>(optix::Buffer&, NPY<unsigned long long>* );
template OXRAP_API void OContext::resizeBuffer<unsigned long long>(optix::Buffer&, NPY<unsigned long long>*, const char* );

template OXRAP_API optix::Buffer OContext::createBuffer(NPY<float>*, const char* );
template OXRAP_API optix::Buffer OContext::createBuffer(NPY<short>*, const char* );
template OXRAP_API optix::Buffer OContext::createBuffer(NPY<unsigned long long>*, const char* );
template OXRAP_API optix::Buffer OContext::createBuffer(NPY<unsigned>*, const char* );


