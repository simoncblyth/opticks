
#include "OContext.hh"
#include "OTimes.hh"
#include "OConfig.hh"

// npy-
#include "timeutil.hpp"
#include "NPY.hpp"
#include "GLMFormat.hpp"

#include <iomanip>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

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
    }
}


void OContext::init()
{
    m_cfg = new OConfig(m_context);

    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    //m_context->setPrintLaunchIndex(0,0,0);

    unsigned int num_ray_type = getNumRayType() ;

    m_context->setRayTypeCount( num_ray_type );   // more static than entry type count

    m_top = m_context->createGroup();

    m_context[ "top_object" ]->set( m_top );

    LOG(info) << "OContext::init " 
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


unsigned int OContext::addRayGenerationProgram( const char* filename, const char* progname )
{
    return m_cfg->addRayGenerationProgram(filename, progname, true);
}
unsigned int OContext::addExceptionProgram( const char* filename, const char* progname )
{
    return m_cfg->addExceptionProgram(filename, progname, true);
}



void OContext::setMissProgram( unsigned int index, const char* filename, const char* progname )
{
    m_cfg->setMissProgram(index, filename, progname, true);
}



void OContext::close()
{
    if(m_closed) return ; 

    m_closed = true ; 

    unsigned int num = m_cfg->getNumEntryPoint() ;

    LOG(info) << "OContext::close numEntryPoint " << num ; 

    m_context->setEntryPointCount( num );
  
    m_cfg->dump("OContext::close");

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




void OContext::launch(unsigned int lmode, unsigned int entry, unsigned int width, unsigned int height, OTimes* times )
{
    if(!m_closed) close();

    LOG(info)<< "OContext::launch" 
              << " entry " << entry 
              << " width " << width 
              << " height " << height 
              ;


    if(times) times->count     += 1 ; 


    if(lmode & VALIDATE)
    {
        double t0, t1 ; 
        t0 = getRealTime();

        m_context->validate();

        t1 = getRealTime();
        if(times) times->validate  += t1 - t0 ;
    }

    if(lmode & COMPILE)
    {
        double t0, t1 ; 
        t0 = getRealTime();

        m_context->compile();

        t1 = getRealTime();
        if(times) times->compile  += t1 - t0 ;
    }


    if(lmode & PRELAUNCH)
    {
        double t0, t1 ; 
        t0 = getRealTime();

        m_context->launch( entry, 0, 0); 

        t1 = getRealTime();
        if(times) times->prelaunch  += t1 - t0 ;
    }


    if(lmode & LAUNCH)
    {
        double t0, t1 ; 
        t0 = getRealTime();

        m_context->launch( entry, width, height ); 

        t1 = getRealTime();
        if(times) times->launch  += t1 - t0 ;
    }

}





template <typename T>
void OContext::upload(optix::Buffer& buffer, NPY<T>* npy)
{
    unsigned int numBytes = npy->getNumBytes(0) ;

    LOG(info)<<"OContext::upload" 
             << " numBytes " << numBytes 
             << npy->description("upload")
             ;

    memcpy( buffer->map(), npy->getBytes(), numBytes );
    buffer->unmap(); 
}


template <typename T>
void OContext::download(optix::Buffer& buffer, NPY<T>* npy)
{
    unsigned int numBytes = npy->getNumBytes(0) ;
    LOG(info)<<"OContext::download" 
             << " numBytes " << numBytes 
             << npy->description("download")
             ;

    void* ptr = buffer->map() ; 
    npy->read( ptr );
    buffer->unmap(); 
}




template <typename T>
optix::Buffer OContext::createIOBuffer(NPY<T>* npy, const char* name)
{
    assert(npy);
    unsigned int ni = npy->getShape(0);
    unsigned int nj = npy->getShape(1);  
    unsigned int nk = npy->getShape(2);  
    //unsigned int nl = npy->getShape(3);  

    bool compute = isCompute();

    Buffer buffer;
    if(!compute)
    {
        int buffer_id = npy ? npy->getBufferId() : -1 ;
        if(buffer_id > -1 )
        {
            buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, buffer_id);
            LOG(debug) << "OContext::createIOBuffer (INTEROP) createBufferFromGLBO " 
                      << " name " << std::setw(20) << name
                      << " buffer_id " << buffer_id 
                      << " ( " << ni << "," << nj << "," << nk << ")"
                      ;
        } 
        else
        {
            LOG(warning) << "OContext::createIOBuffer CANNOT createBufferFromGLBO as not uploaded  "
                         << " name " << std::setw(20) << name
                         << " buffer_id " << buffer_id 
                         ; 

            assert(0);  // only recsel buffer is not uploaded, as kludge interop workaround 
            //return buffer ; 
        }
    } 
    else
    {
        LOG(info) << "OContext::createIOBuffer" 
                  << " name " << name
                  << " desc " << npy->description("[COMPUTE]")
                  ;
        buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
    }


    RTformat format = getFormat(npy->getType());
    buffer->setFormat(format);  // must set format, before can set ElementSize

    unsigned int size ; 
    if(format == RT_FORMAT_USER)
    {
        buffer->setElementSize(sizeof(T));
        size = ni*nj*nk ; 
        LOG(debug) << "OContext::createIOBuffer "
                  << " RT_FORMAT_USER " 
                  << " elementsize " << sizeof(T)
                  << " size " << size 
                  ;
    }
    else
    {
        size = ni*nj ; 
        LOG(debug) << "OContext::createIOBuffer "
                  << " (quad) " 
                  << " size (ni*nj) " << size 
                  ;

    }

    buffer->setSize(size); // TODO: check without thus, maybe unwise when already referencing OpenGL buffer of defined size
    return buffer ; 
}




RTformat OContext::getFormat(NPYBase::Type_t type)
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
    return format ; 
}














template void OContext::upload<float>(optix::Buffer&, NPY<float>* );
template void OContext::download<float>(optix::Buffer&, NPY<float>* );

template void OContext::upload<short>(optix::Buffer&, NPY<short>* );
template void OContext::download<short>(optix::Buffer&, NPY<short>* );

template void OContext::upload<unsigned long long>(optix::Buffer&, NPY<unsigned long long>* );
template void OContext::download<unsigned long long>(optix::Buffer&, NPY<unsigned long long>* );


template optix::Buffer OContext::createIOBuffer(NPY<float>*, const char*);
template optix::Buffer OContext::createIOBuffer(NPY<short>*, const char*);
template optix::Buffer OContext::createIOBuffer(NPY<unsigned long long>*, const char*);


