
#include "OContext.hh"
#include "OTimes.hh"
#include "OConfig.hh"

// npy-
#include "timeutil.hpp"
#include "NPY.hpp"

#include <iomanip>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

using namespace optix ; 


void OContext::init()
{
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    //m_context->setPrintLaunchIndex(0,0,0);
    m_context->setStackSize( 2180 ); // TODO: make externally configurable, and explore performance implications

    m_context->setEntryPointCount( getNumEntryPoint() );  
    m_context->setRayTypeCount( getNumRayType() );
    m_top = m_context->createGroup();

    m_cfg = OConfig::makeInstance(m_context);
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
void OContext::setRayGenerationProgram( unsigned int index , const char* filename, const char* progname )
{
    m_cfg->setRayGenerationProgram(index, filename, progname);
}
void OContext::setExceptionProgram( unsigned int index , const char* filename, const char* progname )
{
    m_cfg->setExceptionProgram(index, filename, progname);
}
void OContext::setMissProgram( unsigned int index , const char* filename, const char* progname )
{
    m_cfg->setMissProgram(index, filename, progname);
}
 



void OContext::launch(unsigned int entry, unsigned int width, unsigned int height, OTimes* times)
{
    LOG(info)<< "OContext::launch";

    double t0,t1,t2,t3,t4 ; 

    t0 = getRealTime();
    m_context->validate();
    t1 = getRealTime();
    m_context->compile();
    t2 = getRealTime();
    //m_context->launch( entry, 0); 
    m_context->launch( entry, 0, 0); 
    t3 = getRealTime();
    m_context->launch( entry, width, height ); 
    t4 = getRealTime();

    if(times)
    {
        times->count     += 1 ; 
        times->validate  += t1 - t0 ;
        times->compile   += t2 - t1 ; 
        times->prelaunch += t3 - t2 ; 
        times->launch    += t4 - t3 ; 
    }
}



template <typename T>
void OContext::upload(optix::Buffer& buffer, NPY<T>* npy)
{
    unsigned int numBytes = npy->getNumBytes(0) ;

    LOG(info)<<"OContext::upload" 
             << " numBytes " << numBytes 
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
             ;

    void* ptr = buffer->map() ; 
    npy->read( ptr );
    buffer->unmap(); 
}




template <typename T>
optix::Buffer OContext::createIOBuffer(NPY<T>* npy, const char* name, bool interop)
{
    assert(npy);
    unsigned int ni = npy->getShape(0);
    unsigned int nj = npy->getShape(1);  
    unsigned int nk = npy->getShape(2);  

    Buffer buffer;
    if(interop)
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
        LOG(info) << "OContext::createIOBuffer (COMPUTE)" ;
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
                  << " size " << size 
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


template optix::Buffer OContext::createIOBuffer(NPY<float>*, const char*, bool );
template optix::Buffer OContext::createIOBuffer(NPY<short>*, const char*, bool );
template optix::Buffer OContext::createIOBuffer(NPY<unsigned long long>*, const char*, bool );


