#include "GScintillatorLib.hh"
#include "OScintillatorLib.hh"

#include "OTextureTest.hh"
#include "OContext.hh"
#include "Opticks.hh"

#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "OXRAP_LOG.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);    


    OKCORE_LOG__ ; 
    GGEO_LOG__ ; 
    OXRAP_LOG__ ; 


    Opticks* m_opticks(NULL);
    m_opticks = new Opticks(argc, argv);
    m_opticks->configure();


    GScintillatorLib* slib = GScintillatorLib::load(m_opticks);
    slib->dump();


    optix::Context context = optix::Context::create();

    OScintillatorLib* oscin ;  
    oscin = new OScintillatorLib(context, slib );

    //const char* slice = NULL ; 
    const char* slice = "0:1" ; 
    oscin->convert(slice);

    OContext::Mode_t mode = m_opticks->isCompute() ? OContext::COMPUTE : OContext::INTEROP ;

    OContext* m_ocontext(NULL);
    m_ocontext = new OContext(context, mode);

    optix::Group top = m_ocontext->getTop();


    const char* builder = "NoAccel" ;
    const char* traverser = "NoAccel" ; 
    optix::Acceleration acceleration = context->createAcceleration(builder, traverser);
    top->setAcceleration(acceleration);

 
    OTextureTest* m_ott(NULL);
    m_ott = new OTextureTest(m_ocontext, m_opticks);
    m_ott->launch();

    LOG(info) << "DONE" ; 


    return 0 ;     
}

/*

Without the slicing::

    2016-07-07 18:24:11.036 INFO  [14856619] [OScintillatorLib::makeReemissionTexture@46] OScintillatorLib::makeReemissionTexture  nx 4096 ny 1 ni 2 nj 4096 nk 1 step 0.000244141 empty 0
    2016-07-07 18:24:11.036 INFO  [14856619] [OPropertyLib::upload@23] 32768
    2016-07-07 18:24:11.036 INFO  [14856619] [OPropertyLib::upload@25] 0x7ffa91008c00
    2016-07-07 18:24:11.037 INFO  [14856619] [OContext::init@126] OContext::init  mode INTEROP num_ray_type 3
    OOTextureTest(53959,0x7fff74d63310) malloc: *** error for object 0x7ffa90ddd208: incorrect checksum for freed object - object was probably modified after being freed.
    *** set a breakpoint in malloc_error_break to debug
    Abort trap: 6
    simon:optixrap blyth$ 


With OptiX 4.0.0::

    2016-08-10 15:05:35.709 INFO  [1903430] [OContext::launch@214] OContext::launch entry 0 width 1 height 1
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Invalid value (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Unsupported combination of texture index, wrap and filter modes:  RT_TEXTURE_INDEX_ARRAY_INDEX, RT_WRAP_REPEAT, RT_FILTER_LINEAR, file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Util/TextureDescriptor.cpp, line: 138)
    Abort trap: 6
    simon:~ blyth$ 





*/



