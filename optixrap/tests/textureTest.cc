#include "GScintillatorLib.hh"
#include "OScintillatorLib.hh"

#include "OLaunchTest.hh"
#include "OContext.hh"
#include "Opticks.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    

    Opticks* ok(NULL);
    ok = new Opticks(argc, argv, "--compute");
    ok->configure();

    GScintillatorLib* slib = GScintillatorLib::load(ok);
    slib->dump();


    const char* cmake_target = "textureTest" ;
    const char* ptxrel = "tests" ;  
    OContext* m_ocontext = OContext::Create( ok, cmake_target, ptxrel );  
    optix::Context context = m_ocontext->getContext() ; 

    OScintillatorLib* oscin ;  
    oscin = new OScintillatorLib(context, slib );

    //const char* slice = NULL ; 
    const char* slice = "0:1" ; 
    oscin->convert(slice);

    optix::Group top = m_ocontext->getTopGroup();


    const char* builder = "NoAccel" ;
    const char* traverser = "NoAccel" ; 
    optix::Acceleration acceleration = context->createAcceleration(builder, traverser);
    top->setAcceleration(acceleration);

    OLaunchTest* m_ott(NULL);
    m_ott = new OLaunchTest(m_ocontext, ok, "textureTest.cu", "textureTest", "exception");
    m_ott->launch();

    LOG(info) << "DONE" ; 


    delete m_ocontext ; 

    //optix::cudaDeviceSynchronize();


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



