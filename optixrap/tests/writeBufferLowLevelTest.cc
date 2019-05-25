
#include "OXPPNS.hh"
#include "OKConf.hh"
#include "NPY.hpp"

#include "OPTICKS_LOG.hh"


int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);

    optix::Context context = optix::Context::create();
    context->setEntryPointCount(1);
    context->setRayTypeCount(1);    // <-- without this segments at launch (new behaviour in OptiX_600)  
    context->setPrintEnabled(true);

    unsigned ni = 100 ; 
    unsigned nj = 4 ; 
    unsigned nk = 4 ; 

    NPY<float>* npy = NPY<float>::make(ni, nj, nk) ;
    npy->zero();

    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT ) ;
    buffer->setFormat( RT_FORMAT_FLOAT4 );  
    buffer->setSize( ni*nj ) ; 

    context["output_buffer"]->set(buffer);


    const char* cmake_target = "writeBufferLowLevelTest" ; 
    const char* cu_name = "writeBufferLowLevelTest.cu" ;
    const char* ptxrel = "tests" ; 
    const char* ptx_path = OKConf::PTXPath(cmake_target, cu_name, ptxrel ); 
    const char* progname = "writeBuffer" ; 
    optix::Program program = context->createProgramFromPTXFile( ptx_path , progname ); 

    unsigned entry = 0 ; 
    context->setRayGenerationProgram( entry, program ); 

    unsigned width = ni ; 
    context->launch( entry, width  );

    NPYBase::setGlobalVerbose();

    npy->dump();
    npy->save("$TMP/writeBufferLowLevelTest.npy");

    return 0;
}
