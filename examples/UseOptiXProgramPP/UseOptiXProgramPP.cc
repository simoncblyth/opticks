/**
UseOptiXProgramPP
===================

NB no oxrap : aiming to operate at lower level in here
as preliminary to finding whats going wrong with 6.0.0

**/


#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

#include "OPTICKS_LOG.hh"
#include "OKConf.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ; 

    optix::Context context = optix::Context::create();

    context->setRayTypeCount(1); 

    context->setPrintEnabled(true); 
    //context->setPrintLaunchIndex(5,0,0); 
    context->setPrintBufferSize(4096); 
    context->setEntryPointCount(1); 

    const char* cmake_target = "UseOptiXProgramPP" ; 
    const char* cu_name = "UseOptiXProgramPP_minimal.cu" ;  
    const char* path = OKConf::PTXPath( cmake_target, cu_name ); 
    const char* progname = "basicTest" ;

    optix::Program program = context->createProgramFromPTXFile( path , progname );  
    unsigned entry_point_index = 0u ;
    context->setRayGenerationProgram( entry_point_index, program ); 

    unsigned width = 10u ; 
    context->launch( entry_point_index , width  ); 

    LOG(info) << argv[0] ; 

    return 0 ; 
}


