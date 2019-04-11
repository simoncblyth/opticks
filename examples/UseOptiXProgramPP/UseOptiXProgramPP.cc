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



void printUsageAndExit(const char* name)
{
    std::cout 
        << "Expect either zero, one or two arguments eg::" << std::endl 
        << std::endl 
        << "    " << name << std::endl 
        << "    " << name << " " << "basicTest.cu            ## progname defaults to basicTest here " << std::endl 
        << "    " << name << " " << "basicTest.cu printTest  ## progname specified as printTest " << std::endl 
        << std::endl 
        ;
    exit(1); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ; 

    if((argc > 1 && ( strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 )) || argc > 3) printUsageAndExit(argv[0]);  

    const char* cmake_target = "UseOptiXProgramPP" ; 
    const char* cu_name = NULL ;  
    const char* progname = NULL  ;

    if(argc == 1)
    {
        cu_name = "basicTest.cu" ;
        progname = "basicTest" ;  
    } 
    else if( argc == 2)
    {
        std::string arg = argv[1] ;
        std::string base = arg.substr(0, arg.find_last_of(".")) ; 
        cu_name = strdup( arg.c_str() ); 
        progname = strdup( base.c_str() ) ; 
    }
    else if( argc == 3)
    {
        cu_name = argv[1] ; 
        progname = argv[2] ; 
    }

    const char* ptx_path = OKConf::PTXPath( cmake_target, cu_name ); 
    for(int i=0 ; i < argc ; i++) std::cout << argv[i] << " " ;  

    std::cout 
        << std::endl
        << " cmake_target : " << cmake_target << std::endl
        << " cu_name (1)  : " << cu_name << std::endl 
        << " progname (2) : " << progname << std::endl 
        << " ptx_path     : " << ptx_path << std::endl 
        ;




    optix::Context context = optix::Context::create();

    context->setRayTypeCount(1); 

    context->setPrintEnabled(true); 
    //context->setPrintLaunchIndex(5,0,0); 
    context->setPrintBufferSize(4096); 
    context->setEntryPointCount(1); 


    optix::Program program = context->createProgramFromPTXFile( ptx_path , progname );  
    unsigned entry_point_index = 0u ;
    context->setRayGenerationProgram( entry_point_index, program ); 

    unsigned width = 10u ; 
    context->launch( entry_point_index , width  ); 

    LOG(info) << argv[0] ; 

    return 0 ; 
}


