#include <sstream>
#include <string>
#include <cstring>
#include <cassert>
#include <optix.h>

#include "OKConf.hh"

// from SDK/sutil/sutil.h

struct APIError
{   
    APIError( RTresult c, const std::string& f, int l ) 
        : code( c ), file( f ), line( l ) {}
    RTresult     code;
    std::string  file;
    int          line;
};

// Error check/report helper for users of the C API 
#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      throw APIError( code, __FILE__, __LINE__ );           \
  } while(0)



std::string ptx_path(const char* target, const char* cu_name)
{
    std::stringstream ss ; 
    ss << OKConf::OpticksInstallPrefix()
       << "/installcache/PTX/"
       << target
       << "_generated_"
       << cu_name
       << ".ptx" 
       ;
    return ss.str();
}



int main()
{

    // extracts from /Developer/OptiX/SDK/optixDeviceQuery/optixDeviceQuery.cpp

    unsigned num_devices;
    unsigned version;

    RT_CHECK_ERROR(rtDeviceGetDeviceCount(&num_devices));
    RT_CHECK_ERROR(rtGetVersion(&version));

    printf("OptiX %d.%d.%d\n", version / 10000, (version % 10000) / 100, version % 100); // major.minor.micro
    printf("Number of Devices = %d\n\n", num_devices);

    for(unsigned i = 0; i < num_devices; ++i) 
    {
            char name[256];
            int computeCaps[2];
            RTsize total_mem;

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name));
            printf("Device %d: %s\n", i, name);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps));
            printf("  Compute Support: %d %d\n", computeCaps[0], computeCaps[1]);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(total_mem), &total_mem));
            printf("  Total Memory: %llu bytes\n", (unsigned long long)total_mem);
    } 


    RTcontext context = 0;

    RTprogram raygen ;
    RTbuffer  buffer;
    RTvariable variable ; 

    int width = 1024u ; 
    int height = 768u ; 

    RT_CHECK_ERROR( rtContextCreate( &context ) );
    RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 1 ) );
    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( buffer, RT_FORMAT_FLOAT4 ) );
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, width, height ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "variable", &variable ) );
    RT_CHECK_ERROR( rtContextSetPrintEnabled( context, 1 ) );
    RT_CHECK_ERROR( rtContextSetPrintBufferSize( context, 4096 ) );

    RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) ); 

    const char* cmake_target = "UseOptiXProgram" ;
    const char* cu_name = "UseOptiXProgram_minimal.cu" ; 

    std::string ptx_filename = ptx_path(cmake_target, cu_name );  
    const char* raygen_ptx_filename = ptx_filename.c_str();
    
    //const char* raygen_ptx_filename_2 = "/usr/local/opticks/installcache/PTX/UseOptiXProgram_generated_UseOptiXProgram_minimal.cu.ptx"  ;
    //assert(strcmp( raygen_ptx_filename, raygen_ptx_filename_2) == 0 );

    const char* raygen_program_name = "basicTest" ;


    RT_CHECK_ERROR( rtProgramCreateFromPTXFile(context, raygen_ptx_filename, raygen_program_name, &raygen )) ;

    unsigned entry_point_index = 0u ;  
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, entry_point_index, raygen )); 

    RTsize w = 10u ; 
    RT_CHECK_ERROR( rtContextLaunch1D( context, entry_point_index, w ));

    return 0 ; 
}

