#include <string>
//#include <optix.h>
#include <optix_world.h>

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

    //RTprogram program;
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


    RTformat format = RT_FORMAT_FLOAT4 ; 
    size_t size = 0 ; 
    RT_CHECK_ERROR( rtuGetSizeForRTformat( format, &size) ); 

    std::cout << " RT_FORMAT_FLOAT4 size " << size << std::endl ; 
    assert( size == sizeof(float)*4 ) ; 



    return 0 ; 
}

