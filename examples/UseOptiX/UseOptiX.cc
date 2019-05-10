#include <cstring>
#include <cstdlib>
#include <string>
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




int main(int argc, char** argv)
{

    bool ctx(false); 

    char buf[256] ; 
    for(int i=1 ; i < argc ; i++)
    {
        if(i < argc - 1 && strcmp(argv[i], "--cvd") == 0) 
        {
            snprintf(buf, 256, "CUDA_VISIBLE_DEVICES=%s", argv[i+1]) ; 
            printf("setting envvar internally : %s\n", buf );
            putenv(buf);  
        }
        else if( i < argc && strcmp(argv[i], "--ctx" ) == 0)
        {
            ctx = true ;    
        } 
    }

    // extracts from /Developer/OptiX/SDK/optixDeviceQuery/optixDeviceQuery.cpp

    unsigned num_devices;
    unsigned version;


    RT_CHECK_ERROR(rtDeviceGetDeviceCount(&num_devices));
    RT_CHECK_ERROR(rtGetVersion(&version));

    printf("OptiX version %d major.minor.micro %d.%d.%d   Number of devices = %d \n\n", version, version / 10000, (version % 10000) / 100, version % 100, num_devices ); // major.minor.micro

    for(unsigned i = 0; i < num_devices; ++i) 
    {
            char name[256];
            int computeCaps[2];
            RTsize total_mem;

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name));
            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps));
            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(total_mem), &total_mem));

            printf(" Device %d: %30s ", i, name);
            printf(" Compute Support: %d %d ", computeCaps[0], computeCaps[1]);
            printf(" Total Memory: %llu bytes \n", (unsigned long long)total_mem);
    } 



    if(!ctx) return 0 ; 

    RTcontext context = 0;

    //RTprogram program;
    RTbuffer  buffer;
    RTvariable variable ; 

    int width = 1024u ; 
    int height = 768u ; 
   
    std::cout << std::endl << std::endl ; 
    std::cout << "( creating context " << std::endl ; 
    RT_CHECK_ERROR( rtContextCreate( &context ) );
    std::cout << ") creating context " << std::endl ; 
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

