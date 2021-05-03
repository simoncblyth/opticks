#include <iostream>

#include <optix.h>

#if OPTIX_VERSION >= 70000
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#else
#include <optixu/optixpp_namespace.h>
#endif

int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ; 

#if OPTIX_VERSION >= 50000 && OPTIX_VERSION < 60000
    std::cout << " OPTIX 5  " << std::endl ; 

    std::cout << "[ create context " << std::endl ; 
    optix::Context context = optix::Context::create() ;  
    std::cout << "] create context " << std::endl ; 

#elif OPTIX_VERSION >= 60000 && OPTIX_VERSION < 70000
    std::cout << " OPTIX 6  " << std::endl ; 

    std::cout << "[ create context " << std::endl ; 
    optix::Context context = optix::Context::create() ;  
    std::cout << "] create context " << std::endl ; 


#elif OPTIX_VERSION >= 70000 && OPTIX_VERSION < 80000
    std::cout << " OPTIX 7  " << std::endl ; 
#else
    std::cout << "ERROR : UNEXPECTED OPTIX_VERSI " << std::endl ; 
#endif

    return 0 ; 
}

