#include <optix.h>
#include <optix_stubs.h>

#include <optix_function_table_definition.h>

#include <cuda_runtime.h>
#include <iostream>

#include "OPTIX_CHECK.h"


void check_DEVICE_PROPERTY_RTCORE_VERSION(int device_id)
{
    // 1. Select the target CUDA device and initialize driver/context
    cudaSetDevice(device_id);
    cudaFree(0);

    // 2. Initialize OptiX function table
    OPTIX_CHECK( optixInit() );

    // 3. Create the OptiX Device Context (using default null CUcontext for current device)
    OptixDeviceContext context = nullptr;
    OptixDeviceContextOptions options = {};
    OPTIX_CHECK( optixDeviceContextCreate(static_cast<CUcontext>(0), &options, &context) );

    // 4. Query the property using optixDeviceContextGetProperty
    unsigned int rtCoreVersion = 0;
    OPTIX_CHECK( optixDeviceContextGetProperty(
        context,
        OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
        &rtCoreVersion,
        sizeof(unsigned int)
    ));

    std::cout << "Device " << device_id
              << " OPTIX_DEVICE_PROPERTY_RTCORE_VERSION: "
              << rtCoreVersion << std::endl;

    switch (rtCoreVersion) {
        case 0:
            std::cout << "No hardware RT acceleration (Software/Emulated fallback)." << std::endl;
            break;
        case 10:
            std::cout << "1st Gen RT Cores (Turing architecture)." << std::endl;
            break;
        case 20:
            std::cout << "2nd Gen RT Cores (Ampere architecture)." << std::endl;
            break;
        case 30:
            std::cout << "3rd Gen RT Cores (Ada Lovelace / Hopper architectures)." << std::endl;
            break;
        default:
            std::cout << "Hardware RT core present (Gen version " << rtCoreVersion << ")." << std::endl;
            break;
    }


    // 5. Clean up context
    OPTIX_CHECK(optixDeviceContextDestroy(context));
}


int main()
{
    check_DEVICE_PROPERTY_RTCORE_VERSION(0);
    return 0 ;
}

