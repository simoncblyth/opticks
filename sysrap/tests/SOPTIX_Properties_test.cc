#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cassert>

#include "OPTIX_CHECK.h"

#include "SOPTIX_Properties.h"


struct SOPTIX_Properties_test
{
    int device_id ;
    OptixDeviceContextOptions options = {};
    OptixDeviceContext context = nullptr;
    int init_rc ;
    SOPTIX_Properties* props ;

    static int Init(int device_id, OptixDeviceContextOptions& options, OptixDeviceContext& context);

    SOPTIX_Properties_test(int device_id);
    std::string desc() const ;
    virtual ~SOPTIX_Properties_test();
};


inline int SOPTIX_Properties_test::Init(int device_id, OptixDeviceContextOptions& options, OptixDeviceContext& context)
{
    // 1. Select the target CUDA device and initialize driver/context
    cudaSetDevice(device_id);
    cudaFree(0);

    // 2. Initialize OptiX function table
    OPTIX_CHECK( optixInit() );

    // 3. Create the OptiX Device Context (using default null CUcontext for current device)
    OPTIX_CHECK( optixDeviceContextCreate(static_cast<CUcontext>(0), &options, &context) );

    return 0 ;
}

inline SOPTIX_Properties_test::SOPTIX_Properties_test(int device_id_)
    :
    device_id(device_id_),
    init_rc(Init(device_id, options, context)),
    props(new SOPTIX_Properties(context))
{
}

inline std::string SOPTIX_Properties_test::desc() const
{
    std::stringstream ss ;
    ss
       << "[SOPTIX_Properties_test device_id "
       << device_id
       << "\n"
       << props->desc()
       << "\n"
       << "]SOPTIX_Properties_test device_id "
       << "\n"
       ;
    std::string str = ss.str() ;
    return str ;
}

inline  SOPTIX_Properties_test::~SOPTIX_Properties_test()
{
    // 5. Clean up context
    OPTIX_CHECK_NOTHROW(optixDeviceContextDestroy(context));
}



int main()
{
    SOPTIX_Properties_test t(0);
    std::cout << t.desc();
    return 0 ;
}

