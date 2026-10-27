// ~/o/sysrap/tests/SKernelProfile_test.sh

#include <cuda_runtime.h>

#include <cassert>
#include "ssys.h"
#include "SKernelProfile.h"

struct SKernelProfile_test
{
    static int ctor();
    static int main();
};

inline int SKernelProfile_test::ctor()
{
    SKernelProfile* kernel_profile = new SKernelProfile {} ;
    SKernelProfile& kp = *kernel_profile ;
    std::cout << "After ctor\n" << kp.desc() << "\n" ;

    skernel_profile* d_profile = kp.getDevicePtr();
    std::cout << " d_profile " << std::hex << d_profile << std::dec  << "\n" ;
    assert( d_profile == nullptr );


    kp.dummy_populate();
    std::cout << "After dummy_populate\n" << kp.desc() << "\n" ;

    kp.upload();
    std::cout << "After upload\n" << kp.desc() << "\n" ;

    kp.zero_host();
    std::cout << "After zero_host\n" << kp.desc() << "\n" ;

    kp.download();
    std::cout << "After download\n" << kp.desc() << "\n" ;

    cudaDeviceSynchronize();
    std::cout << "After sync\n" << kp.desc() << "\n" ;

    kp.zero_device();
    kp.download();
    std::cout << "After zero_device and download\n" << kp.desc() << "\n" ;


    return 0 ;
}

inline int SKernelProfile_test::main()
{
    const char* test = "ctor" ;
    const char* TEST = ssys::getenvvar("TEST", test );
    bool ALL = 0 == strcmp(TEST, "ALL") ;

    std::cout << "[SKernelProfile_test::main" << " TEST[" << TEST << "]\n" ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"ctor")) rc += ctor() ;

    std::cout << "]SKernelProfile_test::main" << " TEST[" << TEST << "]" << " RC " << rc << "\n" ;
    return rc ;
}


int main(){ return SKernelProfile_test::main(); }



