#include <stdio.h>
#include <optix.h>
#include <sutil.h>

int main(int argc, char** argv)
{
    unsigned int num_devices;
    unsigned int version;

    RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetDeviceCount(&num_devices));
    RT_CHECK_ERROR_NO_CONTEXT(rtGetVersion(&version));
    printf("OptiX %d.%d.%d\n", version/1000, (version%1000)/10, version%10);
    printf("Number of Devices = %d\n\n", num_devices);

    //RTcontext context;

    return 0 ;
}

