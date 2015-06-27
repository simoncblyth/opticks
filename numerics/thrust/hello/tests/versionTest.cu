/*
https://code.google.com/p/thrust/wiki/QuickStartGuide

simon:hello blyth$ nvcc version.cu -o version
simon:hello blyth$ ./version 
Thrust v1.7

*/


#include <thrust/version.h>
#include <iostream>

int main(void)
{
    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;

    std::cout << "Thrust v" << major << "." << minor << std::endl;

    return 0;
}
