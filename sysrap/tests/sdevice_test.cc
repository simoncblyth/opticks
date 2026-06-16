/**
sdevice_test.cc
==================

See notes in sdevice_test.sh

**/


#include "sdevice.h"

int main(int argc, char** argv)
{

    int device_count = sdevice::DeviceCount();
    std::cout
        << " device_count " << device_count
        << "\n"
        ;



    std::vector<sdevice> visible ;
    sdevice::Visible(visible );

    std::cout
       << "[sdevice::Desc(visible)\n"
       << sdevice::Desc( visible )
       << "]sdevice::Desc(visible)\n"
       << "[sdevice::Brief(visible)\n"
       << sdevice::Brief( visible )  << "\n"
       << "]sdevice::Brief(visible)\n"
       ;

    return 0 ;
}
