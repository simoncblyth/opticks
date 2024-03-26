/**
sppm_test.cc
=============

~/o/sysrap/tests/sppm_test.sh
~/o/sysrap/tests/sppm_test.cc


**/
#include <cstdlib>
#include "sppm.h"

int main()
{
    int width = 1024 ;  
    int height = 768 ; 
    int ncomp = 4 ;  

    unsigned char* data = sppm::CreateImageData(width, height, ncomp ); 

    bool yflip = true ; 
    const char* path = getenv("PPM_PATH"); 
    sppm::Write(path, width, height, 4, data, yflip); 

    return 0 ; 
}
