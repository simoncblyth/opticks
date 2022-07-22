#include <iostream>
#include "OPTICKS_LOG.hh"
#include "SFrameConfig.hh"

void test_Desc()
{
    LOG(info); 

    //SFrameConfig::SetFrameMask("isect,fphoton,pixel"); 

    std::cout << SFrameConfig::Desc() << std::endl ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_Desc(); 

    return 0 ; 
}
