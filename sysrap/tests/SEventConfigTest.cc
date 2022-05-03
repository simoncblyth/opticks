#include <iostream>
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"

void test_Desc()
{
    SEventConfig::SetMaxPhoton(101); 
    std::cout << SEventConfig::Desc() << std::endl ; 
}
void test_OutPath()
{
    const char* path_0 = SEventConfig::OutPath("stem", 101, ".npy" ); 
    const char* path_1 = SEventConfig::OutPath("local_reldir", "stem", 101, ".npy" ); 
    LOG(info) << " SEventConfig::OutPath path_0 " << path_0 ; 
    LOG(info) << " SEventConfig::OutPath path_1 " << path_1 ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    /*
    test_Desc(); 
    */
    test_OutPath(); 

    return 0 ; 
}
