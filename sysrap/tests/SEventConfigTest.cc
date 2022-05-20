#include <iostream>
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"

void test_Desc()
{
    LOG(info); 
    SEventConfig::SetMaxPhoton(101); 
    std::cout << SEventConfig::Desc() << std::endl ; 
}
void test_OutPath()
{
    LOG(info); 
    const char* path_0 = SEventConfig::OutPath("stem", 101, ".npy" ); 
    const char* path_1 = SEventConfig::OutPath("local_reldir", "stem", 101, ".npy" ); 
    LOG(info) << " SEventConfig::OutPath path_0 " << path_0 ; 
    LOG(info) << " SEventConfig::OutPath path_1 " << path_1 ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_OutPath(); 
    test_Desc(); 

    return 0 ; 
}
