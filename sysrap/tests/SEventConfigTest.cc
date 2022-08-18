#include <iostream>
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SComp.h"

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

void test_CompList()
{
    std::vector<unsigned> comps ; 
    SEventConfig::CompList(comps) ; 
    std::cout << SComp::Desc(comps) << std::endl ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    /*
    test_OutPath(); 
    test_Desc(); 
    */
    test_CompList(); 

    return 0 ; 
}
