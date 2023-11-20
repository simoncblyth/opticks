#include <cstdlib>
#include <iostream>

#include "salloc.h"
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SComp.h"


void test_Desc()
{
    LOG(info); 
    std::cout << SEventConfig::Desc() << std::endl ; 
}



void test_OutPath()
{
    LOG(info); 
    const char* path_0 = SEventConfig::OutPath("stem", 101, ".npy" ); 
    const char* path_1 = SEventConfig::OutPath("local_reldir", "stem", 101, ".npy" ); 
    const char* path_2 = SEventConfig::OutPath("snap", -1, ".jpg" );

    LOG(info) << " SEventConfig::OutPath path_0 " << path_0 ; 
    LOG(info) << " SEventConfig::OutPath path_1 " << path_1 ; 
    LOG(info) << " SEventConfig::OutPath path_2 " << path_2 ; 
}


void test_GatherCompList()
{
    std::vector<unsigned> gather_comps ; 
    SEventConfig::GatherCompList(gather_comps) ; 
    std::cout << SComp::Desc(gather_comps) << std::endl ; 
}

void test_SaveCompList()
{
    std::vector<unsigned> save_comps ; 
    SEventConfig::SaveCompList(save_comps) ; 
    std::cout << SComp::Desc(save_comps) << std::endl ; 
}





void test_CompAuto()
{
    unsigned gather_mask = 0 ; 
    unsigned save_mask = 0 ; 

    SEventConfig::CompAuto(gather_mask,save_mask ); 

    LOG(info) << " SComp::Desc(gather_mask) " << SComp::Desc(gather_mask) ; 
    LOG(info) << " SComp::Desc(save_mask) " << SComp::Desc(save_mask) ; 
}


void test_SetCompAuto()
{
    std::cout 
        << "test_SetCompAuto.0"
        << std::endl 
        << SEventConfig::Desc() 
        << std::endl
        ; 

    /*
    SEventConfig::SetCompAuto() ;     // NOW DONE BY SEventConfig::Initialize

    std::cout 
        << "test_SetCompAuto.1"
        << std::endl 
        << SEventConfig::Desc() 
        << std::endl
        ; 
    */
}

void test_SetDefault()
{
    std::cout 
        << "test_SetDefault.0"
        << std::endl 
        << SEventConfig::Desc() 
        << std::endl
        ; 

    SEventConfig::SetDefault() ;     

    std::cout 
        << "test_SetDefault.1"
        << std::endl 
        << SEventConfig::Desc() 
        << std::endl
        ; 
}

void test_Save()
{
    SEventConfig::Save("$FOLD"); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEventConfig::Initialize(); 

    /*
    test_OutPath(); 
    test_GatherCompList(); 
    test_SaveCompList(); 
    test_CompAuto(); 
    test_SetCompAuto(); 
    test_SetDefault(); 
    test_Save(); 
    test_Desc(); 
    test_SetCompAuto(); 
    */

    test_Desc(); 


    return 0 ; 
}
