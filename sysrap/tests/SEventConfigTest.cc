#include <cstdlib>
#include <iostream>

#include "salloc.h"
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SComp.h"


struct SEventConfigTest
{
    static void Desc(); 
    static void OutPath(); 
    static void GatherCompList();
    static void SaveCompList();
    static void CompAuto();
    static void SetCompAuto();
    static void SetDefault();
    static void Save();
    static void InputGenstep(); 

    static int main(); 

}; 

void SEventConfigTest::Desc()
{
    LOG(info); 
    std::cout << SEventConfig::Desc() << std::endl ; 
}

void SEventConfigTest::OutPath()
{
    LOG(info); 

    bool unique = false ; 
    const char* path_0 = SEventConfig::OutPath("stem", 101, ".npy", unique ); 
    const char* path_1 = SEventConfig::OutPath("local_reldir", "stem", 101, ".npy", unique ); 
    const char* path_2 = SEventConfig::OutPath("snap", -1, ".jpg", unique );

    LOG(info) << " SEventConfig::OutPath path_0 " << path_0 ; 
    LOG(info) << " SEventConfig::OutPath path_1 " << path_1 ; 
    LOG(info) << " SEventConfig::OutPath path_2 " << path_2 ; 
}


void SEventConfigTest::GatherCompList()
{
    std::vector<unsigned> gather_comps ; 
    SEventConfig::GatherCompList(gather_comps) ; 
    std::cout << SComp::Desc(gather_comps) << std::endl ; 
}

void SEventConfigTest::SaveCompList()
{
    std::vector<unsigned> save_comps ; 
    SEventConfig::SaveCompList(save_comps) ; 
    std::cout << SComp::Desc(save_comps) << std::endl ; 
}

void SEventConfigTest::CompAuto()
{
    unsigned gather_mask = 0 ; 
    unsigned save_mask = 0 ; 

    SEventConfig::CompAuto(gather_mask,save_mask ); 

    LOG(info) << " SComp::Desc(gather_mask) " << SComp::Desc(gather_mask) ; 
    LOG(info) << " SComp::Desc(save_mask) " << SComp::Desc(save_mask) ; 
}


void SEventConfigTest::SetCompAuto()
{
    std::cout 
        << "SEventConfigTest::SetCompAuto.0"
        << std::endl 
        << SEventConfig::Desc() 
        << std::endl
        ; 

    /*
    SEventConfig::SetCompAuto() ;     // NOW DONE BY SEventConfig::Initialize

    std::cout 
        << "SEventConfigTest::SetCompAuto.1"
        << std::endl 
        << SEventConfig::Desc() 
        << std::endl
        ; 
    */
}

void SEventConfigTest::SetDefault()
{
    std::cout 
        << "SEventConfigTest::SetDefault.0"
        << std::endl 
        << SEventConfig::Desc() 
        << std::endl
        ; 

    SEventConfig::SetDefault() ;     

    std::cout 
        << "SEventConfigTest::SetDefault.1"
        << std::endl 
        << SEventConfig::Desc() 
        << std::endl
        ; 
}

void SEventConfigTest::Save()
{
    SEventConfig::Save("$FOLD"); 
}

void SEventConfigTest::InputGenstep()
{
    for(int idx=0 ; idx < 100 ; idx++)
    {
        const char* path = SEventConfig::InputGenstep(idx) ; 
        std::cout << " idx " << std::setw(4) << idx <<  " path " << ( path ? path : "-" ) << std::endl ; 
    }

}


int SEventConfigTest::main()
{
    SEventConfig::Initialize(); 

    /*
    OutPath(); 
    GatherCompList(); 
    SaveCompList(); 
    CompAuto(); 
    SetCompAuto(); 
    SetDefault(); 
    Save(); 
    SetCompAuto(); 
    InputGenstep(); 
    */
    Desc(); 


    return 0 ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    return SEventConfigTest::main() ; 
}
