#include <cstdlib>
#include <iostream>

#include "ssys.h"
#include "salloc.h"
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SComp.h"


struct SEventConfigTest
{
    static int Desc(); 
    static int OutPath(); 
    static int GatherCompList();
    static int SaveCompList();
    static int Initialize_Comp_();
    static int SetDefault();
    static int Save();
    static int InputGenstep(); 

    static int main(); 

}; 

int SEventConfigTest::Desc()
{
    LOG(info); 
    std::cout << SEventConfig::Desc() << std::endl ; 
    return 0 ; 
}

int SEventConfigTest::OutPath()
{
    LOG(info); 

    bool unique = false ; 
    const char* path_0 = SEventConfig::OutPath("stem", 101, ".npy", unique ); 
    const char* path_1 = SEventConfig::OutPath("local_reldir", "stem", 101, ".npy", unique ); 
    const char* path_2 = SEventConfig::OutPath("snap", -1, ".jpg", unique );

    LOG(info) << " SEventConfig::OutPath path_0 " << path_0 ; 
    LOG(info) << " SEventConfig::OutPath path_1 " << path_1 ; 
    LOG(info) << " SEventConfig::OutPath path_2 " << path_2 ; 
    return 0 ; 
}


int SEventConfigTest::GatherCompList()
{
    std::vector<unsigned> gather_comps ; 
    SEventConfig::GatherCompList(gather_comps) ; 
    std::cout << SComp::Desc(gather_comps) << std::endl ; 
    return 0 ; 
}

int SEventConfigTest::SaveCompList()
{
    std::vector<unsigned> save_comps ; 
    SEventConfig::SaveCompList(save_comps) ; 
    std::cout << SComp::Desc(save_comps) << std::endl ; 
    return 0 ; 
}

int SEventConfigTest::Initialize_Comp_()
{
    unsigned gather_mask = 0 ; 
    unsigned save_mask = 0 ; 

    SEventConfig::Initialize_Comp_(gather_mask,save_mask ); 

    LOG(info) << " SComp::Desc(gather_mask) " << SComp::Desc(gather_mask) ; 
    LOG(info) << " SComp::Desc(save_mask) " << SComp::Desc(save_mask) ; 
    return 0 ; 
}




int SEventConfigTest::SetDefault()
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
    return 0 ; 
}

int SEventConfigTest::Save()
{
    SEventConfig::Save("$FOLD"); 
    return 0 ; 
}

int SEventConfigTest::InputGenstep()
{
    for(int idx=0 ; idx < 100 ; idx++)
    {
        const char* path = SEventConfig::InputGenstep(idx) ; 
        std::cout << " idx " << std::setw(4) << idx <<  " path " << ( path ? path : "-" ) << std::endl ; 
    }

    return 0 ; 
}


int SEventConfigTest::main()
{
    SEventConfig::Initialize(); 

    const char* TEST = ssys::getenvvar("TEST", "Desc"); 
    bool ALL = strcmp(TEST, "ALL") == 0 ; 

    int rc = 0 ; 
    if(ALL||strcmp(TEST,"OutPath") == 0)          rc += OutPath();  
    if(ALL||strcmp(TEST,"GatherCompList") == 0)   rc += GatherCompList();  
    if(ALL||strcmp(TEST,"SaveCompList") == 0)     rc += SaveCompList();  
    if(ALL||strcmp(TEST,"Initialize_Comp_") == 0) rc += Initialize_Comp_();  
    if(ALL||strcmp(TEST,"SetDefault") == 0)       rc += SetDefault();  
    if(ALL||strcmp(TEST,"Save") == 0)             rc += Save();  
    if(ALL||strcmp(TEST,"Desc") == 0)             rc += Desc();  
    if(ALL||strcmp(TEST,"InputGenstep") == 0)     rc += InputGenstep();  

    return rc ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    return SEventConfigTest::main() ; 
}
