#include <cstdlib>
#include <iostream>

#include "salloc.h"
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SComp.h"

const char* BASE = getenv("BASE"); 

void test_Desc()
{
    LOG(info); 
    //SEventConfig::SetMaxPhoton(101); 
    std::cout << SEventConfig::Desc() << std::endl ; 
}

void test_EstimateAlloc()
{
    LOG(info); 
    //SEventConfig::SetMaxPhoton(101); 
    std::cout << SEventConfig::Desc() << std::endl ; 

    salloc* al = salloc::Load(BASE) ; 
    LOG(info)  << "al.desc" << std::endl << ( al ? al->desc() : "-" ) ; 
}






/**

epsilon:sysrap blyth$ SEventConfigTest 
2022-08-18 16:46:45.830 INFO  [32281222] [test_OutPath@15] 
2022-08-18 16:46:45.831 INFO  [32281222] [test_OutPath@20]  SEventConfig::OutPath path_0 /tmp/blyth/opticks/SEventConfigTest/stem00101.npy
2022-08-18 16:46:45.831 INFO  [32281222] [test_OutPath@21]  SEventConfig::OutPath path_1 /tmp/blyth/opticks/SEventConfigTest/local_reldir/stem00101.npy
2022-08-18 16:46:45.831 INFO  [32281222] [test_OutPath@22]  SEventConfig::OutPath path_2 /tmp/blyth/opticks/SEventConfigTest/snap.jpg

epsilon:sysrap blyth$ GEOM=hello SEventConfigTest
2022-08-18 16:47:09.672 INFO  [32281483] [test_OutPath@15] 
2022-08-18 16:47:09.674 INFO  [32281483] [test_OutPath@20]  SEventConfig::OutPath path_0 /tmp/blyth/opticks/hello/SEventConfigTest/stem00101.npy
2022-08-18 16:47:09.674 INFO  [32281483] [test_OutPath@21]  SEventConfig::OutPath path_1 /tmp/blyth/opticks/hello/SEventConfigTest/local_reldir/stem00101.npy
2022-08-18 16:47:09.674 INFO  [32281483] [test_OutPath@22]  SEventConfig::OutPath path_2 /tmp/blyth/opticks/hello/SEventConfigTest/snap.jpg

**/


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


void test_CompList()
{
    std::vector<unsigned> comps ; 
    SEventConfig::CompList(comps) ; 
    std::cout << SComp::Desc(comps) << std::endl ; 
}

void test_CompMaskAuto()
{
    LOG(info) << " SEventConfig::CompMaskAuto() " << SEventConfig::CompMaskAuto() ; 
    LOG(info) << " SComp::Desc(SEventConfig::CompMaskAuto()) " << SComp::Desc(SEventConfig::CompMaskAuto()) ; 
}

void test_SetCompMaskAuto()
{
    std::cout 
        << "test_SetCompMaskAuto.0"
        << std::endl 
        << SEventConfig::Desc() 
        << std::endl
        ; 

    SEventConfig::SetCompMaskAuto() ;     

    std::cout 
        << "test_SetCompMaskAuto.1"
        << std::endl 
        << SEventConfig::Desc() 
        << std::endl
        ; 
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

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEventConfig::Initialize(); 

    /*
    test_Desc(); 
    test_OutPath(); 
    test_CompList(); 
    test_CompMaskAuto(); 
    test_SetCompMaskAuto(); 
    test_SetDefault(); 
    */

    test_EstimateAlloc(); 

    return 0 ; 
}
