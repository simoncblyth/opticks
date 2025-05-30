/**
SRecordInfo_test.cc
====================

~/o/sysrap/tests/SRecordInfo_test.sh


**/

#include "ssys.h"
#include "SRecordInfo.h"

struct SRecordInfo_test
{
    static int Load();
    static int LoadNonExisting();
    static int Main();
};


inline int SRecordInfo_test::Load()
{
    std::cout << "[SRecordInfo_test::Load" << std::endl ;
    SRecordInfo* sr= SRecordInfo::Load("$SRECORD_PATH") ;
    sr->desc() ;
    std::cout << "]SRecordInfo_test::Load" << std::endl ;
    return 0 ;
}

inline int SRecordInfo_test::LoadNonExisting()
{
    std::cout << "[SRecordInfo_test::LoadNonExisting" << std::endl ;
    SRecordInfo* sr= SRecordInfo::Load("$SRECORD_PATH_NON_EXISTING") ;
    assert( sr == nullptr ); 
    std::cout << "]SRecordInfo_test::LoadNonExisting" << std::endl ;
    return 0 ;
}


inline int SRecordInfo_test::Main()
{
    int rc(0) ;
    const char* TEST = ssys::getenvvar("TEST", "LoadNonExisting");

    if ( strcmp(TEST,"Load") == 0 )              rc += Load() ;
    if ( strcmp(TEST,"LoadNonExisting") == 0 )   rc += LoadNonExisting() ;

    return rc ;
}

int main()
{
    return SRecordInfo_test::Main();
}

