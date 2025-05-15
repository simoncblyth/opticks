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
    static int Main();
};


inline int SRecordInfo_test::Load()
{
    std::cout << "[SRecordInfo_test::Load" << std::endl ;
    SRecordInfo* sr= SRecordInfo::Load("$SRECORD_PATH") ;
    sr->init_minmax2D();
    sr->desc() ;
    std::cout << "]SRecordInfo_test::Load" << std::endl ;
    return 0 ;
}

inline int SRecordInfo_test::Main()
{
    int rc(0) ;
    const char* TEST = ssys::getenvvar("TEST", "Load");

    if ( strcmp(TEST,"Load") == 0 )           rc += Load() ;

    return rc ;
}

int main()
{
    return SRecordInfo_test::Main();
}

