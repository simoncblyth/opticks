/**
SRecord_test.cc
====================

~/o/sysrap/tests/SRecord_test.sh


**/

#include "ssys.h"
#include "SRecord.h"

struct SRecord_test
{
    static int Load();
    static int LoadNonExisting();
    static int Main();
};


inline int SRecord_test::Load()
{
    std::cout << "[SRecord_test::Load" << std::endl ;
    SRecord* sr= SRecord::Load("$SRECORD_FOLD") ;
    sr->desc() ;
    std::cout << "]SRecord_test::Load" << std::endl ;
    return 0 ;
}

inline int SRecord_test::LoadNonExisting()
{
    std::cout << "[SRecord_test::LoadNonExisting" << std::endl ;
    SRecord* sr= SRecord::Load("$SRECORD_FOLD_NON_EXISTING") ;
    assert( sr == nullptr );
    std::cout << "]SRecord_test::LoadNonExisting" << std::endl ;
    return 0 ;
}


inline int SRecord_test::Main()
{
    int rc(0) ;
    const char* TEST = ssys::getenvvar("TEST", "LoadNonExisting");

    if ( strcmp(TEST,"Load") == 0 )              rc += Load() ;
    if ( strcmp(TEST,"LoadNonExisting") == 0 )   rc += LoadNonExisting() ;

    return rc ;
}

int main()
{
    return SRecord_test::Main();
}

