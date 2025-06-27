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
    static int LoadSlice();
    static int LoadNonExisting();
    static int Main();
};


inline int SRecord_test::Load()
{
    std::cout << "[SRecord_test::Load" << std::endl ;
    SRecord* sr= SRecord::Load("$SRECORD_FOLD") ;
    std::cout << sr->desc() ;
    std::cout << "]SRecord_test::Load" << std::endl ;
    return 0 ;
}

inline int SRecord_test::LoadSlice()
{
    std::cout << "[SRecord_test::LoadSlice" << std::endl ;
    SRecord* ar = SRecord::Load("$AFOLD", "$AFOLD_RECORD_SLICE" ) ;
    std::cout << ar->desc();
    std::cout << "]SRecord_test::LoadSlice" << std::endl ;
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
    const char* TEST = ssys::getenvvar("TEST", "LoadSlice");
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    if (ALL||strcmp(TEST,"Load") == 0 )              rc += Load() ;
    if (ALL||strcmp(TEST,"LoadSlice") == 0 )         rc += LoadSlice() ;
    if (ALL||strcmp(TEST,"LoadNonExisting") == 0 )   rc += LoadNonExisting() ;

    return rc ;
}

int main()
{
    return SRecord_test::Main();
}

