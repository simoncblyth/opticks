/**
SRecord_test.cc
====================

~/o/sysrap/tests/SRecord_test.sh


**/

#include "ssys.h"
#include "SRecord.h"
#include "NPFold.h"

struct SRecord_test
{
    static int Load();
    static int LoadSlice();
    static int LoadNonExisting();
    static int getPhotonAtTime();
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

inline int SRecord_test::getPhotonAtTime()
{
    SRecord* ar = SRecord::Load("$AFOLD", "$AFOLD_RECORD_SLICE" ) ;
    std::cout << ar->desc();

    float t = ssys::getenvfloat("AFOLD_RECORD_TIME", 1.0 ); // ns

    NP* ph = ar->getPhotonAtTime(t);

    std::cout
        << "SRecord_test::getPhotonAtTime"
        << " t " << std::setw(7) << std::fixed << std::setprecision(3) << t
        << " ph " << ( ph ? ph->sstr() : "-" )
        << "\n"
        ;

    NPFold* sub = new NPFold ;
    sub->add("record", ar->record );
    sub->add("ph", ph );

    NPFold* top = new NPFold ;
    top->add_subfold( "getPhotonAtTime", sub );
    top->save("$FOLD");

    return 0 ;
}

inline int SRecord_test::Main()
{
    int rc(0) ;
    const char* test = "getPhotonAtTime" ;
    const char* TEST = ssys::getenvvar("TEST", test );
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    if(ALL||strcmp(TEST,"Load") == 0 )              rc += Load() ;
    if(ALL||strcmp(TEST,"LoadSlice") == 0 )         rc += LoadSlice() ;
    if(ALL||strcmp(TEST,"LoadNonExisting") == 0 )   rc += LoadNonExisting() ;
    if(ALL||strcmp(TEST,"getPhotonAtTime") == 0 )   rc += getPhotonAtTime() ;

    return rc ;
}

int main()
{
    return SRecord_test::Main();
}

