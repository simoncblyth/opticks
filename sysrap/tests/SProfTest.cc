// ~/o/sysrap/tests/SProfTest.sh

#include <iostream>
#include "SProf.hh"

struct SProfTest
{ 
    static int Add_Write_Read();
    static int Read();
    static int SetTag();
    static int Main();
};

inline int SProfTest::Add_Write_Read()
{
    std::cout << __FUNCTION__ << std::endl ;

    SProf::Add("start");
    SProf::Add("red");

    for(int i=0 ; i < 10 ; i++ )
    {
        SProf::SetTag(i, "A%0.3d_" ) ;
        SProf::Add("red");
        SProf::Add("green");
        SProf::Add("blue");
        SProf::Add("cyan","photons=10");
        SProf::Write();      // frequent write to have something in case of crash
    }
    SProf::UnsetTag();
    SProf::Add("stop");

    std::cout << SProf::Desc() ;

    std::cout << "--------------------------------" << std::endl ;

    SProf::Write();
    SProf::Read();

    std::cout << SProf::Desc() ;
    return 0;
}


/**
SProfTest::Read
-----------------

Note that running "Read" in the same process before "Add_Write_Read"
without clearing makes it look like the writes are appending
as the vectors get prepopulated, added to then written.

**/


inline int SProfTest::Read()
{
    SProf::Read();
    std::cout << SProf::Desc() ;
    SProf::Clear();
    return 0;
}

inline int SProfTest::SetTag()
{
    for(int i=0 ; i < 100 ; i++)
    {
        SProf::SetTag(i, "A%0.3d_" );
        if(i % 10 == 0 ) SProf::UnsetTag();
        std::cout << "[" << SProf::TAG << "]" << std::endl ;
    }
    SProf::UnsetTag();
    return 0;
}



inline int SProfTest::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;
    if(ALL||0==strcmp(TEST, "Read"))   rc += Read();
    if(ALL||0==strcmp(TEST, "SetTag")) rc += SetTag();
    if(ALL||0==strcmp(TEST, "Add_Write_Read")) rc += Add_Write_Read() ;

    return rc ;
}

int main()
{
    return SProfTest::Main();
}
