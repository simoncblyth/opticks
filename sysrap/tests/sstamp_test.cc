/**
~/o/sysrap/tests/sstamp_test.sh
**/

#include <iostream>
#include <iomanip>
#include <vector>

#include "ssys.h"
#include "sstamp.h"


struct sstamp_test
{
    static int Format();
    static int FormatLog();
    static int FormatTimeStem();
    static int Main();
};


inline int sstamp_test::Format()
{
    int64_t t = 0 ;

    // 2025-08-20 14:11:59.684

    std::vector<std::string> fmts = {
           "%FT%T.",
           "%FT%T",
           "%F",
           "%FT",
           "%T",
           "%FT%T",
           "%Y",
           "%M",
           "%m",
           "%d",
           "%D",
           "%H",
           "%S",
           "%Y%m%d",
           "%Y%m%d_%H%M%S",
           "%Y-%m-%d %H:%M:%S"
          } ;

    int wid = t == 0 ? 1 : 16 ;

    for(unsigned i=0 ; i < fmts.size() ; i++)
    {
        const char* fmt = fmts[i].c_str();
        for(unsigned j=0 ; j < 3 ; j++)
        {
            int wsubsec = j*3 ; // 0,3,6
            std::string tf = sstamp::Format(t,fmt,wsubsec);
            std::cout
                << " sstamp::Format(" << std::setw(wid) << t << ",\"" << std::setw(15) << fmt << "\"," << wsubsec << ")"
                << " : "
                <<  tf
                << "\n"
                ;
        }
        std::cout << "\n" ;

    }
    return 0 ;
}

int sstamp_test::FormatLog()
{
    for(int i=0 ; i < 100 ; i++) std::cout << sstamp::FormatLog() << "\n" ;
    return 0 ;
}

int sstamp_test::FormatTimeStem()
{
    std::cout
        << std::setw(40)
        << " sstamp::FormatTimeStem() "
        << " : "
        << sstamp::FormatTimeStem()
        << "\n"
        ;

    std::cout
        << std::setw(40)
        << " sstamp::FormatTimeStem(nullptr) "
        << " : "
        << sstamp::FormatTimeStem(nullptr)
        << "\n"
        ;

    std::cout
        << std::setw(40)
        << " sstamp::FormatTimeStem(\"%Y%m%d\") "
        << " : "
        << sstamp::FormatTimeStem("%Y%m%d")
        << "\n"
        ;


    std::cout
        << std::setw(40)
        << " sstamp::FormatTimeStem(\"before_%Y%m%d_after\") "
        << " : "
        << sstamp::FormatTimeStem("before_%Y%m%d_after")
        << "\n"
        ;


    std::cout
        << std::setw(40)
        << " sstamp::FormatTimeStem(\"before_without_percent_after\") "
        << " : "
        << sstamp::FormatTimeStem("before_without_percent_after")
        << "\n"
        ;

    return 0 ;
}


int sstamp_test::Main()
{
    int rc = 0 ;
    const char* TEST = ssys::getenvvar("TEST","ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    if(ALL||0==strcmp(TEST,"Format")) rc += Format();
    if(ALL||0==strcmp(TEST,"FormatLog")) rc += FormatLog();
    if(ALL||0==strcmp(TEST,"FormatTimeStem")) rc += FormatTimeStem();

    return rc ;
}


int main()
{
    return sstamp_test::Main();
}
