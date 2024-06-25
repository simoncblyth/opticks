/**
~/o/sysrap/tests/sstamp_test.sh 
**/

#include <iostream>
#include <iomanip>
#include <vector>

#include "sstamp.h"


void test_Format()
{
    int64_t t = 0 ; 

    std::vector<std::string> fmts = {
           "%FT%T.",
           "%F",
           "%FT",
           "%T", 
           "%T.", 
           "%FT%T.",
           "%Y",
           "%M",
           "%m",
           "%d",
           "%D",
           "%H",
           "%S",
           "%Y%m%d",
           "%Y%m%d_%H%M%S",
          } ; 

    for(unsigned i=0 ; i < fmts.size() ; i++)
    {
        const char* fmt = fmts[i].c_str(); 
        for(unsigned j=0 ; j < 2 ; j++)
        { 

            bool sub = j == 0 ; 
            std::string tf = sstamp::Format(t,fmt,sub); 
            std::cout 
                << std::setw(15) << fmt 
                << " sub " << ( sub ? "Y" : "N" ) 
                << " : "
                <<  tf 
                << "\n"
                ;
        } 
        std::cout << "\n" ; 
 
    }
}


void test_FormatTimeStem()
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


}



int main()
{
    test_FormatTimeStem(); 

    return 0 ; 
}
