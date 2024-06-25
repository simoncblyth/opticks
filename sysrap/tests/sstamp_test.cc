/**
~/o/sysrap/tests/sstamp_test.sh 
**/

#include <iostream>
#include <iomanip>
#include <vector>

#include "sstamp.h"

int main()
{
    int64_t t = 0 ; 

    std::vector<std::string> fmts = {
           "%FT%T.",
           "%F",
           "%FT",
           "%T", 
           "%T.", 
           "%FT%T."
          } ; 

    for(unsigned i=0 ; i < fmts.size() ; i++)
    {
        const char* fmt = fmts[i].c_str(); 
        std::string tf = sstamp::Format(t,fmt); 
        std::cout 
            << std::setw(15) << fmt 
            << " : "
            <<  tf 
            << "\n"
            ; 
    }


    return 0 ; 
}
