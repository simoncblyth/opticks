#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iterator>

struct SArgs
{
    int argc ;
    char** argv ; 

    std::vector<std::string> elem ; 

    SArgs(const std::string& line)
    {
        std::stringstream ss(line);
        typedef std::istream_iterator<std::string> ISI ; 
        ISI begin(ss);
        ISI end ; 
        std::vector<std::string> vs(begin, end);

        elem.resize(vs.size());
        std::copy(vs.begin(), vs.end(), elem.begin());

        argc = 1 + elem.size();
        argv = new char*[argc];

        argv[0] = const_cast<char*>("dummy") ;
        for(int i=1 ; i < argc ; i++) argv[i] = const_cast<char*>(elem[i-1].c_str()) ;
    }

    void dump()
    {
        for(int i=0 ; i < argc ; i++)
        {
            std::cout << std::setw(3) << i 
                      << std::setw(15) << elem[i]
                      << std::setw(15) << argv[i]
                      << std::endl ; 
        }
    }
};


