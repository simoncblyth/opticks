#pragma once

struct sdebug
{
    int addGenstep ; 
    int beginPhoton ; 
    int rjoinPhoton ; 
    int pointPhoton ; 
    int finalPhoton ; 
    int d12match_fail ; 

    void zero(); 
    std::string desc() const ; 
};

inline void sdebug::zero()
{
    addGenstep = 0 ; 
    beginPhoton = 0 ; 
    rjoinPhoton = 0 ; 
    pointPhoton = 0 ; 
    finalPhoton = 0 ; 
    d12match_fail = 0 ; 
}

#include <string>
#include <sstream>
#include <iomanip>

inline std::string sdebug::desc() const 
{
    std::stringstream ss ; 
    ss << "sdebug::desc" << std::endl 
       << std::setw(20) << " addGenstep "     << " : " << std::setw(10) <<  addGenstep << std::endl 
       << std::setw(20) << " beginPhoton "    << " : " << std::setw(10) <<  beginPhoton << std::endl 
       << std::setw(20) << " rjoinPhoton "    << " : " << std::setw(10) <<  rjoinPhoton << std::endl 
       << std::setw(20) << " pointPhoton "    << " : " << std::setw(10) <<  pointPhoton << std::endl 
       << std::setw(20) << " finalPhoton "    << " : " << std::setw(10) <<  finalPhoton << std::endl 
       << std::setw(20) << " d12match_fail "  << " : " << std::setw(10) <<  d12match_fail << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}




