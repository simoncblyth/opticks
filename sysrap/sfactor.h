#pragma once

#include <cassert>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>


struct sfactor
{
    static constexpr const int NV = 10 ; // sub is equivalent to 8 integer fields 
    unsigned index ; 
    int      freq ; 
    char     sub[32] ;  // caution : no null termination 

    void set_sub(const char* s); 
    std::string get_sub() const ; 

    std::string desc() const ; 
}; 

inline void sfactor::set_sub(const char* s)
{
    assert( strlen(s) == 32 ); 
    memcpy( &sub, s, 32 ); 
}

inline std::string sfactor::get_sub() const 
{
    std::string sub_(sub, 32);  // needed as sub array is not null terminated 
    return sub_ ; 
}


inline std::string sfactor::desc() const 
{
    std::stringstream ss ; 
    ss << "sfactor"
       << " index " << std::setw(3) << index
       << " freq " << std::setw(6) << freq
       << " sub [" << std::setw(32) << get_sub()  << "]" 
       ;   
    std::string s = ss.str(); 
    return s ; 
}


