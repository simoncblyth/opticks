#pragma once

#include <string>
#include <sstream>

struct U4Fake
{
    enum { 
        FAKE_STEP_MM = 0x1 << 0, 
        FAKE_FDIST   = 0x1 << 1, 
        FAKE_SURFACE = 0x1 << 2,  
        FAKE_MANUAL  = 0x1 << 3  
        }; 

    static constexpr const char* FAKE_STEP_MM_ = "FAKE_STEP_MM" ;
    static constexpr const char* FAKE_FDIST_ = "FAKE_FDIST" ;
    static constexpr const char* FAKE_SURFACE_ = "FAKE_SURFACE" ;
    static constexpr const char* FAKE_MANUAL_ = "FAKE_MANUAL" ;

    static std::string Desc(unsigned fakemask); 
};

inline std::string U4Fake::Desc(unsigned fakemask)
{
    std::stringstream ss ; 
    if(fakemask & FAKE_STEP_MM) ss << FAKE_STEP_MM_ << "|" ; 
    if(fakemask & FAKE_FDIST)   ss << FAKE_FDIST_ << "|" ; 
    if(fakemask & FAKE_SURFACE)  ss << FAKE_SURFACE_ << "|" ; 
    if(fakemask & FAKE_MANUAL)  ss << FAKE_MANUAL_ << "|" ; 
    std::string str = ss.str(); 
    return str ; 
}

