#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <iomanip>
#include <sstream>

struct strid
{
    union uif64_t {
        uint64_t  u ; 
        int64_t   i ; 
        double    f ; 
    }; 

    static constexpr const uint64_t U=~0ull ; 
    static void Encode(       glm::tmat4x4<double>& tr, uint64_t  e03=U, uint64_t  e13=U, uint64_t  e23=U, uint64_t  e33=U ); 

    static void Decode( const glm::tmat4x4<double>& tr, uint64_t& e03,   uint64_t& e13  , uint64_t& e23  , uint64_t& e33   ); 
    static void Decode( const glm::tmat4x4<double>& tr, uint64_t& e03,   uint64_t& e13  , uint64_t& e23  ); 
    static void Decode( const glm::tmat4x4<double>& tr, uint64_t& e03,   uint64_t& e13  ); 
    static void Decode( const glm::tmat4x4<double>& tr, uint64_t& e03 ); 

    static void Encode(       double* ptr , uint64_t  e ); 
    static void Decode( const double* ptr , uint64_t& e, double skip ); 
    static void Clear(         glm::tmat4x4<double>& tr); 
    static bool IsClear( const glm::tmat4x4<double>& tr);
    static std::string Desc(  const glm::tmat4x4<double>& tr);
}; 

inline void strid::Encode(      glm::tmat4x4<double>& tr, uint64_t e03, uint64_t e13, uint64_t e23, uint64_t e33)
{
    double* tr00 = glm::value_ptr(tr) ; 
    Encode( tr00+4*0+3, e03 ); 
    Encode( tr00+4*1+3, e13 ); 
    Encode( tr00+4*2+3, e23 ); 
    Encode( tr00+4*3+3, e33 ); 
} 

inline void strid::Decode( const glm::tmat4x4<double>& tr, uint64_t& e03, uint64_t& e13, uint64_t& e23, uint64_t& e33)
{
    const double* tr00 = glm::value_ptr(tr) ; 
    Decode( tr00+4*0+3, e03, 0.) ;  
    Decode( tr00+4*1+3, e13, 0.) ;   
    Decode( tr00+4*2+3, e23, 0.) ;   
    Decode( tr00+4*3+3, e33, 1.) ;   
}
inline void strid::Decode( const glm::tmat4x4<double>& tr, uint64_t& e03, uint64_t& e13, uint64_t& e23 )
{
    const double* tr00 = glm::value_ptr(tr) ; 
    Decode( tr00+4*0+3, e03, 0.) ;  
    Decode( tr00+4*1+3, e13, 0.) ;   
    Decode( tr00+4*2+3, e23, 0.) ;   
}
inline void strid::Decode( const glm::tmat4x4<double>& tr, uint64_t& e03, uint64_t& e13 )
{
    const double* tr00 = glm::value_ptr(tr) ; 
    Decode( tr00+4*0+3, e03, 0.) ;  
    Decode( tr00+4*1+3, e13, 0.) ;   
}
inline void strid::Decode( const glm::tmat4x4<double>& tr, uint64_t& e03 )
{
    const double* tr00 = glm::value_ptr(tr) ; 
    Decode( tr00+4*0+3, e03, 0.) ;  
}


 
inline void strid::Encode( double* ptr, uint64_t e)
{
    if( e == U ) return ; 
    uif64_t uif ; 
    uif.u = e ; 
    *ptr = uif.f ; 
}
inline void strid::Decode( const double* ptr, uint64_t& e, double skip )
{
    uif64_t uif ; 
    uif.f = *ptr ; 
    e = uif.f == skip ? U : uif.u ;  
}
inline void strid::Clear( glm::tmat4x4<double>& tr )
{
    double* tr00 = glm::value_ptr(tr) ; 
    *(tr00+4*0+3) = 0. ;  
    *(tr00+4*1+3) = 0. ;  
    *(tr00+4*2+3) = 0. ;  
    *(tr00+4*3+3) = 1. ;  
} 
inline bool strid::IsClear( const glm::tmat4x4<double>& tr )
{
    const double* tr00 = glm::value_ptr(tr) ; 
    return 
         *(tr00+4*0+3) == 0. && 
         *(tr00+4*1+3) == 0. && 
         *(tr00+4*2+3) == 0. && 
         *(tr00+4*3+3) == 1. ; 
}




inline std::string strid::Desc( const glm::tmat4x4<double>& tr )
{
    uint64_t e03, e13, e23, e33 ; 
    Decode(tr, e03, e13, e23, e33 ); 

    bool clear = IsClear(tr); 

    std::string spc(5, ' '); 

    const double* tr00 = glm::value_ptr(tr) ; 
    std::stringstream ss ; 
    for(unsigned r=0 ; r < 4 ; r++) for(unsigned c=0 ; c < 4 ; c++)
    { 
        unsigned i = r*4 + c ; 
        if( c == 0 ) ss << std::endl ;

        if( c < 3 || ( c == 3 && clear) )
        {
            ss << std::fixed << std::setw(10) << std::setprecision(3) << tr00[i] << " " ;  
        }
        else
        {
            switch(r)
            {
                case 0: ss << spc << std::setw(16) << std::hex << e03 << std::dec ; break ; 
                case 1: ss << spc << std::setw(16) << std::hex << e13 << std::dec ; break ; 
                case 2: ss << spc << std::setw(16) << std::hex << e23 << std::dec ; break ; 
                case 3: ss << spc << std::setw(16) << std::hex << e33 << std::dec ; break ; 
            }
        }
        if( i == 15 ) ss << std::endl ; 
    }
    std::string s = ss.str(); 
    return s ; 
}



