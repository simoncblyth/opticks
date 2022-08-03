#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
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

    union uif32_t {
        uint32_t  u ; 
        int32_t   i ; 
        float     f ; 
    }; 

    static void Encode(       glm::tmat4x4<double>& tr, const glm::tvec4<uint64_t>& col3 ); 
    static void Encode(       double* ptr , uint64_t  e ); 
    static void Decode( const glm::tmat4x4<double>& tr,       glm::tvec4<uint64_t>& col3 ); 
    static void Decode( const double* ptr , uint64_t& e ); 

    static void Encode(       glm::tmat4x4<float>& tr, const glm::tvec4<uint32_t>& col3 ); 
    static void Encode(       float* ptr , uint32_t  e ); 
    static void Decode( const glm::tmat4x4<float>& tr,       glm::tvec4<uint32_t>& col3 ); 
    static void Decode( const float* ptr , uint32_t& e ); 


    template<typename T>
    static void Clear(              glm::tmat4x4<T>& tr); 

    template<typename T>
    static bool IsClear(      const glm::tmat4x4<T>& tr);


    template<typename T, typename S>
    static std::string Desc(  const glm::tmat4x4<T>& tr);

    static void Narrow( glm::tmat4x4<float>& dst,  const glm::tmat4x4<double>& src ); 
    static void Narrow( std::vector<glm::tmat4x4<float>>& dst,  const std::vector<glm::tmat4x4<double>>& src ); 
}; 


inline void strid::Encode(      glm::tmat4x4<double>& tr, const glm::tvec4<uint64_t>& col3 )
{
    double* tr00 = glm::value_ptr(tr) ; 
    for(int r=0 ; r < 4 ; r++) Encode(  tr00+4*r+3, col3[r] ) ; 
} 
inline void strid::Encode( double* ptr, uint64_t e)
{
    if(e == 0) return ; // kludge to keep [:,3,3] 1. for simpler comparison with GGeo cf.inst  
    uif64_t uif ; 
    uif.u = e ; 
    *ptr = uif.f ; 
}
inline void strid::Decode( const glm::tmat4x4<double>& tr,      glm::tvec4<uint64_t>& col3  )
{
    const double* tr00 = glm::value_ptr(tr) ; 
    for(int r=0 ; r < 4 ; r++) Decode( tr00+4*r+3, col3[r] ) ;  
}
inline void strid::Decode( const double* ptr, uint64_t& e )
{
    uif64_t uif ; 
    uif.f = *ptr ; 
    e = uif.u ;  
}





inline void strid::Encode(      glm::tmat4x4<float>& tr, const glm::tvec4<uint32_t>& col3 )
{
    float* tr00 = glm::value_ptr(tr) ; 
    for(int r=0 ; r < 4 ; r++) Encode(  tr00+4*r+3, col3[r] ) ; 
} 
inline void strid::Encode( float* ptr, uint32_t e)
{
    if(e == 0) return ; // kludge to keep [:,3,3] 1. for simpler comparison with GGeo cf.inst  
    uif32_t uif ; 
    uif.u = e ; 
    *ptr = uif.f ; 
}
inline void strid::Decode( const glm::tmat4x4<float>& tr,      glm::tvec4<uint32_t>& col3  )
{
    const float* tr00 = glm::value_ptr(tr) ; 
    for(int r=0 ; r < 4 ; r++) Decode( tr00+4*r+3, col3[r] ) ;  
}
inline void strid::Decode( const float* ptr, uint32_t& e )
{
    uif32_t uif ; 
    uif.f = *ptr ; 
    e = uif.u ;  
}





template<typename T>
inline void strid::Clear( glm::tmat4x4<T>& tr )
{
    T* tr00 = glm::value_ptr(tr) ; 
    *(tr00+4*0+3) = T(0.) ;  
    *(tr00+4*1+3) = T(0.) ;  
    *(tr00+4*2+3) = T(0.) ;  
    *(tr00+4*3+3) = T(1.) ;  
} 


template<typename T>
inline bool strid::IsClear( const glm::tmat4x4<T>& tr )
{
    const T* tr00 = glm::value_ptr(tr) ; 
    return 
         *(tr00+4*0+3) == T(0.) && 
         *(tr00+4*1+3) == T(0.) && 
         *(tr00+4*2+3) == T(0.) && 
         *(tr00+4*3+3) == T(1.) ; 
}




template<typename T, typename S>
inline std::string strid::Desc( const glm::tmat4x4<T>& tr )
{
    glm::tvec4<S> col3 ; 
    Decode(tr, col3); 

    bool clear = IsClear(tr); 

    std::string spc(5, ' '); 

    const T* tr00 = glm::value_ptr(tr) ; 
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
            ss << spc << std::setw(16) << std::hex << col3[r] << std::dec ; break ; 
        }
        if( i == 15 ) ss << std::endl ; 
    }
    std::string s = ss.str(); 
    return s ; 
}

template std::string strid::Desc<double, uint64_t>(const glm::tmat4x4<double>& tr ); 
template std::string strid::Desc<float,  uint32_t>(const glm::tmat4x4<float>& tr ); 





inline void strid::Narrow( glm::tmat4x4<float>& dst_,  const glm::tmat4x4<double>& src_ )
{
    glm::tvec4<uint64_t> src_col3 ; 
    Decode(src_, src_col3); 



    float* dst = glm::value_ptr(dst_); 
    const double* src = glm::value_ptr(src_); 

    for(unsigned r=0 ; r < 4 ; r++) 
    for(unsigned c=0 ; c < 4 ; c++)
    {
        unsigned i=r*4 + c ; 

        /*
        std::cout 
            << " r " << std::setw(2) << r  
            << " c " << std::setw(2) << c
            << " i " << std::setw(2) << i 
            << std::endl 
            ;
        */

        dst[i] = float(src[i]); 
    }


    glm::tvec4<uint32_t> dst_col3 ; 
    for(int r=0 ; r < 4 ; r++) dst_col3[r] = ( src_col3[r] & 0xffffffff ) ; 

    Encode(dst_, dst_col3); 

}

inline void strid::Narrow( std::vector<glm::tmat4x4<float>>& dst_,  const std::vector<glm::tmat4x4<double>>& src_ )
{
    dst_.resize(src_.size()); 
    for(unsigned i=0 ; i < src_.size() ; i++)
    {  
        const glm::tmat4x4<double>& src = src_[i] ; 
        glm::tmat4x4<float>& dst = dst_[i] ; 
        Narrow(dst, src); 
    }
}


