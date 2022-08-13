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

    template<typename T>
    static std::string Desc_(  const glm::tmat4x4<T>& tr);

    template<typename T>
    static std::string Desc_(  const char* a_label, const char* b_label, const glm::tmat4x4<T>& a, const glm::tmat4x4<T>& b );

    template<typename T>
    static std::string Desc_(  const char* a_label, const char* b_label, const char* c_label, 
                               const glm::tmat4x4<T>& a, const glm::tmat4x4<T>& b, const glm::tmat4x4<T>& c );

    static void Narrow( glm::tmat4x4<float>& dst,  const glm::tmat4x4<double>& src ); 
    static void Narrow( std::vector<glm::tmat4x4<float>>& dst,  const std::vector<glm::tmat4x4<double>>& src ); 

    template<typename T>
    static T DiffFromIdentity( const glm::tmat4x4<T>& tr ); 

}; 


inline void strid::Encode(      glm::tmat4x4<double>& tr, const glm::tvec4<uint64_t>& col3 )
{
    double* tr00 = glm::value_ptr(tr) ; 
    for(int r=0 ; r < 4 ; r++) Encode(  tr00+4*r+3, col3[r] ) ; 
} 

/**
strid::Encode
---------------

Formerly kludge skipped e=0 for some reason. 

Note that without encoding the double 0.,0.,0.,1. 
of the fourth column 

**/

inline void strid::Encode( double* ptr, uint64_t e)
{
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


template<typename T>
inline std::string strid::Desc_( const glm::tmat4x4<T>& tr )
{
    const T* tr00 = glm::value_ptr(tr) ; 
    std::stringstream ss ; 
    for(unsigned r=0 ; r < 4 ; r++) for(unsigned c=0 ; c < 4 ; c++)
    { 
        unsigned i = r*4 + c ; 
        if( c == 0 ) ss << std::endl ;
        ss << std::fixed << std::setw(10) << std::setprecision(3) << tr00[i] << " " ;  
        if( i == 15 ) ss << std::endl ; 
    }
    std::string s = ss.str(); 
    return s ; 
}


template<typename T>
inline std::string strid::Desc_(  const char* a_label, const char* b_label, const glm::tmat4x4<T>& a, const glm::tmat4x4<T>& b )
{
    const T* aa = glm::value_ptr(a) ; 
    const T* bb = glm::value_ptr(b) ; 
    unsigned num = 2 ; 

    std::stringstream ss ; 

    ss << std::setw(10) << a_label << std::setw(40) << " " << std::setw(10) << b_label << std::endl ; 

    for(unsigned r=0 ; r < 4 ; r++) 
    for(unsigned t=0 ; t < num ; t++) 
    for(unsigned c=0 ; c < 4 ; c++)
    { 
        const T* vv = nullptr ; 
        switch(t)
        {
           case 0: vv = aa ; break ; 
           case 1: vv = bb ; break ; 
        }
        unsigned i = r*4 + c ; 
        if( c == 0 && t == 0) ss << std::endl ;
        ss << std::fixed << std::setw(10) << std::setprecision(3) << vv[i] << " " ;  
        if( c == 3 && t == 0) ss << std::setw(10) << " " ; 
        if( i == 15 && t == num - 1) ss << std::endl ; 
    }
    std::string s = ss.str(); 
    return s ; 
}



template<typename T>
inline std::string strid::Desc_(
    const char* a_label, 
    const char* b_label, 
    const char* c_label, 
    const glm::tmat4x4<T>& a, 
    const glm::tmat4x4<T>& b,
    const glm::tmat4x4<T>& c
   )
{
    const T* aa = glm::value_ptr(a) ; 
    const T* bb = glm::value_ptr(b) ; 
    const T* cc = glm::value_ptr(c) ; 
    unsigned num = 3 ; 
    std::stringstream ss ; 
    ss 
        << std::setw(10) << a_label << std::setw(40) << " " 
        << std::setw(10) << b_label << std::setw(40) << " " 
        << std::setw(10) << c_label << std::setw(40) << " " 
        << std::endl
        ; 

    for(unsigned r=0 ; r < 4 ; r++) 
    for(unsigned t=0 ; t < num ; t++) 
    for(unsigned c=0 ; c < 4 ; c++)
    { 
        unsigned i = r*4 + c ; 
        const T* vv = nullptr ; 
        switch(t)
        {
           case 0: vv = aa ; break ; 
           case 1: vv = bb ; break ; 
           case 2: vv = cc ; break ; 
        }
        if( c == 0 && t == 0) ss << std::endl ;
        ss << std::fixed << std::setw(10) << std::setprecision(3) << vv[i] << " " ;  
        if( c == 3 && t < num - 1 )  ss << std::setw(10) << " " ; 
        if( i == 15 && t == num - 1) ss << std::endl ; 
    }
    std::string s = ss.str(); 
    return s ; 
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

template<typename T>
inline T strid::DiffFromIdentity(const glm::tmat4x4<T>& tr)
{
    const T* tt = glm::value_ptr(tr); 
    T max_delta = T(0.) ; 
    for(int r=0 ; r < 4 ; r++) for(int c=0 ; c < 4 ; c++)
    {
        int i = r*4 + c ; 
        T id = r == c ? T(1.) : T(0.) ; 
        T delta = std::abs( tt[i] - id ) ; 
        if(delta > max_delta) max_delta = delta ; 
    }
    return max_delta ; 
}

