#pragma once
/**
S4.h
=====

After X4.hh

**/
#include <string>
#include <cstring>

struct S4
{
    static constexpr const char* IMPLICIT_PREFIX = "Implicit_RINDEX_NoRINDEX_" ; 

    static const char* Name( const std::string& name );
    template<typename T> static const char* Name( const T* const obj );

    static std::string Strip( const std::string& name );

    static std::string ImplicitBorderSurfaceName_Debug( 
          const std::string& pv1, 
          const char* mat1, 
          const std::string& pv2,
          const char* mat2, 
          bool flip=false 
       ); 

    static std::string ImplicitBorderSurfaceName( 
          const std::string& pv1, 
          const std::string& pv2, 
          bool flip=false 
       );  
};



inline const char* S4::Name( const std::string& name )
{
    return strdup( name.c_str() );
}

template<typename T>
inline const char* S4::Name( const T* const obj )
{    
    if(obj == nullptr) return nullptr ; 
    const std::string& name = obj->GetName();
    return Name(name);
}


inline std::string S4::Strip( const std::string& name )
{
    std::string sname = name.substr(0, name.find("0x")) ;
    return sname ; 
}

inline std::string S4::ImplicitBorderSurfaceName_Debug( 
          const std::string& pv1, 
          const char* mat1, 
          const std::string& pv2,
          const char* mat2, 
          bool flip 
       )
{
    std::string spv1 = Strip(pv1); 
    std::string spv2 = Strip(pv2); 
    std::stringstream ss ; 
    ss << "Implicit" ; 
 
    if( flip == false )
    {
        ss 
            << "__RINDEX__" << spv1 << "__" << mat1 
            << "__NoRINDEX__" << spv2 << "__" << mat2 
            ; 
    }
    else
    {
        ss 
            << "__RINDEX__" << spv2 << "__" << mat2 
            << "__NoRINDEX__" << spv1 << "__" << mat1 
            ; 
    } 
    std::string str = ss.str();
    return str ; 
}


inline std::string S4::ImplicitBorderSurfaceName( 
          const std::string& pv1, 
          const std::string& pv2,
          bool flip 
       )
{
    std::string spv1 = Strip(pv1); 
    std::string spv2 = Strip(pv2); 
    std::stringstream ss ; 
    ss << IMPLICIT_PREFIX ; 
    if( flip == false )
    {
        ss << spv1 << "_" << spv2 ; 
    }
    else
    {
        ss << spv2 << "_" << spv1 ;
    } 
    std::string str = ss.str();
    return str ; 
}
