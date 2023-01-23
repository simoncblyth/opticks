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
    static const char* Name( const std::string& name );
    template<typename T> static const char* Name( const T* const obj );
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


