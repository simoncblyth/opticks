#pragma once

#include <iostream>

struct Obj ; 

/**
_Obj : persistable type must not have any pointers 
**/
struct _Obj
{
    static void Serialize(    _Obj& p, const Obj* o ); 
    static Obj* Import( const _Obj& p ); 

    int type ;  
    int left ; 
    int right ; 
};

inline void _Obj::Serialize( _Obj& p, const Obj* o ) // static
{
    std::cerr << "_Obj::Serialize" << std::endl ; 
}
inline Obj* _Obj::Import( const _Obj& p ) // static
{
    std::cerr << "_Obj::Import" << std::endl ; 
    return nullptr ; 
}


