#pragma once

#include "s_pool.h"

/**
Obj : type to be persisted needs to instrument ctor and dtor with pool.add(this) pool.remove(this) 
**/

struct Obj ; 

struct Obj 
{
    typedef s_pool<Obj> POOL ;
    static POOL pool ;

    Obj( int type, Obj* left=nullptr, Obj* right=nullptr ); 
    ~Obj();  

    int   pid ;   // HMM: not required, but useful for debug   

    int   type ; 
    Obj*  left ; 
    Obj*  right ; 
};


inline Obj::Obj( int type_, Obj* left_, Obj* right_ )
    :
    pid(pool.add(this)),
    type(type_),
    left(left_),
    right(right_)
{
}
inline Obj::~Obj()
{
    delete left ; 
    delete right ; 

    int pid1 = pool.remove(this) ; 
    assert( pid == pid1 );  
}

