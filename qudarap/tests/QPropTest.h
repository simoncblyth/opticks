#pragma once

#include "NPFold.h"
#include "QProp.hh"

template <typename T>
struct QPropTest
{
    static constexpr const char* RELDIR = sizeof(T) == 8 ? "double" : "float" ; 
    const QProp<T>* qprop ; 
    int nx ; 
    const NP* a ; 
    const NP* x ; 
    NP* y ; 

    QPropTest( const NP* propcom, T x0, T x1, int nx_ ); 
    void lookup(); 
    NPFold* serialize() const ; 
    void save() const ; 
}; 

template <typename T>
inline QPropTest<T>::QPropTest( const NP* propcom, T x0, T x1, int nx_ )
    :
    qprop(new QProp<T>(propcom)),
    nx(nx_),
    a(qprop->a), 
    x(NP::Linspace<T>( x0, x1, nx )),
    y(NP::Make<T>(qprop->ni, nx ))
{
}


/**
QPropTest::lookup
-------------------

nx lookups in x0->x1 inclusive for each property yielding nx*qp.ni values.

1. create *x* domain array of shape (nx,) with values in range x0 to x1 
2. create *y* lookup array of shape (qp.ni, nx ) 
3. invoke QProp::lookup collecting *y* lookup values from kernel call 
4. save prop, domain and lookup into fold/reldir

**/

template <typename T>
inline void QPropTest<T>::lookup()
{
    qprop->lookup(y->values<T>(), x->cvalues<T>(), qprop->ni, nx );
}

template <typename T>
inline NPFold* QPropTest<T>::serialize() const
{
    NPFold* f = new NPFold ; 
    f->add("prop", a );  
    f->add("domain", x );  
    f->add("lookup", y );  
    return f ; 
}

template <typename T>
inline void QPropTest<T>::save() const
{
    NPFold* f = serialize(); 
    f->save("$FOLD", RELDIR ) ; 
}



