#pragma once

#include <iostream>
#include <iomanip>
#include <string>

/**
Tran
=====

Transform handler that creates inverse transforms at every stage  
loosely based on NPY nmat4triple_

Aim to avoid having to use glm::inverse which inevitably risks
numerical issues.

**/

#include "glm/glm.hpp"
#include <vector>

template<typename T>
struct Tran
{
    static const Tran<T>* make_translate( const T tx, const T ty, const T tz, const T sc);
    static const Tran<T>* make_translate( const T tx, const T ty, const T tz);
    static const Tran<T>* make_scale(     const T sx, const T sy, const T sz);
    static const Tran<T>* make_rotate(    const T ax, const T ay, const T az, const T angle_deg);

    static const Tran<T>* product(const Tran<T>* a, const Tran<T>* b, bool reverse);
    static const Tran<T>* product(const Tran<T>* a, const Tran<T>* b, const Tran<T>* c, bool reverse);
    static const Tran<T>* product(const std::vector<const Tran<T>*>& tt, bool reverse );


    Tran( const T* transform, const T* inverse ) ;
    Tran( const glm::tmat4x4<T>& transform, const glm::tmat4x4<T>& inverse ) ;

    bool is_identity(char mat='t', T epsilon=1e-6) const ; 
    std::string brief(bool only_tlate=false, char mat='t', unsigned wid=6, unsigned prec=1) const ;  
  
    const T* tdata() const ; 
    const T* vdata() const ; 
 
    glm::tmat4x4<T> t ;
    glm::tmat4x4<T> v ;
    glm::tmat4x4<T> i ;
};



template<typename T>
inline std::ostream& operator<< (std::ostream& out, const glm::tmat4x4<T>& m  )
{
    int prec = 4 ;   
    int wid = 10 ; 
    bool flip = false ; 
    for(int i=0 ; i < 4 ; i++)
    {   
        for(int j=0 ; j < 4 ; j++) out << std::setprecision(prec) << std::fixed << std::setw(wid) << ( flip ? m[j][i] : m[i][j] ) << " " ; 
        out << std::endl ; 
    }   
    return out ; 
}

template<typename T>
inline std::ostream& operator<< (std::ostream& out, const Tran<T>& tr)
{
    out 
       << std::endl 
       << "tr.t" 
       << std::endl 
       <<  tr.t 
       << std::endl 
       << "tr.v" 
       << std::endl 
       <<  tr.v  
       << "tr.i" 
       << std::endl 
       <<  tr.i  
       << std::endl 
       ;   
    return out;
}



