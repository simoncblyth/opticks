#include "Nuv.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>




bool nuv::is_interior(unsigned margin) const 
{
    unsigned _u = u() ; 
    unsigned _v = v() ;
    unsigned _nu = nu(); 
    unsigned _nv = nv(); 

    bool u_interior = _u > margin &&  _u < _nu - margin ; 
    bool v_interior = _v > margin &&  _v < _nv - margin ; 
    bool uv_interior = u_interior && v_interior  ;

/*
    if(!uv_interior)
    std::cout << "nuv::is_interior"
              << " _u " << _u 
              << " _v " << _v
              << " _nu " << _nu 
              << " _nv " << _nv
              << " u_interior " << u_interior
              << " v_interior " << v_interior
              << std::endl 
             ;

*/

     //  NB there is a +1, so nu=4 corresponds to u range 0...nu
     //
     //   |--|--|--|--|
     //   0  1  2  3  4 
     //      ^^^^^^^       (with margin 0 and nu=4)   

    return uv_interior  ; 
}



std::string nuv::desc() const 
{
    std::stringstream ss ; 

    ss << "(" 
       << std::setw(1) << s()
       << ";"
       << std::setw(2) << u() 
       << ","
       << std::setw(2) << v()
       << ")"
       ;

    return ss.str();
}


std::string nuv::detail() const 
{
    std::stringstream ss ; 

    ss << "nuv "
       << " s " << std::setw(1) << s()
       << " u " << std::setw(3) << u() << "/" << std::setw(3) << nu() 
       << " v " << std::setw(3) << v() << "/" << std::setw(3) << nv() 
       ;

    return ss.str();
}



