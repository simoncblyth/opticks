#include <sstream>
#include <iostream>
#include <iomanip>

#include "NNode.hpp"
#include "NCSG.hpp"
#include "NCSGIntersect.hpp"
#include "GLMFormat.hpp"




void NCSGIntersect::init(NCSG* csg)
{
    _csg = csg ; 
    nnode* root = _csg->getRoot();
    _sdf = root->sdf() ;

    for(unsigned idx=0 ; idx < 16 ; idx++)
    {
        _time[idx] = glm::vec4(0,0,0,0) ; 
        _dist[idx] = glm::vec4(0,0,0,0) ; 
        _count[idx] = 0 ; 
    }
}


/**
https://en.wikipedia.org/wiki/Moving_average

"Cumulative Moving Average"

              x1 + ... + xn
    CMA_n  = -----------------
                   n

                 x_n+1  + n*CMA_n
    CMA_n+1 =   ------------------
                   n+1

**/

float CMA( float a , float v, unsigned n )
{
   return (float(n)*a + v)/float(n+1) ;  
}


std::string NCSGIntersect::desc_dist(unsigned idx, const char* label) const
{
    const unsigned& n = _count[idx] ; 
    const glm::vec4& dist = _dist[idx] ;   
    std::stringstream ss ; 
    ss << ( label ? label : "" ) 
       << "[p:" << std::setw(2) << idx << "]"
       << "(" << std::setw(7) << n << ")"
       << std::setw(30) << gpresent( dist ) << " mm" 
       ;
    return ss.str();
} 

std::string NCSGIntersect::desc_time(unsigned idx, const char* label) const
{
    const unsigned& n = _count[idx] ; 
    const glm::vec4& time = _time[idx] ;   
    std::stringstream ss ; 
    ss << ( label ? label : "" ) 
       << "[p:" << std::setw(2) << idx << "]"
       << "(" << std::setw(7) << n << ")"
       << std::setw(30) << gpresent( time ) << " ns"
       ;
    return ss.str();
} 





void NCSGIntersect::add(unsigned idx, const glm::vec4& post )
{
    assert( idx < 16 ) ; 

    float sd = _sdf(post.x, post.y, post.z ) ;
    float td = post.w ; 

    unsigned& n = _count[idx] ; 
    glm::vec4& dist = _dist[idx] ;   
    glm::vec4& time = _time[idx] ;   

    if( n == 0)
    {
        dist.x = sd ;    // min
        dist.y = sd ;    // max
        dist.z = sd ;    // avg
        dist.w  = 0 ;    // spare

        time.x = td ;    // min
        time.y = td ;    // max
        time.z = td ;    // cma  
        time.w  = 0 ;    // spare
    }
    else
    {
        dist.x = sd < dist.x ? sd : dist.x ; 
        dist.y = sd > dist.y ? sd : dist.y ; 
        dist.z = CMA( dist.z , sd ,  n  ) ;  
        dist.w = 0. ; 

        time.x = td < time.x ? td : time.x ; 
        time.y = td > time.y ? td : time.y ; 
        time.z = CMA( time.z , td ,  n  ) ;  
        time.w = 0. ; 
    }
    n++ ; 

/*
    std::cout << " idx " << std::setw(3) << idx 
              << " post " << std::setw(30) << gpresent( post )
              << " n " << std::setw(2) << n
              << " sd " << std::setw(10) << sd 
              << " td " << std::setw(10) << td 
              << " dist " << std::setw(30) << gpresent( dist )
              << " time " << std::setw(30) << gpresent( time )
              << std::endl 
              ;
*/



}
