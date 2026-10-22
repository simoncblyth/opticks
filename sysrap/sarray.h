#pragma once
/**
sarray.h : Minimal Vector Methods to assist with tests
========================================================

From ~/customgeant4/tests/V.h


**/

#include <array>
#include <iomanip>
#include <sstream>
#include <string>

namespace opticks
{
    enum { X, Y, Z } ; 

    /**
                                        
    | a_x |   | b_x |       |   i    j    z   |       |   a_y b_z - a_z b_y   | 
    |     |   |     |       |                 |       |                       |
    | a_y | X | b_y |   =   |  a_x  a_y  a_z  |   =   | -(a_x b_z - a_z b_x)  |
    |     |   |     |       |                 |       |                       |
    | a_z |   | b_z |       |  b_x  b_y  b_z  |       |   a_x b_y - a_y b_x   |

    https://en.wikipedia.org/wiki/Cross_product
    **/

    template<typename F>
    void cross( std::array<F,3>& c, const std::array<F,3>& a, const std::array<F,3>& b )
    {
        // make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
        c[X] = a[Y]*b[Z] - a[Z]*b[Y] ; 
        c[Y] = a[Z]*b[X] - a[X]*b[Z] ;  // note order flip from the -ve  (right-hand-basis)
        c[Z] = a[X]*b[Y] - a[Y]*b[X] ; 
    } 

    template<typename F>
    F dot( const std::array<F,3>& a, const std::array<F,3>& b )
    {
        return a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z] ;  
    } 

    template<typename F>
    void normalize( std::array<F,3>& n , const std::array<F,3>& v )
    {
        const F one(1.); 
        F invLen = one / sqrt(V::dot<F>(v, v));
        n[X] = invLen*v[X] ; 
        n[Y] = invLen*v[Y] ; 
        n[Z] = invLen*v[Z] ; 
    }
    

    /**
    V::rotateUz
    ------------

    V::rotateUz inplace rotates vector *d* such that its original Z-axis 
    ends up in the direction of *u*. Many rotations would accomplish this. 
    The one selected uses *u* as its third column. For a full explanation 
    of the implementation and tests see smath::rotateUz from ~/opticks/sysrap/smath.h 
    **/

    template<typename F>
    void rotateUz(std::array<F,3>& d, const std::array<F,3>& u ) 
    {
        F zero(0.); 
        F up = u[X]*u[X] + u[Y]*u[Y] ;
        if (up>zero) 
        {   
            up = sqrt(up);
            F px = d[X] ;
            F py = d[Y] ;
            F pz = d[Z] ;
            d[X] = (u[X]*u[Z]*px - u[Y]*py)/up + u[X]*pz;
            d[Y] = (u[Y]*u[Z]*px + u[X]*py)/up + u[Y]*pz;
            d[Z] =    -up*px +                   u[Z]*pz;
        }   
        else if (u[Z] < zero ) 
        {   
            d[X] = -d[X]; 
            d[Z] = -d[Z]; 
        }         
    }

    /**
    V::make_transverse
    -------------------

    Form vector *pol* that is transverse to mom chosen with XY phase of *frac_twopi*
    such that phase zero corresponds to the original frame X direction.
    
    This is adapted from sphoton::set_polarization from  ~/opticks/sysrap/sphoton.h 
    **/

    template<typename F>
    void make_transverse( std::array<F,3>& pol, const std::array<F,3>& mom , F frac_twopi )
    {
        F phase = 2.*M_PI*frac_twopi ; 
        pol[X] = cosf(phase) ; 
        pol[Y] = sinf(phase) ; 
        pol[Z] = 0. ; 
        rotateUz(pol, mom); // rotate pol to be transverse to mom
    } 

    template<typename F>
    F length( const std::array<F,3>& v )
    {
        return sqrtf( dot(v,v) );  
    }

    template<typename F>
    std::string desc( const std::array<F,3>& a )
    {
        std::stringstream ss ; 
        ss 
           << " [ "
           << std::setw(10) << std::setprecision(4) << std::fixed << a[X] 
           << " " 
           << std::setw(10) << std::setprecision(4) << std::fixed << a[Y] 
           << " " 
           << std::setw(10) << std::setprecision(4) << std::fixed << a[Z] 
           << " ] "
           ;
        std::string str = ss.str(); 
        return str ; 
    } 
}


