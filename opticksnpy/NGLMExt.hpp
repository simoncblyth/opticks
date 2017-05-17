#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

struct NPY_API nmat4pair 
{
    static nmat4pair* product(const std::vector<nmat4pair*>& tt);

    nmat4pair* clone();
    nmat4pair( const glm::mat4& transform ); 
    nmat4pair( const glm::mat4& transform, const glm::mat4& inverse ) : t(transform), v(inverse) {} ;
    std::string digest();

    glm::mat4 t ; 
    glm::mat4 v ; 
};


struct NPY_API nmat4triple
{
    static nmat4triple* product(const std::vector<nmat4triple*>& tt, bool swap=false);
    static nmat4triple* make_translated(nmat4triple* src, const glm::vec3& tlate );
    static nmat4triple* make_transformed(nmat4triple* src, const glm::mat4& txf, bool pre);

    nmat4triple* clone();
    nmat4triple* make_translated(const glm::vec3& tlate );
    nmat4triple( const glm::mat4& transform ); 
    nmat4triple( float* data ); 
    nmat4triple( const glm::mat4& transform, const glm::mat4& inverse, const glm::mat4& inverse_T ) 
         : 
            t(transform), 
            v(inverse), 
            q(inverse_T) 
         {} ;

    std::string digest();


    glm::mat4 t ; 
    glm::mat4 v ; 
    glm::mat4 q ; 
};



struct NPY_API ndeco
{  
    glm::mat4 t ; 
    glm::mat4 r ; 
    glm::mat4 s ; 

    glm::mat4 it ; 
    glm::mat4 ir ; 
    glm::mat4 is ; 

    glm::mat4 rs ;

    glm::mat4 trs ;
    glm::mat4 isirit ;
};


struct NPY_API nglmext 
{ 
    static void copyTransform( std::array<float,16>& dst, const glm::mat4& src );
    static std::string xform_string( const std::array<float, 16>& xform );
    static std::array<float, 16> _float4x4_mul( const std::array<float, 16>& a, const std::array<float, 16>& b) ;
    static const std::array<float, 16> _identity_float4x4 ;

    static glm::mat4 invert_tr( const glm::mat4& tr ); 
    static glm::mat4* invert_tr( const glm::mat4* tr ); 

    static float compDiff2(const glm::mat4& a , const glm::mat4& b, bool fractional=false, float epsilon=1e-7);
    static float compDiff2(const float a_ , const float b_, bool fractional=false, float epsilon=1e-7);
    static float compDiff(const glm::mat4& a , const glm::mat4& b );

    static glm::mat4 average_to_inverse_transpose( const glm::mat4& m );
    static ndeco polar_decomposition( const glm::mat4& trs, bool verbose=false );

    static glm::mat4 invert_trs( const glm::mat4& trs ); 
    static glm::mat4 make_transform(const std::string& order, const glm::vec3& tlat, const glm::vec4& axis_angle, const glm::vec3& scal );
    static glm::mat4 make_transform(const std::string& order);

};


NPY_API std::ostream& operator<< (std::ostream& out, const nmat4triple& triple); 
NPY_API std::ostream& operator<< (std::ostream& out, const nmat4pair& pair); 

NPY_API std::ostream& operator<< (std::ostream& out, const glm::mat3& v);
NPY_API std::ostream& operator<< (std::ostream& out, const glm::mat4& v); 
NPY_API std::ostream& operator<< (std::ostream& out, const glm::vec4& v); 
NPY_API std::ostream& operator<< (std::ostream& out, const glm::vec3& v);
NPY_API std::ostream& operator<< (std::ostream& out, const glm::ivec3& v);
 

