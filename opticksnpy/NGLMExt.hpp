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

    nmat4pair( const glm::mat4& tr, const glm::mat4& irit ) : tr(tr), irit(irit) {} ;
    std::string digest();

    glm::mat4 tr   ; 
    glm::mat4 irit ; 
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
    static glm::mat4 invert_tr( const glm::mat4& tr ); 
    static glm::mat4* invert_tr( const glm::mat4* tr ); 

    static float compDiff(const glm::mat4& a , const glm::mat4& b );

    static glm::mat4 average_to_inverse_transpose( const glm::mat4& m );
    static ndeco polar_decomposition( const glm::mat4& trs, bool verbose=false );

    static glm::mat4 invert_trs( const glm::mat4& trs ); 
    static glm::mat4 make_transform(const std::string& order, const glm::vec3& tlat, const glm::vec4& axis_angle, const glm::vec3& scal );
    static glm::mat4 make_transform(const std::string& order);

};


NPY_API std::ostream& operator<< (std::ostream& out, const nmat4pair& mp); 
NPY_API std::ostream& operator<< (std::ostream& out, const glm::mat3& v);
NPY_API std::ostream& operator<< (std::ostream& out, const glm::mat4& v); 
NPY_API std::ostream& operator<< (std::ostream& out, const glm::vec4& v); 
NPY_API std::ostream& operator<< (std::ostream& out, const glm::vec3& v);
NPY_API std::ostream& operator<< (std::ostream& out, const glm::ivec3& v);
 

