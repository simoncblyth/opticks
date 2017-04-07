#pragma once

#include <iostream>
#include <iomanip>
#include <vector>

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

struct NPY_API nmat4pair 
{
    static nmat4pair* product(const std::vector<nmat4pair*>& tt);

    nmat4pair( const glm::mat4& tr, const glm::mat4& irit ) : tr(tr), irit(irit) {} ;

    glm::mat4 tr   ; 
    glm::mat4 irit ; 
};



NPY_API glm::mat4 invert_tr( const glm::mat4& tr ); 

NPY_API std::ostream& operator<< (std::ostream& out, const nmat4pair& mp); 



inline std::ostream& operator<< (std::ostream& out, const glm::ivec3& v) 
{
    out << "{" 
        << " " << std::setw(4) << v.x 
        << " " << std::setw(4) << v.y 
        << " " << std::setw(4) << v.z
        << "}";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const glm::vec3& v) 
{
    out << "{" 
        << " " << std::fixed << std::setprecision(4) << std::setw(9) << v.x 
        << " " << std::fixed << std::setprecision(4) << std::setw(9) << v.y
        << " " << std::fixed << std::setprecision(4) << std::setw(9) << v.z 
        << "}";

    return out;
}

inline std::ostream& operator<< (std::ostream& out, const glm::vec4& v) 
{
    out << "{" 
        << " " << std::setprecision(2) << std::setw(7) << v.x 
        << " " << std::setprecision(2) << std::setw(7) << v.y
        << " " << std::setprecision(2) << std::setw(7) << v.z 
        << " " << std::setprecision(2) << std::setw(7) << v.w 
        << "}";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const glm::mat4& v) 
{
    out << "( "
        << " " << v[0]
        << " " << v[1]
        << " " << v[2]
        << " " << v[3]
        << " )"
        ; 

    return out;
}


inline std::ostream& operator<< (std::ostream& out, const glm::mat3& v) 
{
    out << "( "
        << " " << v[0]
        << " " << v[1]
        << " " << v[2]
        << " )"
        ; 

    return out;
}




