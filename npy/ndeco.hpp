#pragma once

#include "NPY.hpp"
#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"


template <typename T>
struct NPY_API ndeco_
{  
    glm::tmat4x4<T> t ; 
    glm::tmat4x4<T> r ; 
    glm::tmat4x4<T> s ; 

    glm::tmat4x4<T> it ; 
    glm::tmat4x4<T> ir ; 
    glm::tmat4x4<T> is ; 

    glm::tmat4x4<T> rs ;

    glm::tmat4x4<T> tr ;
    glm::tmat4x4<T> trs ;
    glm::tmat4x4<T> isirit ;

    int count ; 
    T   diff ; 
    T   diff2 ; 
    T   epsilon ; 
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

    glm::mat4 tr ;
    glm::mat4 trs ;
    glm::mat4 isirit ;

    int count ; 
    float diff ; 
    float diff2 ; 
    float epsilon ; 
};


