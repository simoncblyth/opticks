#pragma once
/**
sfr.h
======


Unionized glm::tvec4 see examples/UseGLMSimple

**/

#include <glm/glm.hpp>



template<typename R, typename I>
struct sfr
{
    glm::tvec4<R> ce ; 


    glm::tmat4<R> m2w ; 
    glm::tmat4<R> w2m ; 
    glm::tvec4<I> aux ; 

}; 
