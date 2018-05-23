#pragma once

#include <string>

// trying to fwd declare leads to linker errors for static NPY methods with some tests : G4StepNPYTest.cc, HitsNPYTest.cc see tests/CMakeLists.txt
//template <typename T> class NPY ; 
#include "NPY.hpp"

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"


struct NPY_API NGLMCF
{
    const glm::mat4& A ; 
    const glm::mat4& B ; 

    float epsilon_translation ; 
    float epsilon ; 
    float diff ; 
    float diff2 ; 
    float diffFractional ;
    float diffFractionalMax ;
    bool match ;
 
    NGLMCF( const glm::mat4& A_, const glm::mat4& B_ ) ;
    std::string desc(const char* msg="NGLMCF::desc"); 

};



