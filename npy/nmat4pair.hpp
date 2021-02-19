#pragma once

#include <vector>
#include <string>

#include "NPY.hpp"

#include "plog/Severity.h"
#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"


struct NPY_API nmat4pair 
{
    static nmat4pair* product(const std::vector<nmat4pair*>& tt);

    nmat4pair* clone();
    nmat4pair( const glm::mat4& transform ); 
    nmat4pair( const glm::mat4& transform, const glm::mat4& inverse ) : t(transform), v(inverse) {} ;
    std::string digest();

    bool match ; 
    glm::mat4 t ; 
    glm::mat4 v ; 
};



