#pragma once

#include <string>
#include <array>
#include <functional>

class NCSG ; 

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

/**

NCSGIntersect
==============

Collect min/max/avg signed distances to an NCSG solid surface 
and time ranges for up to 16 indexed points.

**/


struct NPY_API NCSGIntersect
{
    typedef std::function<float(float,float,float)> SDF ;

    void init(NCSG* csg);
    void add( unsigned idx, const glm::vec4& post );

    std::string desc_time(unsigned idx, const char* label=NULL) const ;
    std::string desc_dist(unsigned idx, const char* label=NULL) const ;



    NCSG*                       _csg  ;  
    SDF                         _sdf  ; 

    std::array<unsigned, 16>    _count ; 
    std::array<glm::vec4, 16>   _dist ; 
    std::array<glm::vec4, 16>   _time ; 

};
