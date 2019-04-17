#pragma once

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

struct NPY_API ellipse
{
    static const unsigned NSTEP ; 
    ellipse( double ex, double ey ); 
    glm::dvec2 hemi ; 
    glm::dvec2 closest_approach_to_point( const glm::dvec2& p );
};


