#pragma once

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

struct NPY_API NTriSource 
{
    virtual unsigned get_num_tri() const = 0 ;
    virtual unsigned get_num_vert() const = 0 ;
    virtual void get_vert( unsigned i, glm::vec3& v ) const = 0 ;
    virtual void get_normal( unsigned i, glm::vec3& n ) const = 0 ;
    virtual void get_uv(  unsigned i, glm::vec3& v ) const = 0 ;
    virtual void get_tri( unsigned i, glm::uvec3& t ) const = 0 ;
    virtual void get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const = 0;

};

