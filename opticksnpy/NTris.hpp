#pragma once

#include <vector>
#include <string>

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

struct NPY_API NTris 
{
    std::vector<glm::vec3>  verts ; 
    std::vector<glm::uvec3> tris ; 

    void add( const glm::vec3& a, const glm::vec3& b, const glm::vec3& s);

    unsigned get_num_tri() const ;
    unsigned get_num_vert() const ;
    void get_vert( unsigned i, glm::vec3& v ) const ;
    void get_tri( unsigned i, glm::uvec3& t ) const ;
    void get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const ;
    void dump(const char* msg="Tris::dump") const ;
    std::string brief() const ;


    static NTris* make_sphere( unsigned n_polar=8, unsigned n_azimuthal=8, float ctmin=-1.f, float ctmax=1.f ) ;

};

