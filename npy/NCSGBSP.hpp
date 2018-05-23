#pragma once


#include "OpticksCSG.h"
#include "NPY_API_EXPORT.hh"

#include "NTriSource.hpp" 

struct csgjs_csgnode ; 
struct csgjs_model ; 

struct NPY_API NCSGBSP : NTriSource 
{
    static csgjs_csgnode* ConvertToBSP( const NTriSource* tris) ; 

    NCSGBSP(const NTriSource* left_, const NTriSource* right_, OpticksCSG_t operation );
    void init();

    csgjs_csgnode* left ; 
    csgjs_csgnode* right ; 
    OpticksCSG_t   operation ; 

    csgjs_csgnode* combined ; 
    csgjs_model*   model ; 


    // NTriSource interface
    unsigned get_num_tri() const ;
    unsigned get_num_vert() const ;
    void get_vert( unsigned i, glm::vec3& v ) const ;
    void get_normal( unsigned i, glm::vec3& n ) const ;
    void get_uv(  unsigned i, glm::vec3& uv ) const ;
    void get_tri( unsigned i, glm::uvec3& t ) const ;
    void get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const ;


};


