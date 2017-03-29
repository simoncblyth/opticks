#pragma once

#include <cstddef>
#include "NPY_API_EXPORT.hh"
#include "NGLM.hpp"

struct NField3 ; 
struct NGrid3 ; 
struct nivec3 ; 
struct nvec3 ; 


struct NPY_API NFieldGrid3  
{
    NFieldGrid3( NField3* field, NGrid3* grid, bool offset=false ) ;

    float value( const nivec3& ijk ) const ;
    nvec3 position( const nivec3& ijk ) const ; 

    float value_f( const glm::vec3& ijkf ) const ;
    float value_f( const nvec3& ijkf ) const ;
    nvec3 position_f( const nvec3& ijkf ) const ;  // floated coordinates 

    NField3* field ; 
    NGrid3*  grid ; 
    bool     offset ; 

};




