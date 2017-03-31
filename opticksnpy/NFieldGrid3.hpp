#pragma once

#include <cstddef>
#include "NPY_API_EXPORT.hh"
#include "NGLM.hpp"

template <typename FVec, typename IVec, int DIM> struct NField ; 
template <typename FVec, typename IVec, int DIM> struct NGrid ; 


template<typename FVec, typename IVec>
struct NPY_API NFieldGrid3  
{
    NFieldGrid3( NField<FVec,IVec,3>* field, NGrid<FVec,IVec,3>* grid, bool offset=false ) ;
    
    // grid coordinate to field value
    float value( const IVec& ijk ) const ;
    float value_f( const FVec& ijkf ) const ;

    // grid coordinate to world position
    FVec position( const IVec& ijk ) const ; 
    FVec position_f( const FVec& ijkf ) const ;  

    NField<FVec,IVec,3>* field ; 
    NGrid<FVec,IVec,3>*  grid ; 

    bool     offset ; 

};




