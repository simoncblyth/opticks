//clang NGrid3.cpp -I$(glm-dir) -c -o /dev/null
#pragma once

#include <functional>
#include <string>
#include <cassert>

#include "NQuad.hpp"

#include "NPY_API_EXPORT.hh"


template <typename FVec, typename IVec, int DIM>
struct NPY_API NGrid
{
    static const int MAXLEVEL = 11 ;   // 32 bits not enough for level 11 
    static std::string desc(const IVec& ijk, const char* msg="ijk");

    NGrid(int level);

    std::string desc() const ;
    std::string desc(const FVec& fpos, const char* msg="fpos" ) const ; 

    FVec fpos(const IVec& ijk ) const ;  // grid int coordinates in 0:size-1 to fractional coordinates in 0:1. NB accepts floated coordinates too

    IVec ijk(const FVec& fpos) const ;   // fractional coordinates in 0:1. to grid int coordinates in 0:size-1
    IVec ijk(int c) const ;               // z-order morton code in 0:nloc-1 to grid int coordinates in 0:size-1

    FVec fpos(int c ) const ;             // z-order morton code in 0:nloc-1 to fractional grid coordinates in 0:1

    int    loc(const IVec&  ijk ) const ;  // grid int coordinates in 0:size-1 to z-order morton code in 0:nloc-1  
    int    loc(const FVec& fpos ) const ;  // fractional coordinates in 0:1 to z-order morton code in 0:nloc-1  
    int    loc(const int i, const int j, const int k) const ;  // grid int coordinates in 0:nsize-1 to z-order morton code in 0:nloc-1  


    int upscale_factor( const NGrid& coarser ) const 
    { 
        assert((level - coarser.level) >= 0 ); 
        return 1 << (level - coarser.level) ; 
    }  

    int voxel_size(int elevation) const { return 1 << elevation ; }   
                                        // size of voxel at different depth, ie  level - elevation, 
                                        // relative to the nominal voxels for this grid level,
                                        // 
                                        //       elevation 0 -> 1 by construction
                                        //       elevation 1 -> 2       ( 1 << elevation )

    int voxel_num(int elevation) const { return 1 << (DIM*elevation) ; } 
                                        // number of nominal voxels of this grid within a subgrid 
                                        // of another grid at different elevation 
                                        // eg for handling coarse tiles

    const int    level ;  
    const int    size ;       // 1<<level
    const int    nloc ;      
    const IVec   nijk ; 
    const float  elem ; 
    const IVec   half_min ; // horrible half_min
    const IVec   half_max ; // horrible half_max 

};


template<typename FVec, typename IVec, int DIM>
inline NPY_API bool operator == (const NGrid<FVec,IVec,DIM>& a, const NGrid<FVec,IVec,DIM>& b )
{
   return a.level == b.level  ;  
}



template<typename FVec, typename IVec>
struct NPY_API NMultiGrid3
{
    enum { NGRID = 10 };
    NMultiGrid3();
    NGrid<FVec,IVec,3>* grid[NGRID] ; 

    void dump(const char* msg) const ; 
    void dump(const char* msg, const FVec& fpos) const ; 
};





