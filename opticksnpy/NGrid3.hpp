#pragma once

#include <functional>
#include <string>
#include <cassert>

#include "NQuad.hpp"

#include "NPY_API_EXPORT.hh"



struct NPY_API NGrid3 
{
    static const int MAXLEVEL = 11 ;   // 32 bits not enough for level 11 
    static std::string desc(const nivec3& ijk, const char* msg="ijk");

    NGrid3(int level);

    std::string desc() const ;
    std::string desc(const nvec3& fpos, const char* msg="fpos" ) const ; 

    template<typename T>
    nvec3  fpos(const T& ijk ) const ;      // grid int coordinates in 0:size-1 to fractional coordinates in 0:1.
    nivec3 ijk(const nvec3& fpos) const ;   // fractional coordinates in 0:1. to grid int coordinates in 0:size-1
    nivec3 ijk(int c) const ;               // z-order morton code in 0:nloc-1 to grid int coordinates in 0:size-1
    nvec3  fpos(int c ) const ;             // z-order morton code in 0:nloc-1 to fractional grid coordinates in 0:1

    int    loc(const nivec3& ijk ) const ;  // grid int coordinates in 0:size-1 to z-order morton code in 0:nloc-1  
    int    loc(const nvec3& fpos ) const ;  // fractional coordinates in 0:1 to z-order morton code in 0:nloc-1  
    int    loc(const int i, const int j, const int k) const ;  // grid int coordinates in 0:nsize-1 to z-order morton code in 0:nloc-1  

    //nivec3 upscale_ijk(const nivec& ijk, const NGrid3& other, bool offset=false);  // this grids coordinate upscaled to another grid

    int upscale_factor( const NGrid3& coarser ) const 
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

    int voxel_num(int elevation) const { return 1 << (3*elevation) ; } 
                                        // number of nominal voxels of this grid within a subgrid 
                                        // of another grid at different elevation 
                                        // eg for handling coarse tiles

    const int    level ;  
    const int    size ;       // 1<<level
    const int    nloc ;      
    const nivec3 nijk ; 
    const float  elem ; 
    const nivec3 half_min ; // horrible half_min
    const nivec3 half_max ; // horrible half_max 

};


struct NPY_API NMultiGrid3 
{
    enum { NGRID = 10 };
    NMultiGrid3();
    NGrid3* grid[NGRID] ; 

    void dump(const char* msg) const ; 
    void dump(const char* msg, const nvec3& fpos) const ; 
};




/**

Morton magic  : provides indices for same position at multiple resolutions, and a way to go from one resolution to another by bit shifts 

NGrid3 level  0 side     1 nloc            1 fpos   (   0.10    0.20    0.30)  ijk  (    0     0     0)  m            0 m>>3            0 m>>6            0 m>>9            0
NGrid3 level  1 side     2 nloc            8 fpos   (   0.10    0.20    0.30)  ijk  (    0     0     0)  m            0 m>>3            0 m>>6            0 m>>9            0
NGrid3 level  2 side     4 nloc           64 fpos   (   0.10    0.20    0.30)  ijk  (    0     0     1)  m            1 m>>3            0 m>>6            0 m>>9            0
NGrid3 level  3 side     8 nloc          512 fpos   (   0.10    0.20    0.30)  ijk  (    0     1     2)  m           10 m>>3            1 m>>6            0 m>>9            0
NGrid3 level  4 side    16 nloc         4096 fpos   (   0.10    0.20    0.30)  ijk  (    1     3     4)  m           86 m>>3           10 m>>6            1 m>>9            0
NGrid3 level  5 side    32 nloc        32768 fpos   (   0.10    0.20    0.30)  ijk  (    3     6     9)  m          693 m>>3           86 m>>6           10 m>>9            1
NGrid3 level  6 side    64 nloc       262144 fpos   (   0.10    0.20    0.30)  ijk  (    6    12    19)  m         5545 m>>3          693 m>>6           86 m>>9           10
NGrid3 level  7 side   128 nloc      2097152 fpos   (   0.10    0.20    0.30)  ijk  (   12    25    38)  m        44362 m>>3         5545 m>>6          693 m>>9           86
NGrid3 level  8 side   256 nloc     16777216 fpos   (   0.10    0.20    0.30)  ijk  (   25    51    76)  m       354902 m>>3        44362 m>>6         5545 m>>9          693
NGrid3 level  9 side   512 nloc    134217728 fpos   (   0.10    0.20    0.30)  ijk  (   51   102   153)  m      2839221 m>>3       354902 m>>6        44362 m>>9         5545
NGrid3 level 10 side  1024 nloc   1073741824 fpos   (   0.10    0.20    0.30)  ijk  (  102   204   307)  m     22713769 m>>3      2839221 m>>6       354902 m>>9        44362


**/



