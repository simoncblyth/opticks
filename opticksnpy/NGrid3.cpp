#include <sstream>
#include <iostream>
#include <iomanip>

#include "mortonlib/morton3d.h"

#include "NGrid3.hpp"



template <typename FVec, typename IVec>
NMultiGrid3<FVec,IVec>::NMultiGrid3()
{
   for(int i=0 ; i < NGRID ; i++ ) grid[i] = new NGrid<FVec,IVec,3>(i) ; 
}

template <typename FVec, typename IVec>
void NMultiGrid3<FVec,IVec>::dump(const char* msg) const 
{
    std::cout << msg << std::endl ; 
    for(int level=0 ; level < NGRID ; level++)
         std::cout << grid[level]->desc() 
                   << std::endl ; 
}

template <typename FVec, typename IVec>
void NMultiGrid3<FVec,IVec>::dump(const char* msg, const FVec& fpos) const 
{
    std::cout << msg << std::endl ; 
    for(int level=0 ; level < NGRID ; level++)
         std::cout << grid[level]->desc() 
                   << grid[level]->desc( fpos, " fpos " ) 
                   << std::endl ; 
}









template <typename FVec, typename IVec, int DIM>
std::string NGrid<FVec,IVec,DIM>::desc(const IVec& ijk, const char* msg)
{
    morton3 m(ijk.x, ijk.y, ijk.z);
    std::stringstream ss ;  

    ss << msg  
       << " ijk (" 
       << std::setw(4) << ijk.x << "," 
       << std::setw(4) << ijk.y << "," 
       << std::setw(4) << ijk.z 
       << ")" 
       << " m "  << std::setw(12) << m.key 
       << " m>>3 "  << std::setw(12) << (m.key >> 3)
       << " m>>6 "  << std::setw(12) << (m.key >> 6)
       << " m>>9 "  << std::setw(12) << (m.key >> 9)
       ;

    return ss.str();
}


template<typename FVec,typename IVec,int DIM>
NGrid<FVec,IVec,DIM>::NGrid( int level )  // NB everything from the level 
    :
    level(level),
    size( 1 << level ),
    nloc( 1 << (DIM*level) ),
    nijk( size, size, size),
    elem( 1./size ),
    half_min( -size/2, -size/2, -size/2 ),
    half_max(  size/2,  size/2,  size/2 )
{
    assert(level >= 0 && level < MAXLEVEL);
} 




template<typename FVec,typename IVec,int DIM>
std::string NGrid<FVec,IVec,DIM>::desc() const 
{
    std::stringstream ss ;  
    ss << "NGrid"
       << " dim " << DIM
       << " level " << std::setw(2) << level
       << " size "  << std::setw(5) << size
       << " nloc "  << std::setw(12) << nloc
       << " elem "  << std::setw(12) << elem
       ;
    return ss.str();
}

template<typename FVec,typename IVec,int DIM>
std::string NGrid<FVec,IVec,DIM>::desc(const FVec& fpos, const char* msg)  const 
{
    IVec ijk_ = ijk(fpos);
    std::stringstream ss ;  
    ss << msg 
       << " fpos (" 
       << std::setw(5) << fpos.x << ","
       << std::setw(5) << fpos.y << ","
       << std::setw(5) << fpos.z << ")"
       << NGrid<FVec,IVec,DIM>::desc(ijk_, " ijk ")
       ; 

    return ss.str();
} 


template<typename FVec,typename IVec,int DIM>
IVec NGrid<FVec,IVec,DIM>::ijk(const int c) const  
{ 
   // morton code to integer grid coordinate
    bool valid = c < nloc && c > -1 ;
    if(!valid)
        std::cerr << "NGrid::ijk invalid loc " << c << " for grid " << desc() ; 
        
    assert(valid);

    morton3 loc(c);  
    unsigned long long i, j, k ;  
    loc.decode(i, j, k); 
    return IVec(i, j, k);
}

template<typename FVec,typename IVec,int DIM>
IVec NGrid<FVec,IVec,DIM>::ijk(const FVec& fpos) const 
{
   // fractional to integer coordinates
    return IVec( nijk.x*fpos.x, nijk.y*fpos.y , nijk.z*fpos.z ) ; 
}


template<typename FVec,typename IVec,int DIM>
int NGrid<FVec,IVec,DIM>::loc(const IVec& ijk ) const 
{
   // integer coordinate to morton code
    morton3 mloc(ijk.x, ijk.y, ijk.z);
    return mloc.key ;   
}


template<typename FVec,typename IVec,int DIM>
int NGrid<FVec,IVec,DIM>::loc(const FVec& fpos ) const 
{
   // fractional coordinate to morton code
    IVec ijk_ = ijk(fpos); 
    return loc(ijk_);
}



template<typename FVec,typename IVec,int DIM>
FVec NGrid<FVec,IVec,DIM>::fpos(const IVec& ijk ) const 
{
   // integer (or floated integer) coordinate to fractional coordinate 
    return FVec( float(ijk.x)/float(nijk.x), float(ijk.y)/float(nijk.y), float(ijk.z)/float(nijk.z) ); 
}


template<typename FVec,typename IVec,int DIM>
FVec NGrid<FVec,IVec,DIM>::fpos(const int c) const
{
   // morton code to fractional coordinate 
    IVec ijk_ = ijk(c) ; 
    return fpos(ijk_);
}



#include "NGLM.hpp"
#include "NQuad.hpp"


template struct NPY_API NGrid<glm::vec3, glm::ivec3, 3> ; 
template struct NPY_API NMultiGrid3<glm::vec3, glm::ivec3> ; 

template struct NPY_API NGrid<nvec3, nivec3, 3> ; 
template struct NPY_API NMultiGrid3<nvec3, nivec3> ; 


