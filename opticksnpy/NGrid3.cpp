#include <sstream>
#include "mortonlib/morton3d.h"

#include "NGrid3.hpp"
#include "PLOG.hh"
#include "NGLM.hpp"


NMultiGrid3::NMultiGrid3()
{
   for(int i=0 ; i < NGRID ; i++ ) grid[i] = new NGrid3(i) ; 
}

void NMultiGrid3::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    for(int level=0 ; level < NGRID ; level++)
         std::cout << grid[level]->desc() 
                   << std::endl ; 
}

void NMultiGrid3::dump(const char* msg, const nvec3& fpos) const 
{
    LOG(info) << msg ; 
    for(int level=0 ; level < NGRID ; level++)
         std::cout << grid[level]->desc() 
                   << grid[level]->desc( fpos, " fpos " ) 
                   << std::endl ; 
}






std::string NGrid3::desc(const nivec3& ijk, const char* msg)
{
    morton3 m(ijk.x, ijk.y, ijk.z);
    std::stringstream ss ;  
    ss << msg  << std::setw(12) << ijk.desc()
       << " m "  << std::setw(12) << m.key 
       << " m>>3 "  << std::setw(12) << (m.key >> 3)
       << " m>>6 "  << std::setw(12) << (m.key >> 6)
       << " m>>9 "  << std::setw(12) << (m.key >> 9)
       ;
    return ss.str();
}


NGrid3::NGrid3( int level )  // NB everything from the level 
    :
    level(level),
    size( 1 << level ),
    nloc( 1 << (3*level) ),
    nijk( size, size, size),
    elem( 1./size ),
    half_min( -size/2, -size/2, -size/2 ),
    half_max(  size/2,  size/2,  size/2 )
{
    assert(level >= 0 && level < MAXLEVEL);
} 




std::string NGrid3::desc() const 
{
    std::stringstream ss ;  
    ss << "NGrid3"
       << " level " << std::setw(2) << level
       << " size "  << std::setw(5) << size
       << " nloc "  << std::setw(12) << nloc
       << " elem "  << std::setw(12) << elem
       ;
    return ss.str();
}

std::string NGrid3::desc(const nvec3& fpos, const char* msg)  const 
{
    nivec3 ijk_ = ijk(fpos);
    std::stringstream ss ;  
    ss << msg 
       << " " << fpos.desc()
       << NGrid3::desc(ijk_, " ijk ")
       ; 

    return ss.str();
} 


nivec3 NGrid3::ijk(const int c) const
{ 
    bool valid = c < nloc && c > -1 ;
    if(!valid)
        LOG(fatal) << "NGrid3::ijk invalid loc " << c << " for grid " << desc() ; 
        
    assert(valid);

    morton3 loc(c);  
    unsigned long long i, j, k ;  
    loc.decode(i, j, k); 
    return nivec3(i, j, k);
}



int NGrid3::loc(const nivec3& ijk ) const 
{
    morton3 mloc(ijk.x, ijk.y, ijk.z);
    return mloc.key ;   
}

int NGrid3::loc(const int i, const int j, const int k) const 
{
    morton3 mloc(i, j, k);
    return mloc.key ;   
}



int NGrid3::loc(const nvec3& fpos ) const 
{
    nivec3 ijk_ = ijk(fpos); 
    return loc(ijk_);
}




nivec3 NGrid3::ijk(const nvec3& fpos) const 
{
    return nivec3( nijk.x*fpos.x, nijk.y*fpos.y , nijk.z*fpos.z ) ; 
}


template<typename T>
nvec3 NGrid3::fpos(const T& ijk ) const 
{
    return make_nvec3( float(ijk.x)/float(nijk.x), float(ijk.y)/float(nijk.y), float(ijk.z)/float(nijk.z) ); 
}



nvec3 NGrid3::fpos(const int c) const
{
    nivec3 ijk_ = ijk(c) ; 
    return fpos(ijk_);
}


template nvec3 NGrid3::fpos(const glm::ivec3& ) const ;
template nvec3 NGrid3::fpos(const nivec3& ) const ;

// floated coordinates OK too
template nvec3 NGrid3::fpos(const glm::vec3& ) const ;
template nvec3 NGrid3::fpos(const nvec3& ) const ;



