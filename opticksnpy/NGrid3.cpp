#include <sstream>
#include "mortonlib/morton3d.h"

#include "NGrid3.hpp"
#include "PLOG.hh"




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


NGrid3::NGrid3( int level )
    :
    level(level),
    size( 1 << level ),
    nloc( 1 << (3*level) ),
    nijk( size, size, size)
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
       ;
    return ss.str();
}

std::string NGrid3::desc(nvec3& fpos, const char* msg)  const 
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
    assert(c < nloc && c > -1);
    morton3 loc(c);  
    unsigned long long i, j, k ;  
    loc.decode(i, j, k); 
    return nivec3(i, j, k);
}

nivec3 NGrid3::ijk(const nvec3& fpos) const 
{
    return nivec3( nijk.x*fpos.x, nijk.y*fpos.y , nijk.z*fpos.z ) ; 
}

nvec3 NGrid3::fpos(const nivec3& ijk ) const 
{
    return make_nvec3( float(ijk.x)/float(nijk.x), float(ijk.y)/float(nijk.y), float(ijk.z)/float(nijk.z) ); 
}

nvec3 NGrid3::fpos(const int c) const
{
    nivec3 ijk_ = ijk(c) ; 
    return fpos(ijk_);
}






void NGrid3::dump_levels()
{
    nvec3 fpos = make_nvec3(0.1f, 0.2f, 0.3f ); 
    for(unsigned level=0 ; level < 11 ; level++)
    {
         NGrid3 g(level);
         std::cout << g.desc() 
                   << g.desc( fpos, " fpos " ) 
                   << std::endl ; 
         
    }
}


