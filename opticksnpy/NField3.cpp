#include <sstream>

#include "NField3.hpp"
#include "PLOG.hh"


NField3::NField3( F* f, const nvec3& min, const nvec3& max )
    :
    f(f),
    min(min),
    max(max),
    side(make_nvec3(max.x - min.x, max.y - min.y, max.z - min.z))
{
}

std::string NField3::desc()
{
    std::stringstream ss ;  
    ss << "NField3"
       << " min " << min.desc()
       << " max " << min.desc()
       << " side "  << side.desc()
       ;
    return ss.str();
}



nvec3 NField3::pos( const nvec3& fpos ) const
{
    return make_nvec3( min.x + fpos.x*side.x , min.y + fpos.y*side.y , min.z + fpos.z*side.z ) ;
}

float NField3::operator()( const nvec3& fpos ) const 
{
    return (*f)( min.x + fpos.x*side.x , min.y + fpos.y*side.y, min.z + fpos.z*side.z ) ;
}








