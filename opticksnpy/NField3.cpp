#include <sstream>
#include <bitset>

#include "NField3.hpp"
#include "PLOG.hh"

const nvec3 NField3::ZOFFSETS[] = 
{
	{ 0.f, 0.f, 0.f },
	{ 0.f, 0.f, 1.f },
	{ 0.f, 1.f, 0.f },
	{ 0.f, 1.f, 1.f },
	{ 1.f, 0.f, 0.f },
	{ 1.f, 0.f, 1.f },
	{ 1.f, 1.f, 0.f },
	{ 1.f, 1.f, 1.f }
};


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

int NField3::zcorners( const nvec3& fpos , float fdelta ) const 
{
    int corners = 0;
    for(int i=0 ; i < ZCORNER ; i++)
    {
        const nvec3 cpos = fpos + ZOFFSETS[i]*fdelta ; 
        const float density = (*this)(cpos);
        const int material = density < 0.f ? 1 : 0 ; 
        corners |= (material << i);
        //LOG(info) << i << " " << cpos.desc() << " density " << density ; 
    }

    /*
    LOG(info)
          << " zcorners " << corners 
          << " 0x " << std::hex << corners 
          << " 0b " << std::bitset<8>(corners) 
          << std::dec 
          ;
     */

    return corners ; 
}








