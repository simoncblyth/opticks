#include <sstream>
#include <bitset>

#include "NField3.hpp"
#include "NGLM.hpp"
#include "PLOG.hh"

template <typename FVec, typename IVec, int DIM>
const FVec NField<FVec,IVec,DIM>::ZOFFSETS[] = 
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


template <typename FVec, typename IVec, int DIM>
NField<FVec,IVec,DIM>::NField( FN* f, const FVec& min, const FVec& max )
    :
    f(f),
    min(min),
    max(max),
    side(max - min)
{
}

template <typename FVec, typename IVec, int DIM>
std::string NField<FVec,IVec,DIM>::desc()
{
    std::stringstream ss ;  
    ss << "NField"
       << " min (" 
       << std::setw(5) << min.x
       << std::setw(5) << min.y
       << std::setw(5) << min.z
       << ")"
       << " max " 
       << std::setw(5) << max.x
       << std::setw(5) << max.y
       << std::setw(5) << max.z
       << ")"
       << " side "  
       << std::setw(5) << side.x
       << std::setw(5) << side.y
       << std::setw(5) << side.z
       << ")"
       ;
    return ss.str();
}

template <typename FVec, typename IVec, int DIM>
FVec NField<FVec,IVec,DIM>::position( const FVec& fpos ) const
{
    return FVec( min.x + fpos.x*side.x , min.y + fpos.y*side.y , min.z + fpos.z*side.z ) ;
}

template <typename FVec, typename IVec, int DIM>
float NField<FVec,IVec,DIM>::operator()( const FVec& fpos ) const 
{
    return (*f)( min.x + fpos.x*side.x , min.y + fpos.y*side.y, min.z + fpos.z*side.z ) ;
}

template <typename FVec, typename IVec, int DIM>
int NField<FVec,IVec,DIM>::zcorners( const FVec& fpos , float fdelta ) const 
{
    int corners = 0;
    for(int i=0 ; i < ZCORNER ; i++)
    {
        const FVec cpos = fpos + ZOFFSETS[i]*fdelta ; 
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



template struct NPY_API NField<glm::vec3, glm::ivec3, 3> ; 
template struct NPY_API NField<nvec3, nivec3, 3> ; 



