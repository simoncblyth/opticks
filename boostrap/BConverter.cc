
#include <climits>
#include <boost/numeric/conversion/converter.hpp>
#include "BConverter.hh"

template<typename T, typename S> T BConverter::round_to_even(const S& x) 
{
    typedef boost::numeric::conversion_traits<T, S> Traits;
    typedef boost::numeric::def_overflow_handler OverflowHandler;
    typedef boost::numeric::RoundEven<typename Traits::source_type> Rounder;
    typedef boost::numeric::converter<T, S, Traits, OverflowHandler, Rounder> Converter;
    return Converter::convert(x);
}

short BConverter::shortnorm( float v, float center, float extent ) // static
{
    float fv = 32767.0f * (v - center)/extent ;
    return round_to_even<short, float>( fv ) ; 
}


#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)
#define iround(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

short BConverter::shortnorm_old( float v, float center, float extent )  // static 
{
    // range of short is -32768 to 32767
    // Expect no positions out of range, as constrained by the geometry are bouncing on,
    // but getting times beyond the range eg 0.:100 ns is expected
    //  
    int inorm = iround(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
    return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
} 


unsigned char BConverter::my__float2uint_rn_old( float f ) // static
{
    return iround(f);
}

unsigned char BConverter::my__float2uint_rn( float fv ) // static
{
    return BConverter::round_to_even<unsigned char, float>( fv ) ; 
}






template BRAP_API int   BConverter::round_to_even(const float& x);
template BRAP_API short BConverter::round_to_even(const float& x);
template BRAP_API unsigned char BConverter::round_to_even(const float& x);

