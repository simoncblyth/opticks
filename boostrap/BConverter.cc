/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <climits>
//#include <iostream>

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
    float f = (v - center)/extent ;
    return std::abs(f) > 1 ? SHRT_MIN : round_to_even<short, float>( 32767.0f * f ) ; 
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


unsigned char BConverter::my__float2uint_rn_kludge( float fv ) // static
{
    unsigned char uc(0);  
    try 
    {
        uc = BConverter::my__float2uint_rn(fv ) ;
    }     
    catch( boost::numeric::positive_overflow& e  )
    {
        //std::cout << e.what() << std::endl ;  
    }
    catch( boost::numeric::negative_overflow& e  )
    {
        //std::cout << e.what() << std::endl ;  
    }
    return uc ; 
}





template BRAP_API int   BConverter::round_to_even(const float& x);
template BRAP_API short BConverter::round_to_even(const float& x);
template BRAP_API unsigned char BConverter::round_to_even(const float& x);

