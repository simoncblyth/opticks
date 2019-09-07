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

// TEST=BConverterTest om-t

//  https://stackoverflow.com/questions/485525/round-for-float-in-c
//  http://www.boost.org/doc/libs/1_65_1/libs/numeric/conversion/doc/html/boost_numericconversion/converter___function_object.html

#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/numeric/conversion/converter.hpp>

#include "BConverter.hh"
#include "OPTICKS_LOG.hh"

/*

In [59]: vv = np.arange(0, 32767, 1, dtype=np.float64)*1./(32767./451.)

In [60]: vv
Out[60]: array([   0.    ,    0.0138,    0.0275, ...,  450.9587,  450.9725,  450.9862])


In [71]: vv[3450:3453]
Out[71]: array([ 47.4853,  47.499 ,  47.5128])

*/




template<typename T>
struct ShortCompressor
{
    short _imax ;

    T _center ; 
    T _extent ; 
    T _max ; 
    T _step ; 
    T _eps ; 
    T _half ;

    ShortCompressor( T center, T extent )
        :
        _imax(32767),
        _center(center),
        _extent(extent),
        _max(_imax),
        _step(_extent/_max),
        _eps(0.0001),
        _half(0.5)
    {
    }

    T value(int iv)
    {
        return _center + _step*(T(iv)+_half) ;  
    }

    short ivalue0(T v)
    {
        return BConverter::shortnorm_old( v, _center, _extent ) ; 
    } 

    short ivalue1(T v)
    {
        return BConverter::shortnorm( v, _center, _extent ) ; 
    } 

    T fvalue(T v)
    {
        return _max*(v - _center)/_extent ;
    }

    void dump(int i0, int i1, int h0, int h1)
    {
        assert( i0 < _imax ) ; 
        assert( i1 < _imax ) ; 

        for(int i=i0 ; i < i1 ; i++ )
        {
            T v =  value(i) ; 
            T fv = fvalue(v) ; 

            short iv0 = ivalue0(v) ; 
            short iv1 = ivalue1(v) ; 
            bool highlight =  i >= h0 && i < h1 ; 

            std::cout 
                 << " i " 
                 << std::setw(10) << i 
                 << " "
                 << ( highlight ? "*" : " " )
                 << " v "
                 << std::setw(10) << std::fixed << v
                 << " fv "
                 << std::setw(10) << std::fixed << fv
                 << " iv0 "
                 << std::setw(10) << iv0
                 << " iv1 "
                 << std::setw(10) << iv1
                 << " " << ( iv0 != iv1 ? "#######" : " " )
                 << std::endl
                 ;

        
        }
    } 


};



void test_ShortCompressor()
{
    int d0 = 3440 ; 
    int h0 = 3450 ; 

    //ShortCompressor<double> dcomp(0., 451.); 
    //dcomp.dump( d0, d0+20, h0, h0+3 ) ; 

    ShortCompressor<float> fcomp(0., 451.); 
    fcomp.dump( d0, d0+20, h0, h0+3 ) ; 
}


/**
test_BConverter_0
--------------------

Fails on macOS/clang : seems execeptions not percolating up thru the layers with clang ?
Have to directly catch where the exception is thrown, unlike on Linux.

::

    epsilon:boostrap blyth$ BConverterTest 
    libc++abi.dylib: terminating with uncaught exception of type boost::numeric::negative_overflow: bad numeric conversion: negative overflow
    Abort trap: 6

**/

void test_BConverter_0(const std::vector<float>& fvs )
{
    for(unsigned i=0 ; i < fvs.size() ; i++)
    {
        float fv = fvs[i] ; 

        unsigned char uc(0);  

        try 
        {
            uc = BConverter::my__float2uint_rn(fv ) ;
        }     
        catch( boost::numeric::negative_overflow& e  )
        {
            std::cout << e.what() << std::endl ;  
        }
        catch( boost::numeric::positive_overflow& e  )
        {
            std::cout << e.what() << std::endl ;  
        }

        LOG(info) 
            << " i " << std::setw(4) << i    
            << " fv " << std::setw(16) << std::fixed << std::setprecision(8) << fv
            << " uc " << (int)uc
            ;

    }
}

void test_BConverter_1(const std::vector<float>& fvs )
{
    for(unsigned i=0 ; i < fvs.size() ; i++)
    {
        float fv = fvs[i] ; 

        unsigned char uc(0);  
        uc = BConverter::my__float2uint_rn_kludge(fv ) ;

        LOG(info) 
            << " i " << std::setw(4) << i    
            << " fv " << std::setw(16) << std::fixed << std::setprecision(8) << fv
            << " uc " << (int)uc
            ;

    }
}


void test_BConverter()
{
    std::vector<float> fvs = { -1.5f, 0.f, 0.5f, 100.5f, 200.5f, 254.5f, 254.5f , 255.0f, 255.45f, 255.49999f,  255.5f , 256.5f , 300.5f } ; 

    //test_BConverter_0(fvs); 
    test_BConverter_1(fvs); 



}




int main(int argc, char** argv) 
{
    OPTICKS_LOG(argc, argv);

    //test_ShortCompressor(); 
   
    test_BConverter(); 

    return 0 ; 
}
