//  https://stackoverflow.com/questions/485525/round-for-float-in-c
//  http://www.boost.org/doc/libs/1_65_1/libs/numeric/conversion/doc/html/boost_numericconversion/converter___function_object.html

#include <cassert>
#include <iostream>
#include <iomanip>

#include <boost/numeric/conversion/converter.hpp>

template<typename T, typename S> T round2(const S& x) {
  typedef boost::numeric::conversion_traits<T, S> Traits;
  typedef boost::numeric::def_overflow_handler OverflowHandler;
  typedef boost::numeric::RoundEven<typename Traits::source_type> Rounder;
  typedef boost::numeric::converter<T, S, Traits, OverflowHandler, Rounder> Converter;
  return Converter::convert(x);
}

/*

In [59]: vv = np.arange(0, 32767, 1, dtype=np.float64)*1./(32767./451.)

In [60]: vv
Out[60]: array([   0.    ,    0.0138,    0.0275, ...,  450.9587,  450.9725,  450.9862])


In [71]: vv[3450:3453]
Out[71]: array([ 47.4853,  47.499 ,  47.5128])

*/


void test_0()
{
    for(double v=-5. ; v < 5. ; v+= 0.1 )
    {
        int iv = round2<int, double>(v)  ; 
        std::cout 
               << std::setw(10) << v 
               << ' ' 
               << std::setw(10) << iv 
               << std::endl
               ;
    } 
}






#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)
#define iround(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

short shortnorm( float v, float center, float extent )  // static 
{
    // range of short is -32768 to 32767
    // Expect no positions out of range, as constrained by the geometry are bouncing on,
    // but getting times beyond the range eg 0.:100 ns is expected
    //  
    int inorm = iround(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
    return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
} 








template<typename T>
struct ShortCompressor
{
    short _imax ;

    T _center ; 
    T _extent ; 
    T _max ; 
    T _step ; 
    T _eps ; 

    ShortCompressor( T center, T extent )
        :
        _imax(32767),
        _center(center),
        _extent(extent),
        _max(_imax),
        _step(_extent/_max),
        _eps(0.0001)
    {
    }

    T value(int iv)
    {
        return _center + _step*T(iv) ;  
    }

    short ivalue0(T v)
    {
        return shortnorm( v, _center, _extent ) ; 
    } 

    short ivalue1(T v)
    {
        T vv = _max*(v - _center)/_extent ;
        return round2<short, T>(vv) ; 
    } 

    void dump(int i0, int i1, int h0, int h1)
    {
        assert( i0 < _imax ) ; 
        assert( i1 < _imax ) ; 

        for(int i=i0 ; i < i1 ; i++ )
        {
            for(int j=-1 ; j < 2 ; j++)
            {
                T v =  value(i) + j*_eps ; 

                short iv0 = ivalue0(v) ; 
                short iv1 = ivalue1(v) ; 
                bool highlight =  i >= h0 && i < h1 ; 

                std::cout 
                     << " i " 
                     << std::setw(10) << i 
                     << " j " 
                     << std::setw(10) << j 
                     << " "
                     << ( highlight ? "*" : " " )
                     << " v "
                     << std::setw(10) << std::fixed << v
                     << " iv0 "
                     << std::setw(10) << iv0
                     << " iv1 "
                     << std::setw(10) << iv1
                     << std::endl
                     ;

            }
            std::cout << std::endl ; 
        }
    } 


};




int main() 
{
    int d0 = 3440 ; 
    int h0 = 3450 ; 

    //ShortCompressor<double> dcomp(0., 451.); 
    //dcomp.dump( d0, d0+20, h0, h0+3 ) ; 

    ShortCompressor<float> fcomp(0., 451.); 
    fcomp.dump( d0, d0+20, h0, h0+3 ) ; 


    return 0 ; 
}
