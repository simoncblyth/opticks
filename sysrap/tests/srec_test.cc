// name=srec_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name


#include <cassert>
#include <cstdio>
#include <cmath>
#include <iostream>

#include "srec.h"

void test_polw()
{
    assert( sizeof(srec) == sizeof(short4)*2 );


    float w0 = 80.f ; 
    float w1 = 800.f ; 

    float2 wd = make_float2( (w0+w1)/2.f, (w1-w0)/2.f ); 

    srec r ;

    const unsigned num = 3 ; 

    float4 ipolw[num] ;
    float4 opolw[num] ;

    ipolw[0] = make_float4( -1.f  , 0.f, 1.f,  w0  )  ;
    ipolw[1] = make_float4( -0.5f , 0.f, 0.5f, (w0+w1)/2.f )  ;
    ipolw[2] = make_float4( -0.1f , 0.f, 0.1f, w1  )  ;

    for(unsigned i=0 ; i < num ; i++)
    {
        float3& _ipol = (float3&)ipolw[i] ; 
        float3& _opol = (float3&)opolw[i] ; 

        float& _iw = ipolw[i].w ; 
        float& _ow = opolw[i].w ; 

        r.set_polarization( _ipol );
        r.get_polarization( _opol );

        r.set_wavelength( _iw, wd );
        r.get_wavelength( _ow, wd );

        std::cout << " ipolw " << ipolw[i] << " opolw " << opolw[i] << std::endl ;

        //std::cout << r.desc() << std::endl ; 
    }
}


void test_post()
{
    float4 ce = make_float4( 100.f, 100.f, 100.f, 200.f ); 
    float2 td = make_float2( 0.f, 10.f ); 

    const unsigned num = 3 ; 

    float4 ipost[num] ; 
    float4 opost[num] ; 

    ipost[0] = make_float4( ce.x - ce.w , ce.y - ce.w, ce.z - ce.w,  td.x - td.y  ) ; 
    ipost[1] = make_float4( ce.x        , ce.y       , ce.z       ,  td.x         ) ; 
    ipost[2] = make_float4( ce.x + ce.w , ce.y + ce.w, ce.z + ce.w,  td.x + td.y  ) ; 

    srec r = {} ;

    for(unsigned i=0 ; i < num ; i++)
    {
        float3& _ipos = (float3&)ipost[i] ; 
        float3& _opos = (float3&)opost[i] ; 

        r.set_position( _ipos, ce); 
        r.get_position( _opos, ce); 

        r.set_time( ipost[i].w, td ); 
        r.get_time( opost[i].w, td ); 

        std::cout << r.desc() << std::endl ; 
    }
    
    for(unsigned i=0 ; i < num ; i++)
    {
        std::cout << std::setw(3) << i << " ipost " << ipost[i] << " opost " << opost[i] << std::endl; 
    }
}


void test_FLOAT2INT_RN()
{
    for(float f=-100.f ; f <= 100.f ; f+= 10.f )
    {
        std::cout 
            << " f " << std::setw(10) << std::fixed << std::setprecision(3) << f
            << " FLOAT2INT_RN(f) "  << std::setw(5) << FLOAT2INT_RN(f)  
            << std::endl
            ; 
    }
}


int main()
{
    //test_FLOAT2INT_RN(); 
    test_polw(); 
    //test_post(); 


    return 0 ; 
}
