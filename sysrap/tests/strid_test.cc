// ./strid_test.sh 

#include <cassert>
#include <iostream>
#include <array>
#include <glm/glm.hpp>
#include "strid.h"

void set( glm::tvec4<uint64_t>& col3, int mode)
{
    if( mode == 0 )
    {
        col3.x = 0 ; 
        col3.y = 0 ; 
        col3.z = 0 ; 
        col3.w = 0 ; 
    }
    else if(mode == 1)
    {
        col3.x = 0xffffffffff0fffff ; 
        col3.y = 0xaaaaaaaafff0ffff ; 
        col3.z = 0xbbbbbbbbffff0fff ; 
        col3.w = 0xccccccccfffff0ff ; 
    }
    else if(mode == -1)
    {
        col3.x = -1 ; 
        col3.y = 0xffffffffffffffff ; 
        col3.z = -1 ; 
        col3.w = 0xffffffffffffffff ; 
    }
}

void test_Encode_Decode(int mode=0)
{
    std::cout << "test_Encode_Decode " << mode << std::endl ;

    glm::tmat4x4<double> tr(1.); 
    assert( strid::IsClear(tr)==true );

    glm::tvec4<uint64_t> col3 ;  
    set(col3, mode); 
     
    strid::Encode(tr, col3 ); 
    assert( strid::IsClear(tr)==false );
    
    glm::tvec4<uint64_t> col3_ ; 
    strid::Decode(tr, col3_ ); 

    for(unsigned r=0 ; r < 4 ; r++) assert( col3[r] == col3_[r] ); 

    std::cout << strid::Desc<double, uint64_t>(tr) << std::endl ; 
}


/**
Decoding 0.,0.,0.,1. is a hazard 

test_Decode_Unset<double,uint64_t>

                             0 0x                              0
                             0 0x                              0
                             0 0x                              0
           4607182418800017408 0x               3ff0000000000000

test_Decode_Unset<float, uint32_t>

                             0 0x                              0
                             0 0x                              0
                             0 0x                              0
                    1065353216 0x                       3f800000
**/


template <typename T, typename S>   // T:double/float S:uint64_t/uint32_t  
void test_Decode_Unset(const char* label)
{
    std::cout << label  << std::endl ;
    glm::tmat4x4<T> tr(1.); 
    assert( strid::IsClear(tr)==true );

    glm::tvec4<S> col3 ; 
    strid::Decode(tr, col3 ); 

    for(unsigned r=0 ; r < 4 ; r++) 
        std::cout 
            << std::setw(30) << col3[r] 
            << " 0x " << std::setw(30) << std::hex << col3[r] << std::dec 
            << std::endl 
            ; 

    std::cout << strid::Desc<T, S>(tr) << std::endl ; 
}


template void test_Decode_Unset<double, uint64_t>(const char* ) ; 
template void test_Decode_Unset<float , uint32_t>(const char* ) ; 



void test_Narrow(int mode)
{
    std::cout << "test_Narrow " << mode << std::endl ;

    glm::tmat4x4<double> src(1.); 

    glm::tvec4<uint64_t> col3 ; 
    set(col3, mode); 

    strid::Encode(src, col3 ); 
    std::cout << "src\n" << strid::Desc<double, uint64_t>(src) << std::endl ; 

    glm::tmat4x4<float>  dst(1.); 
    strid::Narrow(dst, src); 
    std::cout << "dst\n" << strid::Desc<float,  uint32_t>(dst) << std::endl ; 
}


void test_Encode_Decode()
{
    for(int mode=-1 ; mode < 2 ; mode++)
    {
        test_Encode_Decode(mode); 
        test_Narrow(mode);  
    }

    test_Decode_Unset<double,uint64_t>("test_Decode_Unset<double,uint64_t>"); 
    test_Decode_Unset<float, uint32_t>("test_Decode_Unset<float, uint32_t>"); 
}


void test_Read_()
{
    std::cout << "test_Read_" << std::endl ; 

    std::array<double, 16> src = {{ 
         0.,  1.,  2.,  3.,
         4.,  5.,  6.,  7.,
         8.,  9., 10., 11.,
        12., 13., 14., 15.  }} ; 

    glm::tmat4x4<double> d0(1.); 

    strid::Read_( d0, src.data() ); 
    std::cout << "d0\n" << strid::Desc_(d0) << std::endl ; 
}

void test_Read()
{
    std::cout << "test_Read" << std::endl ; 

    std::array<float, 16> src = {{ 
         0.f,  1.f,  2.f,  3.f,
         4.f,  5.f,  6.f,  7.f,
         8.f,  9.f, 10.f, 11.f,
        12.f, 13.f, 14.f, 15.f  }} ; 

    glm::tmat4x4<double> d0(1.); 
    glm::tmat4x4<double> d1(1.); 

    strid::Read( d0, src.data(), false ); 
    strid::Read( d1, src.data(), true  ); 

    std::cout << "d0\n" << strid::Desc_(d0) << std::endl ; 
    std::cout << "d1\n" << strid::Desc_(d1) << std::endl ; 
}


int main(int argc, char** argv)
{
    //test_Encode_Decode();  
    test_Read(); 
    test_Read_(); 

    return 0 ; 
}
